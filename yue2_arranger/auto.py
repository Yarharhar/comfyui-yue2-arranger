"""
Auto-arrange: mix sections of several related planner scores into one longer, more varied score.

The idea (worked out by hand on YuE2 songs): plan the same seed + lyrics several times with one extra word
appended to the style ("driving", "funky", "dark", "intense", ...). Those plans usually keep the key, tempo
and chords, often share the opening bars, then branch into new material. Splicing the new material
between returns of the base score's chorus gives a long song that keeps developing instead of looping.

Rules (deterministic: same inputs, same output):
  * the first score is the base; others must match its K:/Q:/M:/L: (others are skipped with a reason),
    exact duplicates are skipped, and so are plans that collapsed into a loop (few distinct bars)
  * sections are cut into chunks of at most `max_chunk_bars` at voice-group boundaries
  * intro: the intro chunk with the most distinct bars (a developing intro beats one that just states the riff)
  * then alternate: the unused chunk with the most bars not heard yet, and every `hook_every_bars` a return of
    the base's chorus (rotating through its chunks) -- the returns are what make it feel like one song
  * never place a chunk whose first two bars equal the song's opening or the previous chunk's opening:
    that is what an audible "the song starts over" splice is
  * finish with an outro chunk when the target length is reached
"""
from __future__ import annotations

from dataclasses import dataclass

from .score import (Group, Score, ScoreError, Section, check_compatible, group_bars, has_notes, ins_bars, key_of,
                    make_instrumental, parse, semitones_to_key, set_tempo, stats, transpose_text, NATIVE_LABELS)

DEFAULT_TERMS = "driving, funky, dark, intense, dynamic, evolving"


def style_variants(style: str, terms: str = DEFAULT_TERMS, include_base: bool = True) -> list[str]:
    """The base style plus one style per comma-separated term (term appended at the end).
    Terms already in the style and duplicates are skipped; order is kept."""
    base = (style or "").strip().rstrip(",").strip()
    if len(base) > 4000 or len(terms or "") > 4000:
        raise ScoreError("style and terms must each be at most 4000 characters")
    have = {t.strip().lower() for t in base.split(",")}
    out = [base] if include_base and base else []
    seen = set()
    for t in (terms or "").split(","):
        t = " ".join(t.split())
        if not t or t.lower() in have or t.lower() in seen:
            continue
        if len(t) > 64 or any(c in t for c in "\n[]"):
            raise ScoreError(f"style term {t[:40]!r} is too long or has line breaks / brackets")
        seen.add(t.lower())
        out.append(f"{base}, {t}" if base else t)
    if not out:
        raise ScoreError("no styles: give a base style and/or terms")
    return out



@dataclass(eq=False)  # chunks are compared by identity (two scores can hold identical chunks)
class Chunk:
    source: int          # score index
    section: int         # global section index (as in describe())
    label: str
    lo: int              # group slice within the section
    hi: int
    groups: list[Group]

    @property
    def bars(self) -> int:
        return sum(group_bars(g) for g in self.groups)

    @property
    def tokens(self) -> list[str]:
        return ins_bars(self.groups)

    @property
    def opening(self) -> tuple[str, ...]:
        return tuple(t for t in self.tokens if has_notes(t))[:2]  # skip whole-bar rests


def _chunks(scores: list[Score], max_chunk_bars: int) -> list[Chunk]:
    out, gi = [], 0
    for si, sc in enumerate(scores):
        for s in sc.sections:
            lo = 0
            while lo < len(s.groups):
                hi, bars = lo, 0
                while hi < len(s.groups) and (hi == lo or bars + group_bars(s.groups[hi]) <= max_chunk_bars):
                    bars += group_bars(s.groups[hi])
                    hi += 1
                out.append(Chunk(si, gi, s.label, lo, hi, s.groups[lo:hi]))
                lo = hi
            gi += 1
    return out


def lock(texts: list[str], tempo_lock: float = 0.0, align_keys: bool = True) -> tuple[list[str], list[str]]:
    """Before mixing: optionally set every score to one tempo, and move variants planned in another key of the same
    mode (major/minor) into the base's key. Returns (texts, notes)."""
    notes, out = [], []
    base_key = None
    for i, t in enumerate(texts):
        if not isinstance(t, str) or not t.strip():
            out.append(t)
            continue
        try:
            if tempo_lock:
                t = set_tempo(t, tempo_lock)
            if i == 0:
                base_key = key_of(t)
            elif align_keys and base_key and key_of(t).replace(" ", "") != base_key.replace(" ", ""):
                n = semitones_to_key(t, base_key)
                t = transpose_text(t, n, key=base_key)
                notes.append(f"score {i}: moved {n:+d} semitones to {base_key}")
        except ScoreError as e:  # left as is; select_scores reports why it is unusable
            notes.append(f"score {i}: not aligned ({e})")
        out.append(t)
    return out, notes


def select_scores(texts: list[str], min_distinct_ratio: float = 0.08) -> tuple[list[Score], list[dict]]:
    """Parse, then keep the base and every compatible, non-duplicate, non-collapsed variant."""
    if not texts:
        raise ScoreError("no scores given")
    kept, log, seen = [], [], set()
    for i, t in enumerate(texts):
        norm = "\n".join(l.rstrip() for l in t.strip().splitlines())
        try:
            sc = parse(t)
        except ScoreError as e:
            if i == 0:
                raise
            log.append(dict(score=i, used=False, reason=f"unparseable: {e}"))
            continue
        st = stats(sc)
        reason = None
        if norm in seen:
            reason = "duplicate of an earlier score"
        elif i and (why := check_compatible(kept[0], sc)):
            reason = why
        elif i and st["distinct_ratio"] < min_distinct_ratio:
            reason = f"looping plan ({st['distinct_bars']} distinct of {st['bars']} bars)"
        seen.add(norm)
        log.append(dict(score=i, used=reason is None, reason=reason or "", warnings=sc.warnings, bars=st["bars"],
                        distinct_bars=st["distinct_bars"], key=st["key"], bpm=st["bpm"]))
        if reason is None:
            sc.input_index = i  # position in the caller's list, for reports
            kept.append(sc)
    return kept, log


def auto_arrange(texts: list[str], target_seconds: float = 300.0, max_chunk_bars: int = 16, hook_every_bars: int = 16,
                 instrumental: bool = True, move_vocals: bool = True, transpose: int = 0,
                 min_distinct_ratio: float = 0.08, key_lock: str = "", tempo_lock: float = 0.0,
                 align_keys: bool = True) -> tuple[str, dict]:
    """Mix `texts` (first = base) into one score of about `target_seconds`. key_lock (e.g. "Em") moves the result to
    that key and overrides `transpose`; tempo_lock sets every score's Q: first; align_keys moves variants planned in
    another key of the same mode into the base's key instead of skipping them."""
    if not 30 <= target_seconds <= 900:
        raise ScoreError("target_seconds must be between 30 and 900 (YuE2 renders at most 900 s)")
    if not 4 <= max_chunk_bars <= 64 or not 4 <= hook_every_bars <= 128:
        raise ScoreError("max_chunk_bars must be 4..64 and hook_every_bars 4..128")
    texts, notes = lock(texts, tempo_lock, align_keys)
    scores, log = select_scores(texts, min_distinct_ratio)
    base = scores[0]
    if stats(base)["distinct_ratio"] < min_distinct_ratio:
        notes.append("the base score itself loops (few distinct bars); the arrangement will too -- re-plan it")
    target_bars = min(target_seconds / base.seconds(1), 2000)  # Score clamps tempo/meter; this is a backstop
    chunks = _chunks(scores, max_chunk_bars)
    intros = [c for c in chunks if c.label == "intro" and c.lo == 0]
    outros = [c for c in chunks if c.label == "outro"]
    if not outros:  # no outro anywhere: the base's last chunk is how the planner ended the song
        outros = [[c for c in chunks if c.source == 0][-1]]
    hooks = [c for c in chunks if c.source == 0 and c.label == "chorus"]
    if not hooks:  # no chorus in the base: its body is the returning material
        hooks = [c for c in chunks if c.source == 0 and c.label not in ("intro", "outro")]
    material = [c for c in chunks if c.label not in ("intro", "outro") and c not in hooks and c not in outros]

    def distinct(c: Chunk) -> int:
        return len(set(c.tokens))

    intro = max(intros, key=lambda c: (distinct(c), -c.source, -c.section)) if intros else None
    if intro is None:
        intro = next(c for c in chunks if c.source == 0)
    first_outro = min(outros, key=lambda c: (c.source, c.section))  # the base's outro if it has one
    outro = [c for c in outros if c.source == first_outro.source and c.section == first_outro.section][-1]
    hooks = [c for c in hooks if c is not outro]
    material = [c for c in material if c is not intro]
    seq = [intro]
    heard = set(intro.tokens)
    opening = intro.opening
    used = {id(intro)}
    total = intro.bars
    since_hook = intro.bars  # a chorus may come straight after the intro
    hook_i = 0
    end_bars = outro.bars if outro else 0

    def restarts(c: Chunk) -> bool:
        return bool(c.opening) and (c.opening == opening or c.opening == seq[-1].opening)

    for _ in range(4 * len(chunks) + int(target_bars) + 8):  # every pick adds >= 1 bar; this bound is a backstop
        if total + end_bars >= target_bars:
            break
        pick = None
        if hooks and since_hook >= hook_every_bars and seq[-1] not in hooks:
            for k in range(len(hooks)):
                c = hooks[(hook_i + k) % len(hooks)]
                if not restarts(c):
                    pick, hook_i = c, (hook_i + k + 1) % len(hooks)
                    break
        if pick is None:
            fresh = [c for c in material if id(c) not in used and not restarts(c)]
            fresh.sort(key=lambda c: (-len(set(c.tokens) - heard), c.source, c.section, c.lo))
            if fresh and len(set(fresh[0].tokens) - heard) > 0:
                pick = fresh[0]
        if pick is None and hooks:  # out of new material: keep cycling the hooks and material, oldest first
            pool = [c for c in hooks + material if c is not seq[-1] and not restarts(c)]
            if pool:
                last_use = {id(c): i for i, c in enumerate(seq)}
                pick = min(pool, key=lambda c: (last_use.get(id(c), -1), c.source, c.section, c.lo))
                if not notes or notes[-1] != "out of new material; repeating earlier chunks":
                    notes.append("out of new material; repeating earlier chunks")
        if pick is None:
            notes.append("no chunk fits without restarting the song; stopped early")
            break
        room = target_bars - total - end_bars
        if pick.bars > room + 4 and len(pick.groups) > 1:  # last chunk: take only the groups that fit
            was_hook = pick in hooks
            keep, bars = 1, group_bars(pick.groups[0])
            while keep < len(pick.groups) and bars + group_bars(pick.groups[keep]) <= room + 4:
                bars += group_bars(pick.groups[keep]); keep += 1
            pick = Chunk(pick.source, pick.section, pick.label, pick.lo, pick.lo + keep, pick.groups[:keep])
            if was_hook:
                hooks.append(pick)
        seq.append(pick)
        used.add(id(pick))
        heard |= set(pick.tokens)
        total += pick.bars
        since_hook = 0 if pick in hooks else since_hook + pick.bars
    if outro is not seq[-1]:
        seq.append(outro)
        total += outro.bars
    if total < 0.5 * target_bars:
        raise ScoreError(f"not enough material: the usable scores give {base.seconds(total):.0f} s for a "
                         f"{target_seconds:.0f} s target (add style variants or lower the target)")

    # labels: intro / chorus for hooks / the source label (or verse) for material / outro
    sections: list[Section] = []
    plan = []
    for i, c in enumerate(seq):
        if i == 0:
            lab = "intro"
        elif c is outro and i == len(seq) - 1:
            lab = "outro"
        elif c in hooks:
            lab = "chorus"
        else:
            lab = c.label if c.label in NATIVE_LABELS and c.label not in ("intro", "outro", "chorus") else "verse"
        plan.append(f"{lab}:{c.section}[{c.lo}:{c.hi}]")
        if sections and sections[-1].label == lab:
            sections[-1].groups += list(c.groups)
        else:
            sections.append(Section(lab, list(c.groups), c.source))
    out = Score(list(base.header), sections)
    inst = dict(moved_bars=0, dropped_bars=0)
    if instrumental or move_vocals:
        out, inst = make_instrumental(out, relabel=instrumental, move_vocals=move_vocals)
    text = out.text()
    if key_lock.strip():
        n = semitones_to_key(text, key_lock)
        text = transpose_text(text, n, key=key_lock.strip())
        if n:
            notes.append(f"moved {n:+d} semitones to {key_lock.strip()}")
    elif transpose:
        text = transpose_text(text, transpose)
    t = 0.0
    order = []
    for c in seq:
        order.append(dict(start_s=round(t, 1), bars=c.bars, score=scores[c.source].input_index, section=c.section, label=c.label,
                          groups=[c.lo, c.hi], role="intro" if c is intro else "outro" if c is outro else
                          "hook" if c in hooks else "new"))
        t += base.seconds(c.bars)
    st = stats(parse(text))
    report = dict(bars=total, seconds=round(base.seconds(total), 1), target_seconds=target_seconds,
                  key=st["key"], distinct_bars=st["distinct_bars"], distinct_ratio=st["distinct_ratio"],
                  scores=log, order=order, plan=" ".join(plan), plan_scores=[s.input_index for s in scores],
                  instrumental=inst, notes=notes)
    return text, report
