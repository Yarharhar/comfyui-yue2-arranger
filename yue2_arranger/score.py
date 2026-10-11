"""
ABC score tools for YuE2 planner output (pure Python, no dependencies).

A YuE2 planner score looks like:

    X:1 / T: / M:4/4 / L:1/16 / Q:1/4=104 / V: Vocal ... / V: Ins ... / K:Dm
    % intro
    V: Vocal
    "Dm"z16|"Dm"z16|...          <- vocal line (chord symbols + notes or rests)
    V: Ins
    D2zD2zC2D2z6|...              <- instrument line
    % verse
    ...

The body is a list of sections (`% label`), each a list of *groups*: one `V: Vocal` line, its notes, one
`V: Ins` line, its notes (usually 4 bars). Everything here works on that structure and keeps any line it
does not understand untouched.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

NATIVE_LABELS = ("intro", "verse", "pre-chorus", "chorus", "bridge", "interlude", "outro", "instrumental")
VOCAL_LABELS = ("verse", "pre-chorus", "chorus", "bridge")
MAX_SCORE_CHARS = 2_000_000  # a 15-minute planner score is ~40 kB; refuse anything absurd
MAX_PLAN_ITEMS = 512
MAX_BARS = 4096              # ~2.5 hours at 104 bpm: far beyond YuE2's 900 s render cap


def _clip(s, n: int = 80) -> str:
    """Input text for an error message, shortened so a huge bad input cannot flood the UI or logs."""
    s = str(s)
    return repr(s if len(s) <= n else s[:n] + "...")


class ScoreError(ValueError):
    """The text is not a YuE2-style ABC score this module can work on."""


@dataclass
class Group:
    vocal_head: str
    vocal: str
    ins_head: str
    ins: str

    def lines(self) -> list[str]:
        return [self.vocal_head, self.vocal, self.ins_head, self.ins]


@dataclass
class Section:
    label: str
    groups: list[Group] = field(default_factory=list)
    source: int = 0  # index of the score it came from (0 = base)


@dataclass
class Score:
    header: list[str]
    sections: list[Section]
    input_index: int = 0  # set by auto.select_scores: position in the list the caller passed
    warnings: list[str] = field(default_factory=list)

    # header fields -------------------------------------------------------
    def field(self, key: str, default: str | None = None) -> str | None:
        for line in self.header:
            if line.startswith(key + ":"):
                return line[len(key) + 1:].strip()
        return default

    @property
    def bpm(self) -> float:
        """Quarter notes per minute (Q:1/8=120 is 60)."""
        q = self.field("Q", "1/4=120") or "1/4=120"
        m = re.search(r"(?:(\d{1,3})\s*/\s*(\d{1,3})\s*=\s*)?(\d{1,4}(?:\.\d{1,4})?)\s*$", q)
        if not m:
            raise ScoreError(f"unsupported tempo Q:{_clip(q)}")
        num, den = (int(m.group(1)), int(m.group(2))) if m.group(1) else (1, 4)
        bpm = float(m.group(3)) * num * 4 / max(den, 1)
        if not 20 <= bpm <= 400:
            raise ScoreError(f"tempo Q:{_clip(q)} is outside 20..400 quarter notes per minute")
        return bpm

    @property
    def beats_per_bar(self) -> float:
        num, den = _fraction(self.field("M", "4/4") or "4/4")
        beats = num * 4 / den
        if not 0.5 <= beats <= 16:
            raise ScoreError(f"meter M:{num}/{den} is outside 0.5..16 quarter notes per bar")
        return beats

    @property
    def bar_units(self) -> int:
        """Length of one bar in L: units (z16 for 4/4 with L:1/16)."""
        num, den = _fraction(self.field("M", "4/4") or "4/4")
        lnum, lden = _fraction(self.field("L", "1/8") or "1/8")
        units = Fraction(num, den) / Fraction(lnum, lden)
        if units.denominator != 1 or units < 1:
            raise ScoreError(f"bar length M:{num}/{den} is not a whole number of L:{lnum}/{lden} units")
        return int(units)

    def seconds(self, bars: float) -> float:
        return bars * self.beats_per_bar * 60.0 / self.bpm

    def text(self) -> str:
        out = list(self.header)
        for s in self.sections:
            out.append(f"% {s.label}")
            for g in s.groups:
                out.extend(g.lines())
        return "\n".join(out) + "\n"


def _fraction(s: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d{1,3})\s*/\s*(\d{1,3})\s*", s)
    if not m or int(m.group(1)) == 0 or int(m.group(2)) == 0:
        if s.strip() == "C":
            return 4, 4
        if s.strip() == "C|":
            return 2, 2
        raise ScoreError(f"unsupported fraction {_clip(s)}")
    return int(m.group(1)), int(m.group(2))


# ── parsing ───────────────────────────────────────────────────────────────

def parse(text: str) -> Score:
    if not isinstance(text, str) or not text.strip():
        raise ScoreError("empty score")
    if len(text) > MAX_SCORE_CHARS:
        raise ScoreError(f"score is {len(text)} characters; limit is {MAX_SCORE_CHARS}")
    lines = [l.rstrip("\r") for l in text.strip("\n").split("\n")]
    first = next((i for i, l in enumerate(lines) if l.startswith("% ")), None)
    if first is None:
        raise ScoreError("no '% section' comments found (is this YuE2 planner output?)")
    header = lines[:first]
    if not any(l.startswith("K:") for l in header):
        raise ScoreError("no K: (key) line in the header")
    sections: list[Section] = []
    pending: list[str] = []
    for n, line in enumerate(lines[first:], first + 1):
        if line.startswith("% "):
            if pending:
                raise ScoreError(f"line {n}: section starts in the middle of a voice group")
            sections.append(Section(label=line[2:].strip()))
            continue
        if not line.strip():
            continue
        pending.append(line)
        if len(pending) == 4:
            vh, v, ih, i = pending
            if not (vh.startswith("V:") and "Vocal" in vh and ih.startswith("V:") and "Ins" in ih):
                raise ScoreError(f"line {n}: expected 'V: Vocal' / notes / 'V: Ins' / notes, got {_clip(pending[0])}, "
                                 f"{_clip(pending[2])}")
            sections[-1].groups.append(Group(vh, v, ih, i))
            pending = []
    warnings = []
    if pending:  # the planner hit its token cap mid-group: drop the partial group
        warnings.append(f"dropped an unfinished voice group at the end ({len(pending)} of 4 lines)")
    last = next((s for s in reversed(sections) if s.groups), None)
    if last is not None:
        g = last.groups[-1]
        if not (g.vocal.rstrip().endswith("|") and g.ins.rstrip().endswith("|")):
            last.groups.pop()  # a cut-off notes line (no closing barline): the final bar is partial
            warnings.append("dropped the last voice group: its notes line was cut off mid-bar")
    for s in sections:
        for g in s.groups:
            if _count_bars(g.vocal) != _count_bars(g.ins):
                warnings.append(f"section '{s.label}': Vocal has {_count_bars(g.vocal)} bars, Ins {_count_bars(g.ins)}")
                break
    empty = [s.label for s in sections if not s.groups]
    if empty:
        warnings.append(f"ignored {len(empty)} empty section(s): {', '.join(empty)}")
    sections = [s for s in sections if s.groups]
    if not sections:
        raise ScoreError("no complete voice groups in the score")
    return Score(header, sections, warnings=warnings)


def bar_tokens(line: str) -> list[str]:
    """The bars of a notes line as stripped strings ('Z4' stays one token)."""
    return [b.strip() for b in line.split("|") if b.strip()]


MAX_REST_BARS = 512


def _rest_bars(token: str) -> int | None:
    """Bars filled by a multi-bar rest token (`Z4`, `"Em"Z4`), else None."""
    m = re.fullmatch(r"Z(\d*)", re.sub(r'"[^"]*"', "", token).strip())
    if not m:
        return None
    digits = m.group(1)
    n = int(digits) if digits and len(digits) <= 4 else (1 if not digits else 0)
    if not 1 <= n <= MAX_REST_BARS:
        raise ScoreError(f"multi-bar rest {_clip(token)} out of range (1..{MAX_REST_BARS})")
    return n


def _count_bars(line: str) -> int:
    return sum(_rest_bars(b) or 1 for b in bar_tokens(line))


def expand_bars(line: str, bar_units: int) -> list[str]:
    """Bar tokens with multi-bar rests expanded into one rest per bar (chord symbols kept on the first)."""
    out = []
    for b in bar_tokens(line):
        n = _rest_bars(b)
        if n is None:
            out.append(b)
        else:
            chords = "".join(re.findall(r'"[^"]*"', b))
            out += [chords + f"z{bar_units}"] + [f"z{bar_units}"] * (n - 1)
    return out


def group_bars(g: Group) -> int:
    return _count_bars(g.ins)


def section_bars(s: Section) -> int:
    return sum(group_bars(g) for g in s.groups)


def ins_bars(groups: list[Group]) -> list[str]:
    out: list[str] = []
    for g in groups:
        out += bar_tokens(g.ins)
    return out


def has_notes(line: str) -> bool:
    return bool(re.search(r"[A-Ga-g]", re.sub(r'"[^"]*"', "", line)))


def check_compatible(base: Score, other: Score) -> str | None:
    """None when `other` can be mixed into `base` (same key, tempo, meter, unit length), else the reason."""
    for key in ("K", "Q", "M", "L"):
        if (base.field(key) or "").replace(" ", "") != (other.field(key) or "").replace(" ", ""):
            return f"{key}: {other.field(key)!r} differs from {base.field(key)!r}"
    return None


def describe(scores: list[Score]) -> list[dict]:
    """One row per section across all scores: index (for plans), source score, label, bars, seconds."""
    rows = []
    i = 0
    for si, sc in enumerate(scores):
        for s in sc.sections:
            b = section_bars(s)
            rows.append(dict(index=i, score=si, label=s.label, groups=len(s.groups), bars=b,
                             seconds=round(sc.seconds(b), 1), distinct_bars=len(set(ins_bars(s.groups)))))
            i += 1
    return rows


def stats(score: Score) -> dict:
    bars = ins_bars([g for s in score.sections for g in s.groups])
    total = sum(section_bars(s) for s in score.sections)
    sung = sum(1 for s in score.sections for g in s.groups if has_notes(g.vocal))
    return dict(key=score.field("K"), bpm=score.bpm, bars=total, seconds=round(score.seconds(total), 1),
                distinct_bars=len(set(bars)), distinct_ratio=round(len(set(bars)) / max(1, len(bars)), 3),
                sections=[s.label for s in score.sections], sung_groups=sung)


# ── making a score instrumental ───────────────────────────────────────────

def vocals_to_ins(g: Group, bar_units: int) -> tuple[Group, int, int]:
    """Move sung notes to the Ins voice bar by bar: a sung bar replaces an Ins bar that only rests; where both
    voices play, the Ins bar is kept and the sung bar is dropped. The Vocal line keeps only its chord symbols.
    Returns (group, bars_moved, bars_dropped)."""
    if not has_notes(g.vocal):
        return g, 0, 0
    voc, ins = expand_bars(g.vocal, bar_units), expand_bars(g.ins, bar_units)
    rests = "|".join("".join(re.findall(r'"[^"]*"', b)) + f"z{bar_units}" for b in voc) + "|"
    if len(voc) != len(ins):  # misaligned voices: only move a whole line onto a silent Ins line
        if has_notes(g.ins):
            return Group(g.vocal_head, rests, g.ins_head, g.ins), 0, len([b for b in voc if has_notes(b)])
        return Group(g.vocal_head, rests, g.ins_head, re.sub(r'"[^"]*"', "", g.vocal)), len(voc), 0
    moved = dropped = 0
    new_ins = []
    for vb, ib in zip(voc, ins):
        if has_notes(vb) and not has_notes(ib):
            new_ins.append(re.sub(r'"[^"]*"', "", vb)); moved += 1
        else:
            new_ins.append(ib); dropped += has_notes(vb)
    if moved == len(voc):  # whole line moved: keep the original text (byte-identical to earlier versions)
        ins_line = re.sub(r'"[^"]*"', "", g.vocal)
    elif moved:
        ins_line = "|".join(new_ins) + "|"
    else:
        ins_line = g.ins
    return Group(g.vocal_head, rests, g.ins_head, ins_line), moved, dropped


def instrumental_labels(labels: list[str]) -> list[str]:
    """Replace vocal-song section labels with alternating instrumental/interlude (intro/outro kept).
    YuE2GenerateMusic adds voice sounds to sections labelled verse/chorus/bridge even when the Vocal voice only
    rests; with these labels it does not. Adjacent labels never repeat (YuE2's validator rejects that)."""
    out, n = [], 0
    for lab in labels:
        if lab in ("intro", "outro"):
            out.append(lab)
            continue
        new = ("instrumental", "interlude")[n % 2]
        if out and out[-1] == new:
            n += 1
            new = ("instrumental", "interlude")[n % 2]
        out.append(new)
        n += 1
    return out


def make_instrumental(score: Score, relabel: bool = True, move_vocals: bool = True) -> tuple[Score, dict]:
    moved = dropped = 0
    sections = []
    for s in score.sections:
        groups = []
        for g in s.groups:
            if move_vocals:
                g, mv, dr = vocals_to_ins(g, score.bar_units)
                moved += mv
                dropped += dr
            groups.append(g)
        sections.append(Section(s.label, groups, s.source))
    if relabel:
        for s, lab in zip(sections, instrumental_labels([s.label for s in sections])):
            s.label = lab
    merged: list[Section] = []
    for s in sections:  # YuE2 rejects two identical adjacent section labels
        if merged and merged[-1].label == s.label:
            merged[-1].groups += s.groups
        else:
            merged.append(s)
    return Score(list(score.header), merged), dict(moved_bars=moved, dropped_bars=dropped)


# ── transposition ─────────────────────────────────────────────────────────

PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
MAJOR_FIFTHS = {"C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
                "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7}
SHARP_ORDER, FLAT_ORDER = "FCGDAEB", "BEADGCF"
ACC = {"^^": 2, "^": 1, "=": 0, "_": -1, "__": -2}
ACC_MARK = {2: "^^", 1: "^", 0: "=", -1: "_", -2: "__"}
NOTE_RE = re.compile(r"(\^\^|\^|__|_|=)?([A-Ga-g])([,']*)")
KEY_RE = re.compile(r"\s*([A-G][#b]?)\s*(m|min|minor|maj|major)?\s*", re.I)


def pc_of(name: str) -> int:
    return (PC[name[0].upper()] + name[1:].count("#") - name[1:].count("b")) % 12


LETTER_FIFTHS = {"F": -1, "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5}


def parse_key(k: str) -> tuple[int, bool]:
    pc, minor, _ = parse_key_spelled(k)
    return pc, minor


def parse_key_spelled(k: str) -> tuple[int, bool, int]:
    """(tonic pitch class, minor?, fifths of the key signature AS WRITTEN: K:C# has 7 sharps, K:Db 5 flats)."""
    m = KEY_RE.fullmatch(k or "")
    if not m:
        raise ScoreError(f"unsupported key K:{k} (major and minor keys only)")
    mode = (m.group(2) or "").lower()
    minor = mode.startswith("m") and not mode.startswith("maj")
    tonic = m.group(1)[0].upper() + m.group(1)[1:]
    fifths = LETTER_FIFTHS[tonic[0]] + 7 * (tonic.count("#") - tonic.count("b")) - (3 if minor else 0)
    if not -7 <= fifths <= 7:
        raise ScoreError(f"K:{k} needs {abs(fifths)} accidentals; use its enharmonic key")
    return pc_of(tonic), minor, fifths


def key_name(pc: int, minor: bool) -> tuple[str, int]:
    """Spell the tonic so the key signature has the fewest accidentals. Returns (name, fifths)."""
    rel = (pc + 3) % 12 if minor else pc
    best = min((n for n in MAJOR_FIFTHS if pc_of(n) == rel), key=lambda n: abs(MAJOR_FIFTHS[n]))
    fifths = MAJOR_FIFTHS[best]
    names = SHARP_NAMES if fifths >= 0 else FLAT_NAMES
    return names[pc] + ("m" if minor else ""), fifths


def key_sig(fifths: int) -> dict[str, int]:
    sig = {l: 0 for l in "ABCDEFG"}
    for l in (SHARP_ORDER[:fifths] if fifths > 0 else FLAT_ORDER[:-fifths]):
        sig[l] = 1 if fifths > 0 else -1
    return sig


def _note_midi(acc, letter, octs, sig, bar_acc) -> int:
    up = letter.upper()
    octave = (4 if letter.isupper() else 5) + octs.count("'") - octs.count(",")
    if acc:
        bar_acc[(up, octave)] = ACC[acc]
    alt = bar_acc.get((up, octave), sig[up])
    return 12 * (octave + 1) + PC[up] + alt


def _spell(midi, fifths, sig, bar_acc) -> str:
    name = (SHARP_NAMES if fifths >= 0 else FLAT_NAMES)[midi % 12]
    letter, alt = name[0], name[1:].count("#") - name[1:].count("b")
    octave = (midi - PC[letter] - alt) // 12 - 1
    mark = ""
    if alt != bar_acc.get((letter, octave), sig[letter]):
        mark = ACC_MARK[alt]
        bar_acc[(letter, octave)] = alt
    return mark + (letter.lower() + "'" * (octave - 5) if octave >= 5 else letter + "," * (4 - octave))


def _transpose_chord(sym: str, n: int, fifths: int) -> str:
    names = SHARP_NAMES if fifths >= 0 else FLAT_NAMES
    return re.sub(r"(?:(?<=^)|(?<=/))([A-G][#b]?)", lambda m: names[(pc_of(m.group(1)) + n) % 12], sym)


def _transpose_line(line: str, n: int, sig_in, sig_out, fifths_out) -> str:
    out, i, bar_in, bar_out = [], 0, {}, {}
    tie = None  # (letter, octave, midi) of a note tied into the next one: its pitch carries across a barline
    while i < len(line):
        ch = line[i]
        if ch in '"!+':  # chord symbol / decoration: copied (chord roots transposed), never read as notes
            j = line.find(ch, i + 1)
            if j < 0:
                raise ScoreError(f"unterminated {ch}...{ch} in {_clip(line)}")
            body = line[i + 1:j]
            out.append(ch + (_transpose_chord(body, n, fifths_out) if ch == '"' else body) + ch)
            i = j + 1
            continue
        if ch == "|":
            bar_in, bar_out = {}, {}
            out.append(ch)
            i += 1
            continue
        m = NOTE_RE.match(line, i)
        if m:
            up = m.group(2).upper()
            octave = (4 if m.group(2).isupper() else 5) + m.group(3).count("'") - m.group(3).count(",")
            if tie and not m.group(1) and tie[:2] == (up, octave):
                midi_in = tie[2]
            else:
                midi_in = _note_midi(m.group(1), m.group(2), m.group(3), sig_in, bar_in)
            out.append(_spell(midi_in + n, fifths_out, sig_out, bar_out))
            i = m.end()
            j = i
            while j < len(line) and (line[j].isdigit() or line[j] == "/"):
                j += 1
            tie = (up, octave, midi_in) if j < len(line) and line[j] == "-" else None
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def transpose_text(text: str, semitones: int, key: str | None = None) -> str:
    """Move every note and chord symbol `semitones` and rewrite K: (works on any YuE2-style score text).
    `key` spells the new key (e.g. "C#" rather than "Db"); by default the spelling with fewer accidentals."""
    if not -24 <= semitones <= 24:
        raise ScoreError("transpose by at most two octaves")
    if not isinstance(text, str) or len(text) > MAX_SCORE_CHARS:
        raise ScoreError(f"score must be text of at most {MAX_SCORE_CHARS} characters")
    lines = text.rstrip("\n").split("\n")
    kline = next((i for i, l in enumerate(lines) if l.startswith("K:")), None)
    if kline is None:
        raise ScoreError("no K: line")
    if semitones == 0:
        return "\n".join(lines) + "\n"
    pc, minor, fifths_in = parse_key_spelled(lines[kline][2:])
    if key:
        kpc, kminor, fifths_out = parse_key_spelled(key)
        if (kpc, kminor) != ((pc + semitones) % 12, minor):
            raise ScoreError(f"K:{key} is not {semitones:+d} semitones from K:{lines[kline][2:].strip()}")
        name_out = key.strip()
    else:
        name_out, fifths_out = key_name((pc + semitones) % 12, minor)
    sig_in, sig_out = key_sig(fifths_in), key_sig(fifths_out)
    out = []
    for i, l in enumerate(lines):
        if i == kline:
            out.append("K:" + name_out)
        elif i < kline or l.startswith(("%", "V:")) or re.match(r"^[A-Za-z]:", l):
            out.append(l)
        else:
            out.append(_transpose_line(l, semitones, sig_in, sig_out, fifths_out))
    return "\n".join(out) + "\n"


def set_tempo(text: str, bpm: float) -> str:
    """Rewrite the score's tempo to `bpm` quarter notes per minute. YuE2GenerateMusic follows Q: (the render plays
    faster or slower), so this locks the tempo the planner drifted from."""
    if not 20 <= bpm <= 400:
        raise ScoreError("tempo must be 20..400 bpm")
    if not isinstance(text, str) or len(text) > MAX_SCORE_CHARS:
        raise ScoreError(f"score must be text of at most {MAX_SCORE_CHARS} characters")
    lines = text.rstrip("\n").split("\n")
    q = f"Q:1/4={bpm:g}"
    for i, l in enumerate(lines):
        if l.startswith("Q:"):
            lines[i] = q
            break
        if l.startswith("K:"):
            lines.insert(i, q)
            break
    else:
        raise ScoreError("no K: line")
    return "\n".join(lines) + "\n"


def key_of(text: str) -> str:
    for l in text.split("\n"):
        if l.startswith("K:"):
            return l[2:].strip()
    raise ScoreError("no K: line")


def semitones_to_key(text: str, target: str) -> int:
    """Smallest shift (-6..+5) that moves the score's key to `target` (same mode required)."""
    k = parse(text).field("K") or ""
    pc, minor, _ = parse_key_spelled(k)
    tpc, tminor = parse_key(target)
    if minor != tminor:
        raise ScoreError(f"score is in {k}; {target} is a different mode (major/minor)")
    d = (tpc - pc) % 12
    return d - 12 if d > 5 else d


# ── arranging ─────────────────────────────────────────────────────────────

PLAN_ITEM = re.compile(r"([a-z-]+):([\w-]+)(?:\[(\d*):(\d*)\])?")


def arrange(scores: list[Score], plan: str) -> tuple[Score, dict]:
    """Build a new score from sections of `scores` (the first one supplies the header).

    plan: space-separated `newLabel:source[lo:hi]`; source is a section index (see describe()) or a label of the
    base score (first occurrence); [lo:hi] slices its voice groups. Repeated adjacent labels are merged."""
    if not scores:
        raise ScoreError("no scores")
    base = scores[0]
    pool: list[Section] = []
    for si, sc in enumerate(scores):
        if si and (why := check_compatible(base, sc)):
            raise ScoreError(f"score {si}: {why}")
        pool += [Section(s.label, s.groups, si) for s in sc.sections]
    by_label = {}
    for s in pool:  # first occurrence wins, base score first
        by_label.setdefault(s.label, s)
    out: list[Section] = []
    items = plan.split()
    if not items:
        raise ScoreError("empty plan")
    if len(items) > MAX_PLAN_ITEMS:
        raise ScoreError(f"plan has {len(items)} items; at most {MAX_PLAN_ITEMS}")
    for item in items:
        m = PLAN_ITEM.fullmatch(item)
        if not m:
            raise ScoreError(f"bad plan item {_clip(item)} (want label:source or label:source[lo:hi])")
        new, src, lo, hi = m.groups()
        if new not in NATIVE_LABELS:
            raise ScoreError(f"{_clip(new)} is not a YuE2 section label ({', '.join(NATIVE_LABELS)})")
        if src.isdigit():
            if int(src) >= len(pool):
                raise ScoreError(f"section index {src} out of range (0..{len(pool) - 1})")
            sec = pool[int(src)]
        elif src in by_label:
            sec = by_label[src]
        else:
            raise ScoreError(f"no section {_clip(src)} (have {sorted(by_label)[:20]})")
        groups = sec.groups[int(lo) if lo else None:int(hi) if hi else None]
        if not groups:
            raise ScoreError(f"{_clip(item)} selects no groups")
        if out and out[-1].label == new:
            out[-1].groups += list(groups)
        else:
            out.append(Section(new, list(groups), sec.source))
    result = Score(list(base.header), out)
    total = sum(section_bars(s) for s in out)
    if total > MAX_BARS:
        raise ScoreError(f"the plan makes {total} bars; at most {MAX_BARS}")
    return result, dict(bars=total, seconds=round(result.seconds(total), 1))
