"""
ComfyUI nodes for YuE2 planner scores: style variants, auto-arrange, manual arrange, instrumental, transpose,
and a variety (repetition) report for rendered audio.

Typical graph:
    YuE2 Style Variants ──styles(list)──▶ YuE2GenerateABC ──abc(list)──▶ YuE2 Auto-Arrange ──abc──▶ YuE2GenerateMusic
                                                                                    └─seconds─▶ (max_duration)
    ... ▶ VAEDecodeAudio ──audio──▶ Song Variety Report
"""
from __future__ import annotations

import json
from typing import Any

from .yue2_arranger import score as sc
from .yue2_arranger.auto import DEFAULT_TERMS, auto_arrange, style_variants
from .yue2_arranger import variety

CATEGORY = "audio/YuE2 arranger"
ABC = ("STRING", {"forceInput": True})


MAX_SCORES = 64


def _first(v: Any, default: Any = None, name: str = "input") -> Any:
    """INPUT_IS_LIST nodes get every input as a list; a setting must arrive as exactly one value."""
    if isinstance(v, list):
        if len(v) > 1:
            raise sc.ScoreError(f"{name}: got a list of {len(v)} values; connect a single value")
        return v[0] if v else default
    return v


def _scores(v) -> list[str]:
    """The score list, positions kept: an empty or non-text item stays in place so reports name the right score."""
    out = []
    for x in v if isinstance(v, list) else [v]:
        if isinstance(x, list):
            out += _scores(x)
        else:
            out.append(x if isinstance(x, str) else "")
    if not out or not out[0].strip():
        raise sc.ScoreError("the first score (the base) is empty: check the planner output")
    if len(out) > MAX_SCORES:
        raise sc.ScoreError(f"{len(out)} scores; at most {MAX_SCORES}")
    return out


def _text(abc) -> str:
    if not isinstance(abc, str) or not abc.strip():
        raise sc.ScoreError("empty ABC input (connect a YuE2GenerateABC output or a text node)")
    return abc


class YuE2StyleVariants:
    DESCRIPTION = ("One style per comma-separated term (appended to the base style), plus the base style. Feed the list "
                   "into YuE2GenerateABC with the same seed and lyrics: it plans once per style, and Auto-Arrange mixes "
                   "the plans. Terms that worked: driving, funky, dark, intense, dynamic, evolving.")
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("styles", "base_style")
    OUTPUT_IS_LIST = (True, False)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "style": ("STRING", {"multiline": True, "default": "", "tooltip": "the song's style prompt"}),
            "terms": ("STRING", {"multiline": False, "default": DEFAULT_TERMS,
                                 "tooltip": "comma-separated words; each becomes one extra plan"}),
            "include_base": ("BOOLEAN", {"default": True, "tooltip": "first plan uses the style unchanged (the base)"}),
        }}

    def run(self, style, terms, include_base):
        styles = style_variants(style, terms, include_base)
        return (styles, styles[0])


class YuE2AutoArrange:
    DESCRIPTION = ("Mix several planner scores (first = base) into one longer, more varied score: new material from the "
                   "variants between returns of the base's chorus, no splice that restarts the song, optional "
                   "instrumental clean-up and transpose. Same inputs give the same score.")
    CATEGORY = CATEGORY
    FUNCTION = "run"
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = ("abc", "report", "seconds", "max_duration")
    OUTPUT_TOOLTIPS = ("the arranged score: connect to YuE2GenerateMusic's abc", "what was used, skipped and placed where",
                       "planned length", "planned length + headroom for YuE2GenerateMusic's max_duration (a cap, not a target)")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "scores": (*ABC[:1], {**ABC[1], "tooltip": "a list of ABC scores (e.g. YuE2GenerateABC run over Style Variants)"}),
            "target_seconds": ("FLOAT", {"default": 300.0, "min": 30.0, "max": 900.0, "step": 10.0}),
            "instrumental_labels": ("BOOLEAN", {"default": True, "tooltip": "relabel verse/chorus/bridge as instrumental/"
                                                "interlude; YuE2 adds voice sounds to vocal-song labels"}),
            "vocals_to_instrument": ("BOOLEAN", {"default": True, "tooltip": "move sung notes to the instrument voice"}),
            "transpose": ("INT", {"default": 0, "min": -12, "max": 12, "tooltip": "semitones"}),
            "max_chunk_bars": ("INT", {"default": 16, "min": 4, "max": 64}),
            "hook_every_bars": ("INT", {"default": 16, "min": 4, "max": 128, "tooltip": "bars of new material between "
                                        "returns of the base's chorus"}),
        }}

    def run(self, scores, target_seconds, instrumental_labels, vocals_to_instrument, transpose, max_chunk_bars, hook_every_bars):
        texts = _scores(scores)
        abc, rep = auto_arrange(texts, float(_first(target_seconds, name="target_seconds")),
                                int(_first(max_chunk_bars, name="max_chunk_bars")),
                                int(_first(hook_every_bars, name="hook_every_bars")),
                                bool(_first(instrumental_labels, name="instrumental_labels")),
                                bool(_first(vocals_to_instrument, name="vocals_to_instrument")),
                                int(_first(transpose, name="transpose")))
        text = _auto_summary(rep)
        secs = float(rep["seconds"])
        return {"ui": {"text": [text]},
                "result": (abc, text + "\n\n" + json.dumps(rep, indent=1), secs, _headroom(secs))}


def _headroom(seconds: float) -> float:
    """Renders run ~5% past the planned length; YuE2GenerateMusic's max_duration is a cap (max 900)."""
    return float(min(900.0, round(seconds * 1.1 + 15)))


def _auto_summary(rep: dict) -> str:
    m, s = divmod(int(round(rep["seconds"])), 60)
    lines = [f"{rep['bars']} bars, {m}:{s:02d}, key {rep['key']}, {rep['distinct_bars']} distinct bars "
             f"({100 * rep['distinct_ratio']:.0f}%)"]
    used = [r for r in rep["scores"] if r["used"]]
    lines.append(f"used {len(used)} of {len(rep['scores'])} scores")
    lines += [f"  skipped score {r['score']}: {r['reason']}" for r in rep["scores"] if not r["used"]]
    for o in rep["order"]:
        lines.append(f"  {o['start_s']:6.1f}s  {o['role']:5}  score {o['score']} {o['label']} ({o['bars']} bars)")
    inst = rep["instrumental"]
    if inst["moved_bars"] or inst["dropped_bars"]:
        lines.append(f"sung notes: moved {inst['moved_bars']} bars to the instrument, dropped {inst['dropped_bars']} "
                     "(instrument already playing)")
    for r in rep["scores"]:
        lines += [f"  score {r['score']}: {w}" for w in r.get("warnings", [])]
    lines += [f"note: {n}" for n in rep["notes"]]
    return "\n".join(lines)


class YuE2ScoreSections:
    DESCRIPTION = "List every section of one or more scores with the index Arrange Score's plan uses."
    CATEGORY = CATEGORY
    FUNCTION = "run"
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("sections",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"scores": ABC}}

    def run(self, scores):
        parsed = [sc.parse(_text(t)) for t in _scores(scores)]
        rows = sc.describe(parsed)
        head = [f"score {i}: {sc.stats(p)['key']} {p.bpm:g} bpm, {sc.stats(p)['bars']} bars" for i, p in enumerate(parsed)]
        body = [f"{r['index']:3}: score {r['score']} {r['label']:<12} {r['groups']:2} groups {r['bars']:3} bars "
                f"(~{r['seconds']:.0f}s, {r['distinct_bars']} distinct)" for r in rows]
        text = "\n".join(head + body)
        return {"ui": {"text": [text]}, "result": (text,)}


class YuE2ArrangeScore:
    DESCRIPTION = ("Build a score from a plan of sections: 'intro:0 chorus:2 verse:7[0:2] chorus:2 outro:5' "
                   "(label:index or label:index[first_group:end_group]; indexes from Score Sections).")
    CATEGORY = CATEGORY
    FUNCTION = "run"
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = ("abc", "seconds", "max_duration")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "scores": ABC,
            "plan": ("STRING", {"multiline": True, "default": "intro:0 chorus:1 verse:2 chorus:1 outro:3"}),
            "instrumental_labels": ("BOOLEAN", {"default": True}),
            "vocals_to_instrument": ("BOOLEAN", {"default": True}),
            "transpose": ("INT", {"default": 0, "min": -12, "max": 12}),
        }}

    def run(self, scores, plan, instrumental_labels, vocals_to_instrument, transpose):
        parsed = [sc.parse(_text(t)) for t in _scores(scores)]
        out, info = sc.arrange(parsed, _first(plan, "", "plan"))
        out, _ = sc.make_instrumental(out, bool(_first(instrumental_labels, name="instrumental_labels")),
                                      bool(_first(vocals_to_instrument, name="vocals_to_instrument")))
        return (sc.transpose_text(out.text(), int(_first(transpose, name="transpose"))), float(info["seconds"]),
                _headroom(info["seconds"]))


class YuE2MakeInstrumental:
    DESCRIPTION = ("Keep voices out of a YuE2 render: relabel verse/chorus/bridge sections as instrumental/interlude and "
                   "move any sung notes to the instrument voice. Relabelling is the part that removed the voice in tests.")
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("abc", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"abc": ABC, "relabel": ("BOOLEAN", {"default": True}),
                             "vocals_to_instrument": ("BOOLEAN", {"default": True})}}

    def run(self, abc, relabel, vocals_to_instrument):
        parsed = sc.parse(_text(abc))
        out, info = sc.make_instrumental(parsed, relabel, vocals_to_instrument)
        return (out.text(), f"moved {info['moved_bars']} sung bars to the instrument, dropped {info['dropped_bars']} "
                            "(instrument already playing); labels: " + " ".join(s.label for s in out.sections)
                            + "".join(f"\nwarning: {w}" for w in parsed.warnings))


class YuE2TransposeScore:
    DESCRIPTION = ("Move every note and chord symbol of a score (rewrites K:). Set target_key (e.g. Em) to pick the "
                   "shift automatically; otherwise semitones is used. The render follows the new key.")
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("abc", "semitones")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"abc": ABC, "semitones": ("INT", {"default": 0, "min": -12, "max": 12}),
                             "target_key": ("STRING", {"default": "", "tooltip": "optional, e.g. Em or F#m; overrides semitones"})}}

    def run(self, abc, semitones, target_key):
        text = _text(abc)
        if target_key.strip():
            n = sc.semitones_to_key(text, target_key)
            return (sc.transpose_text(text, n, key=target_key.strip()), n)
        return (sc.transpose_text(text, int(semitones)), int(semitones))


class SongVarietyReport:
    DESCRIPTION = ("How repetitive does this audio sound? Share of bars that bring something new, the longest stretch "
                   "with nothing new, near-copies, and points where it sounds like the song starts over.")
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("summary", "verdict", "new_bar_share", "longest_flat_seconds", "report_json")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "bpm_hint": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.5, "tooltip": "0 = detect"}),
            "beats_per_bar": ("INT", {"default": 4, "min": 2, "max": 12}),
        }}

    def run(self, audio, bpm_hint, beats_per_bar):
        wav = audio["waveform"]
        if len(wav) != 1:
            raise ValueError(f"audio batch of {len(wav)}; analyse one song at a time (split the batch first)")
        x = wav[0].detach().float().cpu().numpy() if hasattr(wav, "detach") else wav[0]
        r = variety.analyse(x, int(audio["sample_rate"]), bpm_hint or None, int(beats_per_bar))
        text = variety.summary(r)
        return {"ui": {"text": [text]},
                "result": (text, r["verdict"], float(r["new_bars"]), float(r["longest_flat_s"]), json.dumps(r, indent=1))}


NODE_CLASS_MAPPINGS = {
    "YuE2StyleVariants": YuE2StyleVariants,
    "YuE2AutoArrange": YuE2AutoArrange,
    "YuE2ScoreSections": YuE2ScoreSections,
    "YuE2ArrangeScore": YuE2ArrangeScore,
    "YuE2MakeInstrumental": YuE2MakeInstrumental,
    "YuE2TransposeScore": YuE2TransposeScore,
    "SongVarietyReport": SongVarietyReport,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "YuE2StyleVariants": "YuE2 Style Variants",
    "YuE2AutoArrange": "YuE2 Auto-Arrange (more varied, longer)",
    "YuE2ScoreSections": "YuE2 Score Sections",
    "YuE2ArrangeScore": "YuE2 Arrange Score (plan)",
    "YuE2MakeInstrumental": "YuE2 Make Instrumental",
    "YuE2TransposeScore": "YuE2 Transpose Score",
    "SongVarietyReport": "Song Variety Report",
}
