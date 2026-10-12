"""
"Varied Song" controller logic: settings, seed schedule, pass/fail judgement and the subgraph each attempt
expands into. Pure Python (no ComfyUI imports): the ComfyUI node in nodes.py only wires this up.

One attempt = (plan round, render index):
  plan round r      plans the base style + every variant with planner seed  plan_seed + r
  render index i    renders the arranged score with render seed  render_seed + i  (render_seed -1 = plan_seed)
An attempt that passes the variety thresholds ends the run. Otherwise the next attempt re-renders the same
arrangement (new render seed) until `renders_per_plan` renders have failed, then re-plans with the next seed.
After `max_attempts` the most varied take is returned.

The subgraph references the native YuE2 nodes by class name only (YuE2GenerateABC, YuE2GenerateMusic,
EmptyYuE2LatentAudio, ConditioningZeroOut, KSampler, VAEDecodeAudio), so the pack never imports their code.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .auto import DEFAULT_TERMS, style_variants
from .score import ScoreError, parse_key

MAX_VARIANT_SOCKETS = 16
DEFAULT_LYRICS = "[Intro]\n\n[Verse]\n\n[Pre-Chorus]\n\n[Chorus]\n\n[Verse]\n\n[Bridge]\n\n[Chorus]\n\n[Outro]"

# Planner / renderer settings from the official YuE2 templates
PLANNER = dict(mode="full", max_abc_tokens=8192, temperature=0.7, top_p=0.9, top_k=30, repetition_penalty=1.005,
               penalty_window=100)
RENDERER = dict(mode="full", temperature=1.0, top_p=0.95, top_k=100, repetition_penalty=1.2)


@dataclass
class Settings:
    style: str
    lyrics: str = DEFAULT_LYRICS
    terms: str = DEFAULT_TERMS
    target_seconds: float = 300.0
    no_vocals: bool = True
    key_lock: str = ""
    tempo_lock: float = 0.0
    plan_seed: int = 0
    render_seed: int = -1  # -1: renders use plan_seed
    steps: int = 6
    sampler_name: str = "dpmpp_2m"
    scheduler: str = "sgm_uniform"
    min_new_bars: float = 0.25
    max_flat_seconds: float = 75.0
    max_attempts: int = 3
    renders_per_plan: int = 2
    styles: list = field(default_factory=list)

    def validate(self) -> "Settings":
        if not (self.style or "").strip():
            raise ScoreError("style is empty")
        if len(self.lyrics) > 4000:
            raise ScoreError("lyrics must be at most 4000 characters")
        if not 0 <= self.plan_seed < 2 ** 63:
            raise ScoreError("plan_seed out of range")
        if not -1 <= self.render_seed < 2 ** 63:
            raise ScoreError("render_seed must be -1 (same as plan_seed) or 0..2^63-1")
        if not 30 <= self.target_seconds <= 900:
            raise ScoreError("target_seconds must be between 30 and 900")
        if not 0 <= self.min_new_bars <= 1 or not 0 < self.max_flat_seconds <= 900:
            raise ScoreError("min_new_bars must be 0..1 and max_flat_seconds 1..900")
        if self.key_lock.strip():
            parse_key(self.key_lock)
        if self.tempo_lock and not 40 <= self.tempo_lock <= 300:
            raise ScoreError("tempo_lock must be 0 (off) or 40..300 bpm")
        if not 1 <= self.max_attempts <= 12 or not 1 <= self.renders_per_plan <= 12:
            raise ScoreError("max_attempts and renders_per_plan must be 1..12")
        if not 1 <= self.steps <= 100:
            raise ScoreError("steps must be 1..100")
        for name in (self.sampler_name, self.scheduler):
            if not name.replace("_", "").isalnum() or len(name) > 40:
                raise ScoreError(f"bad sampler/scheduler name {name[:40]!r}")
        self.styles = style_variants(self.style, self.terms)[:MAX_VARIANT_SOCKETS]
        return self


def schedule(attempt: int, s: Settings) -> tuple[int, int, int]:
    """(plan round, planner seed, render seed) for attempt 0, 1, 2, ..."""
    plan_round, render_idx = divmod(attempt, s.renders_per_plan)
    render_base = s.plan_seed if s.render_seed < 0 else s.render_seed
    # YuE2GenerateABC, YuE2GenerateMusic and KSampler all take seeds up to 2^64-1
    return plan_round, (s.plan_seed + plan_round) % 2 ** 64, (render_base + render_idx) % 2 ** 64


def judge(report: dict, s: Settings) -> tuple[bool, float, list[str]]:
    """(passes, score, reasons) for one take's variety report. Score ranks takes when none pass."""
    reasons = []
    if report["new_bars"] < s.min_new_bars:
        reasons.append(f"{100 * report['new_bars']:.0f}% new bars < {100 * s.min_new_bars:.0f}%")
    if report["longest_flat_s"] > s.max_flat_seconds:
        reasons.append(f"{report['longest_flat_s']:.0f} s with nothing new > {s.max_flat_seconds:.0f} s")
    early = [r for r in report["restarts"] if r["t"] < 0.75 * report["duration_s"]]
    if early:
        reasons.append("restarts at " + ", ".join(f"{r['t']:.0f}s" for r in early))
    score = report["new_bars"] - 0.002 * max(0.0, report["longest_flat_s"] - s.max_flat_seconds) - 0.05 * len(early)
    return not reasons, round(score, 4), reasons


def summary(state: dict) -> str:
    s = state["settings"]
    lines = []
    for h in state["history"]:
        mark = "PASS" if h["passed"] else "fail"
        lines.append(f"attempt {h['attempt'] + 1}: {mark}  plan seed {h['plan_seed']}, render seed {h['render_seed']}  "
                     f"{100 * h['new_bars']:.0f}% new, {h['longest_flat_s']:.0f}s flat, {h['seconds']:.0f}s long"
                     + (f"  ({'; '.join(h['reasons'])})" if h["reasons"] else ""))
    best = state["best"]
    head = (f"{'PASSED' if best['passed'] else 'NO TAKE PASSED; returning the most varied'}: attempt {best['attempt'] + 1} "
            f"of {len(state['history'])} (thresholds: >= {100 * s['min_new_bars']:.0f}% new bars, <= "
            f"{s['max_flat_seconds']:.0f}s flat, no early restarts)")
    return "\n".join([head] + lines + [best.get("arrange_summary", "")]).strip()


# ── subgraph ──────────────────────────────────────────────────────────────

def _ids(prefix: str, attempt: int):
    return lambda name: f"{prefix}.a{attempt}.{name}"


def attempt_graph(state: dict, refs: dict, prefix: str) -> tuple[dict, str]:
    """API-format nodes for the next attempt and the id of its judge node.

    refs: links (["node_id", slot]) to model, clip, vae; 'arrange' (reuse the current plan's arrangement when this
    attempt only re-renders) and 'best_audio' (the best take so far), both optional."""
    s = Settings(**state["settings"])
    attempt = state["attempt"]
    plan_round, plan_seed, render_seed = schedule(attempt, s)
    nid = _ids(prefix, attempt)
    g: dict = {}
    arrange = refs.get("arrange")
    if arrange is None or plan_round != state.get("arranged_round"):
        abc_inputs = {}
        for i, style in enumerate(s.styles):
            g[nid(f"plan{i}")] = {"class_type": "YuE2GenerateABC", "inputs": {
                "clip": refs["clip"], "style": style, "lyrics": s.lyrics, "seed": plan_seed, **PLANNER}}
            abc_inputs[f"abc{i}"] = [nid(f"plan{i}"), 0]
        g[nid("arrange")] = {"class_type": "YuE2VariedSongArrange", "inputs": {
            **abc_inputs, "settings": json.dumps(state["settings"]), "plan_round": plan_round}}
        arrange = [nid("arrange"), 0]
        state["arranged_round"] = plan_round
    arr_id = arrange[0]
    g[nid("music")] = {"class_type": "YuE2GenerateMusic", "inputs": {
        "clip": refs["clip"], "style": s.styles[0], "lyrics": s.lyrics, "abc": [arr_id, 0], "seed": render_seed,
        "max_duration": [arr_id, 1], **RENDERER}}
    g[nid("latent")] = {"class_type": "EmptyYuE2LatentAudio", "inputs": {"seconds": [nid("music"), 1], "batch_size": 1}}
    g[nid("negative")] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": [nid("music"), 0]}}
    g[nid("sampler")] = {"class_type": "KSampler", "inputs": {
        "model": refs["model"], "seed": render_seed, "steps": s.steps, "cfg": 1.0, "sampler_name": s.sampler_name,
        "scheduler": s.scheduler, "positive": [nid("music"), 0], "negative": [nid("negative"), 0],
        "latent_image": [nid("latent"), 0], "denoise": 1.0}}
    g[nid("decode")] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": [nid("sampler"), 0], "vae": refs["vae"]}}
    judge_inputs = {
        "audio": [nid("decode"), 0], "audio_ref": [nid("decode"), 0], "abc": [arr_id, 0], "arrange_info": [arr_id, 2],
        "arrange_ref": [arr_id, 0], "state": json.dumps(state), "model": refs["model"], "clip": refs["clip"],
        "vae": refs["vae"]}
    if refs.get("best_audio"):
        judge_inputs["best_audio"] = refs["best_audio"]
        judge_inputs["best_audio_ref"] = refs["best_audio"]
    g[nid("judge")] = {"class_type": "YuE2VariedSongJudge", "inputs": judge_inputs}
    return g, nid("judge")


def new_state(s: Settings) -> dict:
    return {"settings": asdict(s), "attempt": 0, "history": [], "best": None}
