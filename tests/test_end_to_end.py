"""
End-to-end tests (numpy only, no ComfyUI): the score pipeline the nodes run, the audio report, and the Varied Song
judge node's handling of untrusted state.
For the nodes inside a real ComfyUI see tools/comfy_api_check.py.
"""
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from yue2_arranger import score as sc  # noqa: E402
from yue2_arranger import loop as loops, varied as varied_song, variety  # noqa: E402
from yue2_arranger.auto import auto_arrange, style_variants  # noqa: E402

FX = os.path.join(os.path.dirname(__file__), "fixtures")


def fx(name):
    with open(os.path.join(FX, name)) as f:
        return f.read()


def test_score_pipeline():
    """Style variants -> planner scores -> auto-arrange (instrumental, transposed) -> a score YuE2 can render."""
    assert style_variants("Funk, 104 BPM", "driving, Funk, dark,, driving") == [
        "Funk, 104 BPM", "Funk, 104 BPM, driving", "Funk, 104 BPM, dark"]

    # base plan + style-word variants, plus a looping plan and a duplicate that must be skipped
    texts = [fx(n) for n in ("base.abc", "driving.abc", "funky.abc", "dark.abc", "intense.abc", "catchy.abc", "base.abc")]
    abc, rep = auto_arrange(texts, 300, transpose=2)
    assert (abc, rep) == auto_arrange(texts, 300, transpose=2)            # deterministic
    assert 270 <= rep["seconds"] <= 330                                    # near the target
    skipped = {r["score"]: r["reason"] for r in rep["scores"] if not r["used"]}
    assert "looping" in skipped[5] and "duplicate" in skipped[6]

    out = sc.parse(abc)
    st = sc.stats(out)
    assert st["key"] == "Em"                                               # Dm moved up 2
    assert st["distinct_bars"] > sc.stats(sc.parse(texts[0]))["distinct_bars"]  # more material than the base
    labels = [s.label for s in out.sections]
    assert labels[0] == "intro" and not set(labels) & set(sc.VOCAL_LABELS)  # no vocal-song labels
    assert all(a != b for a, b in zip(labels, labels[1:]))                  # YuE2 rejects repeated labels
    assert len({(o["section"], o["groups"][0]) for o in rep["order"]}) > 6  # it actually mixes chunks

    # key and tempo locks: the result lands in the asked-for key and tempo, whatever the planner chose
    locked, lrep = auto_arrange(texts, 300, key_lock="Em", tempo_lock=104)
    assert sc.parse(locked).field("K") == "Em" and "Q:1/4=104" in locked
    assert sc.set_tempo(fx("base.abc"), 96).count("Q:1/4=96") == 1
    # a variant planned in another key is moved to the base's key; one at another tempo is only usable with the lock
    drifted = [texts[0], sc.transpose_text(texts[1], 3), sc.set_tempo(texts[3], 99)]
    used = lambda r: [s["used"] for s in r["scores"]]  # noqa: E731
    assert used(auto_arrange(drifted, 240)[1]) == [True, True, False]
    assert used(auto_arrange(drifted, 240, tempo_lock=104)[1]) == [True, True, True]

    # a plan with sung notes: they move to the instrument voice, chord symbols stay
    inst, info = sc.make_instrumental(sc.parse(fx("sung.abc")))
    assert info["moved_bars"] > 0 and sc.stats(inst)["sung_groups"] == 0
    assert all('"' in g.vocal for s in inst.sections for g in s.groups)

    # transposition keeps every pitch: there and back gives the same notes
    back = sc.transpose_text(sc.transpose_text(fx("base.abc"), 5), -5)
    assert sc.parse(back).field("K") == "Dm"
    assert sc.transpose_text(back, 0) == sc.transpose_text(sc.transpose_text(back, 7), -7)

    # hostile inputs fail fast with a clear error instead of hanging the queue
    t0 = time.time()
    for bad in (texts[0].replace("Q:1/4=104", "Q:1/4=999999"), texts[0].replace("z16|", "Z0|", 1), "", "garbage"):
        with pytest.raises(sc.ScoreError):
            auto_arrange([bad], 300)
    assert time.time() - t0 < 1


def test_variety_report():
    """A looped figure reads as repetitive, a varied line as varied; silence is refused."""
    sr = 22050
    t = np.arange(sr * 60) / sr
    beat = 0.5  # 120 bpm

    def notes(pitches):
        y = np.zeros_like(t)
        for k, f in enumerate(pitches):
            m = (t >= k * beat) & (t < k * beat + 0.3)
            y[m] += np.sin(2 * np.pi * f * t[m]) * np.exp(-(t[m] - k * beat) * 8)
        return y.astype(np.float32)

    looped = notes([220 * 2 ** ((k % 4) / 12) for k in range(120)])
    varied = notes(110 * 2 ** (np.random.default_rng(1).integers(0, 24, 120) / 12))
    r_loop, r_var = variety.analyse(looped, sr, bpm_hint=120), variety.analyse(varied, sr, bpm_hint=120)
    assert r_loop["verdict"] == "repetitive" and r_var["new_bars"] > r_loop["new_bars"]
    assert r_loop["longest_flat_s"] > r_var["longest_flat_s"]
    assert "REPETITIVE" in variety.summary(r_loop)

    stereo_48k = np.stack([np.interp(np.arange(len(varied) * 48000 // sr) * sr / 48000, np.arange(len(varied)), varied)] * 2)
    assert variety.analyse(stereo_48k, 48000, bpm_hint=120)["bpm"] == pytest.approx(120, abs=0.6)
    with pytest.raises(ValueError, match="silent"):
        variety.analyse(np.zeros(sr * 30, np.float32), sr)

    # seamless loop: whole bars at the song's tempo, and the seam (end -> start) is as smooth as the inside
    y, info = loops.make_loop(varied, sr, bpm=120, first_beat_s=0.0, max_bars=8)
    assert info["bars"] == 8 and y.shape == (1, 8 * 2 * sr)
    seam = abs(float(y[0, -1]) - float(y[0, 0]))
    assert seam <= np.abs(np.diff(y[0])).max() + 1e-6

    # Varied Song's judge and seed schedule
    s = varied_song.Settings(style="Funk", plan_seed=10, renders_per_plan=2).validate()
    assert [varied_song.schedule(a, s)[1:] for a in range(4)] == [(10, 10), (10, 11), (11, 10), (11, 11)]
    s2 = varied_song.Settings(style="Funk", plan_seed=10, render_seed=500, renders_per_plan=2).validate()
    assert [varied_song.schedule(a, s2)[1:] for a in range(4)] == [(10, 500), (10, 501), (11, 500), (11, 501)]
    top = 2 ** 63 - 1  # largest accepted seed: later attempts go past it (downstream nodes take up to 2^64-1)
    s3 = varied_song.Settings(style="Funk", plan_seed=top, render_seed=top, renders_per_plan=2).validate()
    assert [varied_song.schedule(a, s3)[1:] for a in range(3)] == [(top, top), (top, top + 1), (top + 1, top)]
    for bad in (dict(plan_seed=-1), dict(plan_seed=2 ** 63), dict(render_seed=-2), dict(render_seed=2 ** 63)):
        with pytest.raises(sc.ScoreError):
            varied_song.Settings(style="Funk", **bad).validate()
    assert varied_song.judge(r_loop, s)[0] is False and varied_song.judge(r_loop, s)[2]


def _load_nodes():
    """Import the pack's nodes.py as a package module (it uses relative imports) without ComfyUI."""
    import importlib.util
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    pkg = importlib.util.spec_from_file_location("yue2_pack", os.path.join(root, "__init__.py"),
                                                 submodule_search_locations=[root])
    mod = importlib.util.module_from_spec(pkg)
    sys.modules["yue2_pack"] = mod
    pkg.loader.exec_module(mod)
    import importlib
    return importlib.import_module("yue2_pack.nodes")


def test_judge_rejects_crafted_state():
    """The internal judge node can be called straight from /prompt with any state string: settings must be
    re-validated before anything is scheduled or expanded (no node-graph or retry bombs, no unchecked sampler args)."""
    import json
    from dataclasses import asdict
    nodes = _load_nodes()
    judge = nodes.YuE2VariedSongJudge()
    good = varied_song.new_state(varied_song.Settings(style="Funk", terms="driving, dark", max_attempts=3).validate())
    bad_take = {"waveform": np.zeros((2, 2, 10), np.float32), "sample_rate": 44100}  # batch of 2: a failed attempt
    ref = ["n", 0]
    info = json.dumps({"summary": ""})

    def run(state):
        return judge.run(bad_take, ref, "X:1", info, ref, json.dumps(state), ref, ref, ref, unique_id="u")

    # a valid state that fails its take expands exactly one more attempt with at most 16 planner nodes
    out = run(good)
    g = out["expand"]
    assert sum(v["class_type"] == "YuE2GenerateABC" for v in g.values()) == 3  # base + 2 terms
    nxt = json.loads(next(v for v in g.values() if v["class_type"] == "YuE2VariedSongJudge")["inputs"]["state"])
    assert nxt["attempt"] == 1 and len(nxt["history"]) == 1

    def crafted(**over):
        st = json.loads(json.dumps(good))
        st["settings"].update(over)
        return st

    t0 = time.time()
    for st in (crafted(styles="x" * 100_000),           # validate() re-derives styles (the string would be 100k plans)
               crafted(max_attempts=10 ** 18), crafted(steps=10 ** 9), crafted(sampler_name="../x"),
               crafted(render_seed=1.5), crafted(plan_seed=True), crafted(render_seed=-5), crafted(renders_per_plan=0),
               crafted(lyrics="L" * 5000), crafted(style="F" * 3000), crafted(unknown=1),
               {**good, "attempt": 7}, {**good, "attempt": "0"}, {**good, "history": "x"}, {**good, "best": {}},
               {"settings": "x"}, [], "not json"):
        if isinstance(st, dict) and st.get("settings", {}) and "styles" in st["settings"] and \
                isinstance(st["settings"]["styles"], str):
            # styles is derived, so a crafted value is simply replaced: the run proceeds with 3 planner nodes
            g = run(st)["expand"]
            assert sum(v["class_type"] == "YuE2GenerateABC" for v in g.values()) == 3
            continue
        with pytest.raises(Exception, match="bad Varied Song state") as e:  # the package's own ScoreError class
            judge.run(bad_take, ref, "X:1", info, ref, st if isinstance(st, str) else json.dumps(st), ref, ref, ref,
                      unique_id="u")
        assert type(e.value).__name__ == "ScoreError"
    assert time.time() - t0 < 2.0
    # the stored settings are the validated ones
    assert asdict(varied_song.load_state(json.dumps(good))[1]) == good["settings"]

