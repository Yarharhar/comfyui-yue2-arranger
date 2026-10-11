"""
End-to-end tests (numpy only, no ComfyUI): the score pipeline the nodes run, and the audio report.
For the nodes inside a real ComfyUI see tools/comfy_api_check.py.
"""
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from yue2_arranger import score as sc  # noqa: E402
from yue2_arranger import variety  # noqa: E402
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

    loop = notes([220 * 2 ** ((k % 4) / 12) for k in range(120)])
    varied = notes(110 * 2 ** (np.random.default_rng(1).integers(0, 24, 120) / 12))
    r_loop, r_var = variety.analyse(loop, sr, bpm_hint=120), variety.analyse(varied, sr, bpm_hint=120)
    assert r_loop["verdict"] == "repetitive" and r_var["new_bars"] > r_loop["new_bars"]
    assert r_loop["longest_flat_s"] > r_var["longest_flat_s"]
    assert "REPETITIVE" in variety.summary(r_loop)

    stereo_48k = np.stack([np.interp(np.arange(len(varied) * 48000 // sr) * sr / 48000, np.arange(len(varied)), varied)] * 2)
    assert variety.analyse(stereo_48k, 48000, bpm_hint=120)["bpm"] == pytest.approx(120, abs=0.6)
    with pytest.raises(ValueError, match="silent"):
        variety.analyse(np.zeros(sr * 30, np.float32), sr)
