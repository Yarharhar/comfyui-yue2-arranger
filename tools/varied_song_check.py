#!/usr/bin/env python3
"""
Check YuE2 Varied Song's plan -> render -> judge -> retry loop on a TEST ComfyUI that has the stand-in backend
(tools/fake_yue2_backend) installed next to the pack. The stand-in renders a repetitive take for even render seeds
and a varied one for odd seeds, so every path is deterministic:

  seed 21, 2 renders/plan    attempt 1 renders seed 21 (odd) -> passes at once
  seed 10, 2 renders/plan    seed 10 fails, re-render with seed 11 passes, same plan (no re-planning)
  seed 30, 1 render/plan     every attempt renders seed 30 and fails; re-plans with seeds 31, 32; returns the best take
  seed 30, render_seed 21    plans with seed 30, renders seed 21 (odd) -> passes at once
  + key_lock Em, tempo_lock 104 reach the render.

  python tools/varied_song_check.py http://127.0.0.1:8199 FAKE_YUE2_LOG_PATH
"""
import json
import os
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from yue2_arranger import score as sc  # noqa: E402

HOST = "http://127.0.0.1:8199"
STYLE = "Synthwave, analog bass, 110 BPM, A minor, instrumental"


def call(path, data=None):
    req = urllib.request.Request(HOST + path, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def run(seed, renders_per_plan, max_attempts=3, **extra):
    g = {"ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "fake.safetensors"}},
         "song": {"class_type": "YuE2VariedSong", "inputs": {
             "model": ["ckpt", 0], "clip": ["ckpt", 1], "vae": ["ckpt", 2], "style": STYLE,
             "lyrics": "[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Outro]", "plan_seed": seed, "render_seed": -1,
             "terms": "driving, funky, dark",
             "target_seconds": 240.0, "min_new_bars": 0.25, "max_flat_seconds": 75.0, "no_vocals": True,
             "key_lock": "", "tempo_lock": 0.0, "max_attempts": max_attempts, "renders_per_plan": renders_per_plan,
             "steps": 6, "sampler_name": "dpmpp_2m", "scheduler": "sgm_uniform", **extra}}}
    for i, name in enumerate(("audio", "abc", "report", "share")):
        if name != "audio":
            g[f"p_{name}"] = {"class_type": "PreviewAny", "inputs": {"source": ["song", i]}}
    g["p_audio"] = {"class_type": "PreviewAudio", "inputs": {"audio": ["song", 0]}}
    res = call("/prompt", json.dumps({"prompt": g, "client_id": uuid.uuid4().hex}).encode())
    if res.get("node_errors"):
        sys.exit(json.dumps(res["node_errors"], indent=1)[:3000])
    pid = res["prompt_id"]
    while True:
        h = call(f"/history/{pid}")
        if pid in h and (h[pid]["status"].get("completed") or h[pid]["status"].get("status_str") == "error"):
            break
        time.sleep(0.5)
    st = h[pid]["status"]
    if st.get("status_str") == "error":
        sys.exit(json.dumps([m for m in st["messages"] if m[0] == "execution_error"], indent=1)[:3000])
    O = h[pid]["outputs"]
    return {k: O[f"p_{k}"]["text"][0] for k in ("abc", "report", "share")} | {"has_audio": bool(O["p_audio"]["audio"])}


def calls(log, since):
    with open(log) as f:
        lines = [json.loads(x) for x in f.read().splitlines()[since:]]
    return lines


def main():
    global HOST
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    HOST, log = sys.argv[1].rstrip("/"), sys.argv[2]
    open(log, "a").close()
    n0 = len(open(log).read().splitlines())

    t0 = time.time()
    a = run(21, 2)
    la = calls(log, n0)
    b = run(10, 2)
    lb = calls(log, n0 + len(la))
    c = run(30, 1)
    lc = calls(log, n0 + len(la) + len(lb))
    d = run(41, 2, key_lock="Em", tempo_lock=104.0)
    ld = calls(log, n0 + len(la) + len(lb) + len(lc))
    e = run(30, 2, render_seed=21)
    le = calls(log, n0 + len(la) + len(lb) + len(lc) + len(ld))
    print(f"ran in {time.time() - t0:.1f}s\n")
    for name, r in (("A", a), ("B", b), ("C", c), ("D", d)):
        print(f"--- {name}\n{r['report'].split(chr(10) + 'plan')[0][:900]}\n")

    renders = lambda L: [x["seed"] for x in L if x["node"] == "render"]  # noqa: E731
    plans = lambda L: sorted({x["seed"] for x in L if x["node"] == "plan"})  # noqa: E731
    checks = {
        "A passes on attempt 1": a["report"].startswith("PASSED: attempt 1 of 1"),
        "A: one render (seed 21), plans with seed 21": renders(la) == [21] and plans(la) == [21],
        "B passes on attempt 2 (re-render)": b["report"].startswith("PASSED: attempt 2 of 2"),
        "B: render seeds 10 then 11, one plan round": renders(lb) == [10, 11] and plans(lb) == [10],
        "C: none pass, best of 3 returned": c["report"].startswith("NO TAKE PASSED") and "of 3" in c["report"].split("\n")[0],
        "C: re-planned with seeds 30, 31, 32": plans(lc) == [30, 31, 32],
        "A/B/D/E share >= 25%, C below": all(float(r["share"]) >= 0.25 for r in (a, b, d, e)) and float(c["share"]) < 0.25,
        "audio returned": all(r["has_audio"] for r in (a, b, c, d, e)),
        "no vocal-song labels in the score": all(not {s.label for s in sc.parse(r["abc"]).sections} & set(sc.VOCAL_LABELS)
                                                 for r in (a, b, c, d, e)),
        "D: key lock Em reaches the render": "K:Em" in d["abc"] and all(x["key"] == "Em" for x in ld if x["node"] == "render"),
        "D: tempo lock 104": "Q:1/4=104" in d["abc"],
        "E: plans with seed 30, renders with render_seed 21, passes at once": plans(le) == [30] and renders(le) == [21]
                                                                             and e["report"].startswith("PASSED: attempt 1 of 1"),
    }
    for k, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + k)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
