#!/usr/bin/env python3
"""
Run every node of the pack through a real ComfyUI server's /prompt API (no YuE2 model needed) and check the
outputs against the core library.

  python tools/comfy_api_check.py http://127.0.0.1:8188 SONG.ogg
SONG is uploaded for the Song Variety Report node.
"""
import json, os, sys, time, urllib.request, uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from yue2_arranger import score as sc  # noqa: E402
from yue2_arranger.auto import auto_arrange  # noqa: E402

HOST = "http://127.0.0.1:8188"
FX = os.path.join(HERE, "..", "tests", "fixtures")
fx = lambda n: open(os.path.join(FX, n)).read()  # noqa: E731
SOURCES = ["base.abc", "driving.abc", "funky.abc", "dark.abc", "intense.abc", "catchy.abc", "base.abc"]
PLAN = "intro:19[0:6] chorus:2 interlude:9 chorus:4 verse:13 chorus:14 bridge:16 chorus:2 verse:1 chorus:4 outro:5"


def call(path, data=None, headers=None):
    req = urllib.request.Request(HOST + path, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def upload(path):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{os.path.basename(path)}\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n").encode() + open(path, "rb").read() + \
        f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n--{b}--\r\n".encode()
    return call("/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={b}"})["name"]



def main():
    global HOST
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    HOST, song = sys.argv[1].rstrip("/"), sys.argv[2]
    g = {}
    for i, n in enumerate(SOURCES):
        g[f"s{i}"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": fx(n)}}
    g["list"] = {"class_type": "CreateList", "inputs": {f"inputs.input{i}": [f"s{i}", 0] for i in range(len(SOURCES))}}
    g["auto"] = {"class_type": "YuE2AutoArrange", "inputs": {"scores": ["list", 0], "target_seconds": 300.0, "instrumental_labels": True,
                 "vocals_to_instrument": True, "transpose": 0, "max_chunk_bars": 16, "hook_every_bars": 16}}
    g["sections"] = {"class_type": "YuE2ScoreSections", "inputs": {"scores": ["list", 0]}}
    g["arrange"] = {"class_type": "YuE2ArrangeScore", "inputs": {"scores": ["list", 0], "plan": PLAN, "instrumental_labels": True,
                    "vocals_to_instrument": True, "transpose": 0}}
    g["sung"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": fx("sung.abc")}}
    g["inst"] = {"class_type": "YuE2MakeInstrumental", "inputs": {"abc": ["sung", 0], "relabel": True, "vocals_to_instrument": True}}
    g["tp"] = {"class_type": "YuE2TransposeScore", "inputs": {"abc": ["arrange", 0], "semitones": 0, "target_key": "Em"}}
    g["styles"] = {"class_type": "YuE2StyleVariants", "inputs": {"style": "Funk, house, 104 BPM, E minor, instrumental",
                   "terms": "driving, funky, dark", "include_base": True}}
    g["audio"] = {"class_type": "LoadAudio", "inputs": {"audio": upload(song)}}
    g["variety"] = {"class_type": "SongVarietyReport", "inputs": {"audio": ["audio", 0], "bpm_hint": 0.0, "beats_per_bar": 4}}
    outs = {"auto": 0, "auto_r": ("auto", 1), "auto_s": ("auto", 2), "sections": 0, "arrange": 0, "arrange_s": ("arrange", 1),
            "inst": 0, "inst_r": ("inst", 1), "tp": 0, "tp_n": ("tp", 1), "styles": 0, "variety": 0, "verdict": ("variety", 1),
            "new_share": ("variety", 2)}
    for k, v in outs.items():
        node, idx = (k, v) if isinstance(v, int) else v
        g[f"p_{k}"] = {"class_type": "PreviewAny", "inputs": {"source": [node, idx]}}

    t0 = time.time()
    res = call("/prompt", json.dumps({"prompt": g, "client_id": uuid.uuid4().hex}).encode(), {"Content-Type": "application/json"})
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
    text = lambda k: O[f"p_{k}"]["text"]  # noqa: E731  (list, one entry per list item)
    print(f"ran in {time.time() - t0:.1f}s")

    want_auto, rep = auto_arrange([fx(n) for n in SOURCES], 300)
    arranged, _ = sc.arrange([sc.parse(fx(n)) for n in SOURCES], PLAN)
    want_arrange = sc.make_instrumental(arranged)[0].text()
    checks = {
        "auto == core auto_arrange": text("auto") == [want_auto],
        "auto seconds": abs(float(text("auto_s")[0]) - rep["seconds"]) < 0.01,
        "auto report mentions skipped duplicate": "duplicate" in text("auto_r")[0],
        "arrange == core arrange + make_instrumental": text("arrange") == [want_arrange],
        "arrange seconds ~305": abs(float(text("arrange_s")[0]) - 304.6) < 1,
        "instrumental: no sung notes": sc.stats(sc.parse(text("inst")[0]))["sung_groups"] == 0,
        "transpose to Em": "K:Em" in text("tp")[0] and text("tp_n") == ["2"],
        "styles list": text("styles") == ["Funk, house, 104 BPM, E minor, instrumental",
                                          "Funk, house, 104 BPM, E minor, instrumental, driving",
                                          "Funk, house, 104 BPM, E minor, instrumental, funky",
                                          "Funk, house, 104 BPM, E minor, instrumental, dark"],
        "sections lists all 7 scores": text("sections")[0].count("score 6") >= 1,
        "variety verdict": text("verdict")[0] in ("varied", "repetitive"),
    }
    print(text("variety")[0])
    print(text("auto_r")[0].split("\n\n")[0])
    for k, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + k)
    sys.exit(0 if all(checks.values()) else 1)



if __name__ == "__main__":
    main()
