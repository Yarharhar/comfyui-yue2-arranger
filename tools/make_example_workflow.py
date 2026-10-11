#!/usr/bin/env python3
"""
Writes example_workflows/yue2_varied_instrumental.json (ComfyUI UI format) from a running server's /object_info,
so widget orders match the installed node versions.

  python tools/make_example_workflow.py http://127.0.0.1:8199
"""
import json, os, sys, urllib.request

HOST = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8199"
INFO = json.loads(urllib.request.urlopen(HOST + "/object_info", timeout=60).read())
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "example_workflows", "yue2_varied_instrumental.json")

STYLE = "Synthwave, analog bass, arpeggiated synths, gated drums, driving, melodic, 110 BPM, A minor, instrumental"
LYRICS = "[Intro]\n\n[Verse]\n\n[Pre-Chorus]\n\n[Chorus]\n\n[Verse]\n\n[Bridge]\n\n[Chorus]\n\n[Outro]"
SEED = 42

nodes, links = [], []


def widgets(cls, overrides):
    """widgets_values in the frontend's order: every non-forceInput primitive/combo input, seeds get 'fixed' after."""
    out = []
    for group in ("required", "optional"):
        spec = INFO[cls]["input"].get(group, {})
        for name in INFO[cls]["input_order"].get(group, []):
            typ, opts = spec[name][0], (spec[name][1] if len(spec[name]) > 1 else {})
            if opts.get("forceInput") or (isinstance(typ, str) and typ not in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO")):
                continue
            if name in overrides:
                v = overrides[name]
            elif "default" in opts:
                v = opts["default"]
            elif isinstance(typ, list):
                v = typ[0]
            elif typ == "COMBO":
                v = (opts.get("options") or [""])[0]
            else:
                v = {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}[typ]
            out.append(v)
            if opts.get("control_after_generate") or name in ("seed", "noise_seed"):
                out.append("fixed")
    return out


def add(nid, cls, pos, size, overrides=None, title=None, links_in=None):
    spec = INFO[cls]
    inputs = []
    for group in ("required", "optional"):
        for name in spec["input_order"].get(group, []):
            typ, opts = spec["input"][group][name][0], (spec["input"][group][name][1] if len(spec["input"][group][name]) > 1 else {})
            t = "COMBO" if isinstance(typ, list) else typ
            is_widget = t in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO") and not opts.get("forceInput")
            if is_widget and name not in (links_in or {}):
                continue
            entry = {"name": name, "type": t, "link": None}
            if is_widget:
                entry["widget"] = {"name": name}
            inputs.append(entry)
    outputs = [{"name": n, "type": t, "links": [], **({"shape": 6} if spec["output_is_list"][i] else {})}
               for i, (n, t) in enumerate(zip(spec["output_name"], spec["output"]))]
    node = {"id": nid, "type": cls, "pos": pos, "size": size, "flags": {}, "order": len(nodes), "mode": 0,
            "inputs": inputs, "outputs": outputs, "properties": {"Node name for S&R": cls},
            "widgets_values": widgets(cls, overrides or {})}
    if title:
        node["title"] = title
    nodes.append(node)
    for name, (src, slot) in (links_in or {}).items():
        lid = len(links) + 1
        s = next(n for n in nodes if n["id"] == src)
        typ = s["outputs"][slot]["type"]
        links.append([lid, src, slot, nid, next(i for i, x in enumerate(inputs) if x["name"] == name), typ])
        s["outputs"][slot]["links"].append(lid)
        next(x for x in inputs if x["name"] == name)["link"] = lid
    return node


ckpt = next((c for c in INFO["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0] if "yue2" in c.lower()),
            "yue2_3b_int8_convrot.safetensors")
add(1, "CheckpointLoaderSimple", [20, 20], [380, 100], {"ckpt_name": ckpt}, "YuE2 checkpoint")
add(2, "LoraLoaderModelOnly", [20, 170], [380, 90], {"lora_name": "nar_lora_joint_v9_comfyui.safetensors", "strength_model": 1.0},
    "NAR mastering LoRA (optional)", {"model": (1, 0)})
add(3, "PrimitiveStringMultiline", [20, 310], [380, 140], {"value": STYLE}, "Style")
add(4, "PrimitiveStringMultiline", [20, 500], [380, 200], {"value": LYRICS}, "Lyrics (section tags only)")
add(5, "YuE2StyleVariants", [440, 310], [380, 150], {}, None, {"style": (3, 0)})
add(6, "YuE2GenerateABC", [860, 20], [400, 420], {"seed": SEED}, "Planner (runs once per style)",
    {"clip": (1, 1), "style": (5, 0), "lyrics": (4, 0)})
add(7, "YuE2AutoArrange", [1300, 20], [440, 420], {}, None, {"scores": (6, 0)})
add(8, "YuE2GenerateMusic", [1780, 20], [400, 380], {"seed": SEED}, None,
    {"clip": (1, 1), "style": (5, 1), "lyrics": (4, 0), "abc": (7, 0), "max_duration": (7, 3)})
add(9, "EmptyYuE2LatentAudio", [1780, 450], [300, 80], {}, None, {"seconds": (8, 1)})
add(10, "ConditioningZeroOut", [1780, 580], [300, 30], {}, None, {"conditioning": (8, 0)})
add(11, "KSampler", [2220, 20], [320, 300], {"seed": SEED, "steps": 6, "cfg": 1.0, "sampler_name": "dpmpp_2m",
                                             "scheduler": "sgm_uniform", "denoise": 1.0}, None,
    {"model": (2, 0), "positive": (8, 0), "negative": (10, 0), "latent_image": (9, 0)})
add(12, "VAEDecodeAudio", [2220, 370], [300, 50], {}, None, {"samples": (11, 0), "vae": (1, 2)})
add(13, "SaveAudioAdvanced" if "SaveAudioAdvanced" in INFO else "SaveAudio", [2580, 20], [320, 120],
    {"filename_prefix": "audio/yue2_varied"}, None, {"audio": (12, 0)})
add(14, "SongVarietyReport", [2580, 200], [380, 200], {}, None, {"audio": (12, 0)})

wf = {"last_node_id": len(nodes), "last_link_id": len(links), "nodes": nodes, "links": links,
      "groups": [{"id": 1, "title": "Plan once per style variant, then mix the plans", "bounding": [430, -40, 1330, 760],
                  "color": "#3f789e", "font_size": 22, "flags": {}}],
      "config": {}, "extra": {"notes": "YuE2 Style Variants -> YuE2GenerateABC (list) -> YuE2 Auto-Arrange -> YuE2GenerateMusic. "
                                       "Same seed + lyrics for every plan; Auto-Arrange relabels sections so the render "
                                       "has no voice. Song Variety Report checks the result."}, "version": 0.4}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(wf, open(OUT, "w"), indent=1)
print("wrote", os.path.relpath(OUT), len(nodes), "nodes", len(links), "links")
