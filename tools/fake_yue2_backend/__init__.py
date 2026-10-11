"""
Stand-in YuE2 backend for testing YuE2 Varied Song on a CPU ComfyUI without the model. NOT part of the pack:
symlink this folder into a TEST server's custom_nodes. It overrides the native YuE2 / sampler / VAE nodes:

  YuE2GenerateABC     returns a fixture plan: tests/fixtures/<last style term>.abc (base.abc for the plain style)
  YuE2GenerateMusic   passes the score through; seconds = the score's length
  KSampler            passes through, remembers the seed
  VAEDecodeAudio      synthesises the take: an EVEN render seed gives a 2-bar loop (repetitive), an ODD seed a
                      new chord root and timbre every 16 s (varied). That makes pass/fail and the retry path deterministic.

Every planner / render call is appended to $FAKE_YUE2_LOG (one JSON per line) so tests can count them.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, "..", "..", "tests", "fixtures")
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from yue2_arranger import score as sc  # noqa: E402

SR = 44100


def log(**kw):
    path = os.environ.get("FAKE_YUE2_LOG")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(kw) + "\n")


class FakeCheckpoint:
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "run"
    CATEGORY = "testing"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ckpt_name": (["fake.safetensors"],)}}

    def run(self, ckpt_name):
        return ("fake-model", "fake-clip", "fake-vae")


class FakeGenerateABC:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("abc",)
    FUNCTION = "run"
    CATEGORY = "testing"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {k: ("*",) for k in ("clip", "style", "lyrics", "seed", "mode", "max_abc_tokens", "temperature",
                                                 "top_p", "top_k", "repetition_penalty", "penalty_window")}}

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def run(self, clip, style, lyrics, seed, **_):
        term = style.rsplit(",", 1)[-1].strip().replace(" ", "_")
        path = os.path.join(FX, f"{term}.abc")
        log(node="plan", style=style, seed=seed)
        return (open(path if os.path.exists(path) else os.path.join(FX, "base.abc")).read(),)


class FakeGenerateMusic:
    RETURN_TYPES = ("CONDITIONING", "FLOAT")
    RETURN_NAMES = ("conditioning", "seconds")
    FUNCTION = "run"
    CATEGORY = "testing"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {k: ("*",) for k in ("clip", "style", "lyrics", "abc", "seed", "mode", "max_duration",
                                                 "temperature", "top_p", "top_k", "repetition_penalty")}}

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def run(self, clip, style, lyrics, abc, seed, max_duration, **_):
        secs = min(float(max_duration), sc.stats(sc.parse(abc))["seconds"])
        log(node="render", seed=seed, seconds=secs, key=sc.key_of(abc))
        return ({"abc": abc, "seconds": secs}, secs)


class FakeEmptyLatent:
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "testing"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seconds": ("FLOAT",), "batch_size": ("INT",)}}

    def run(self, seconds, batch_size):
        return ({"seconds": seconds},)


class FakeZeroOut:
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "run"
    CATEGORY = "testing"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",)}}

    def run(self, conditioning):
        return (conditioning,)


class FakeKSampler:
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "testing"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {k: ("*",) for k in ("model", "seed", "steps", "cfg", "sampler_name", "scheduler", "positive",
                                                 "negative", "latent_image", "denoise")}}

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def run(self, model, seed, positive, latent_image, **_):
        return ({"seed": int(seed), "seconds": latent_image["seconds"]},)


class FakeDecode:
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "run"
    CATEGORY = "testing"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",)}}

    def run(self, samples, vae):
        secs, seed = samples["seconds"], samples["seed"]
        t = np.arange(int(SR * secs)) / SR
        beat = 0.5
        rng = np.random.default_rng(seed)
        y = np.zeros_like(t)
        for k in range(int(secs / beat) + 1):
            if seed % 2 == 0:  # the same 4-note figure for the whole song
                f, nh = 220 * 2 ** ((k % 4) / 12), 1
            elif k < 32:  # an opening that is never heard again: high, bright, major 7th
                f, nh = 440 * 2 ** ([0, 4, 7, 11][k % 4] / 12), 8
            else:  # then a new chord root and timbre every 16 s, random chord tones within it
                sec = k // 32 - 1
                root = [5, 3, 7, 10, 2, 8, 1][sec % 7]
                f = 110 * 2 ** ((root + [0, 3, 7, 10][rng.integers(0, 4)] + 12 * rng.integers(0, 2)) / 12)
                nh = 1 + sec % 5
            m = (t >= k * beat) & (t < k * beat + 0.3)
            env = np.exp(-(t[m] - k * beat) * 8)
            y[m] += sum(np.sin(2 * np.pi * f * h * t[m]) / h for h in range(1, nh + 1)) * env
        wav = torch.tensor(np.stack([y, y]), dtype=torch.float32)[None]
        return ({"waveform": wav, "sample_rate": SR},)


FAKES = {
    "CheckpointLoaderSimple": FakeCheckpoint,
    "YuE2GenerateABC": FakeGenerateABC,
    "YuE2GenerateMusic": FakeGenerateMusic,
    "EmptyYuE2LatentAudio": FakeEmptyLatent,
    "ConditioningZeroOut": FakeZeroOut,
    "KSampler": FakeKSampler,
    "VAEDecodeAudio": FakeDecode,
}

# ComfyUI ignores custom-node entries that shadow built-in names, so replace them in the registry directly
# (test servers only).
import nodes as _comfy_nodes  # noqa: E402

_comfy_nodes.NODE_CLASS_MAPPINGS.update(FAKES)
NODE_CLASS_MAPPINGS = {}
