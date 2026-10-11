# YuE2 Arranger for ComfyUI

Longer, more varied, vocal-free songs from **YuE2** (the native `YuE2GenerateABC` / `YuE2GenerateMusic` nodes), plus
a report that tells you whether a render actually sounds repetitive.

YuE2's planner writes good material but often stops inventing after a minute or two and then loops. With an
instrumental prompt, the render can still add voice sounds. These nodes work on the planner's ABC score, before the
expensive render, to fix both:

| Node | What it does |
|---|---|
| **YuE2 Style Variants** | Your style prompt plus one variant per comma-separated term (`driving, funky, dark, intense, dynamic, evolving` by default). Its list output runs the planner once per style. |
| **YuE2 Auto-Arrange** | Mixes those plans into one longer score. New material from the variants goes between returns of the base plan's chorus. It avoids splices that would restart the song. Also makes the score instrumental and can transpose it. Same inputs, same score. |
| **YuE2 Make Instrumental** | Relabels `verse/chorus/bridge` sections as `instrumental/interlude` and moves sung notes to the instrument voice. |
| **YuE2 Transpose Score** | Moves every note and chord symbol by N semitones, or to a target key (`Em`). The render follows the new key. |
| **YuE2 Score Sections** / **YuE2 Arrange Score** | Manual mode: list every section with an index, then write a plan like `intro:0 chorus:2 verse:7[0:2] chorus:2 outro:5`. |
| **Song Variety Report** | Takes any `AUDIO` and reports the share of bars that bring something new, the longest stretch with nothing new, near-copies, and points where it sounds like the song starts over. |

No model weights, no extra Python packages (numpy only, which ComfyUI already has). Requires a ComfyUI recent enough
to have the YuE2 nodes (tested on 0.37.2).

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/yarharhar/comfyui-yue2-arranger
```
Restart ComfyUI. The nodes are under **audio → YuE2 arranger**. To try it, load
`example_workflows/yue2_varied_instrumental.json`.

## How it works

1. **Plan variants.** Keep the seed and lyrics fixed and append one word to the style. In our tests, words like
   *driving, funky, dark, intense* kept the key, tempo and chords, often kept the opening bars, then branched into new
   sections. Other words (*energetic, creative, high intensity*) often returned the exact same plan, and some
   (*catchy, hypnotic*) collapsed into one riff looped for 10+ minutes. Auto-Arrange skips duplicate, looping and
   mismatched plans (different key, tempo or meter), and its report says why.
2. **Mix.** Sections are cut into chunks of at most 16 bars. Auto-Arrange starts with the most developed intro, then
   alternates: the chunk with the most bars not heard yet, and every 16 bars a return of the base's chorus. It never
   places a chunk whose first two notated bars match the song's opening or the previous chunk's opening, because
   that is what an audible "the song starts over" splice sounds like. The match is exact ABC text, so a variant
   with the same riff written slightly differently can slip through; Song Variety Report's `restarts` catches it
   in the audio. Near the target it trims the last chunk to fit, then ends with the outro. If the plans hold less
   than half the target, it stops with an error rather than looping.
3. **Keep voices out.** When a section is labelled `verse`, `chorus` or `bridge`, YuE2GenerateMusic sang over it
   even when the vocal voice held only rests. "No vocals" in the style and `[instrumental]` lyrics did not stop it.
   Relabelling those sections `instrumental`/`interlude` did. Auto-Arrange does this by default. Sung notes move to
   the instrument voice bar by bar, wherever the instrument rests; where both play, the instrument's bar is kept.
4. **Lyrics are section tags only.** For example `[Intro]\n\n[Verse]\n\n[Chorus]…[Outro]`, with no words.
5. **Check the render.** Run the decoded audio through Song Variety Report. Every chunk choice was made from the
   score, so the report is how you find out what the render actually did with it.

### Wiring

```
Style ─▶ YuE2 Style Variants ─styles (list)─▶ YuE2GenerateABC ─abc (list)─▶ YuE2 Auto-Arrange ─abc──────▶ YuE2GenerateMusic
                             └─base_style──────────────────────────────────────────────────────────────▶ (style)
                                                                          └─max_duration─▶ (max_duration)
YuE2GenerateMusic ▶ EmptyYuE2LatentAudio / KSampler ▶ VAEDecodeAudio ─audio─▶ Song Variety Report
```
- **Render settings.** Give YuE2GenerateMusic the base style, the same lyrics and the same seed you planned with.
- **max_duration.** `max_duration` is a cap, not a target. Auto-Arrange outputs the planned length plus headroom
  for it.

### Song Variety Report numbers

The report splits the song into bars on a detected beat grid. It describes each bar by its harmony (12-bin chroma)
and its instrumentation (24 log bands). A bar counts as **new** when no earlier bar matches it on both.

| Field | Meaning |
|---|---|
| `new_bars` | Share of bars that are new. Our hand-picked YuE2 songs scored 32–44%. A song that loops after 70 s scored 4%. |
| `longest_flat_s` | Longest stretch with no new bar. |
| `loop_share` | Share of bars that nearly copy the bar 1, 2, 4 or 8 bars before. |
| `restarts` | Times after 0:20 that sound like the opening again. Only those in the first 75% of the song count as a problem; a reprise near the end is normal. |
| `verdict` | `repetitive` when fewer than 25% of bars are new, more than 75 s passes with nothing new, or there is an early restart. |

The thresholds come from about ten YuE2 renders, so treat the verdict as a hint, not a grade.

## Command line / Python

The same code works without ComfyUI:

```python
from yue2_arranger import auto_arrange, make_instrumental, parse, transpose_text
from yue2_arranger.variety import analyse, summary

abc, report = auto_arrange([base_abc, driving_abc, dark_abc], target_seconds=300)
print(summary(analyse(mono_float_array, 44100)))
```

Tests: `python -m pytest` (core, numpy only) and `python tools/comfy_api_check.py http://127.0.0.1:8188 song.ogg`. The
second needs a running ComfyUI with this pack installed; it runs every node through the `/prompt` API.

## Notes

- **Key conversion.** Only major and minor keys (`K:Dm`, `K:F#`) are transposed, read as written (`K:C#` has seven
  sharps). Other modes raise a clear error.
- **Tempo.** Song Variety Report treats a detected tempo above 160 BPM as half-time and says so. Set `bpm_hint`
  for genuinely fast songs.

## License

This pack's code is MIT (see `LICENSE`). It contains no YuE2 code, weights or LoRAs, and imports nothing from
YuE2 or ComfyUI: only the Python standard library and numpy. It works on the ABC text the YuE2 nodes produce and
the audio they render.

YuE2 itself is licensed separately, and those terms apply to whoever runs the model:
- the YuE code is Apache-2.0
- the YuE2 weights are CC BY-NC 4.0, with an added permission for individual creators to use and monetize outputs
  ([MODEL_LICENSE](https://github.com/multimodal-art-projection/YuE/blob/main/MODEL_LICENSE))
- companies need a commercial license from the YuE2 team to use the weights commercially

The ABC files in `tests/fixtures/` are YuE2 planner outputs (#YuE2), included as test data. YuE2's license says
generated outputs are not subject to the weights' NonCommercial restriction solely because they were generated with
YuE2.

Not affiliated with the YuE2 authors (M-A-P) or Comfy Org.
