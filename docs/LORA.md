# LoRA support

`examples/generate.py --lora` merges a Stable Diffusion style LoRA into the model before it
goes to the device. The Blackhole runs the ordinary SD UNet with different numbers in it.
Merging costs under a second at load and nothing per frame.

```bash
python examples/generate.py --prompt "storybook anime illustration, a moonlit crypt" \
  --lora "neonforestmist/sd15-storybook-anime-lora:storybook_anime_lora.safetensors@0.85"
```

Add `--lcm-lora` to cut 25 steps to 4 (see [Faster sampling](#faster-sampling-with-the-lcm-lora)).

`--lora SOURCE[:FILE][@SCALE]` can be repeated to stack LoRAs. `SOURCE` is a Hub repo id or a
local path. `FILE` names the `.safetensors` file inside the repo and is required for Hub repos.
`SCALE` defaults to 1.0. Python entry point: `animatediff_ttnn.lora.merge_loras`.

## How it works

The TTNN UNet is built from a CPU `UNet2DConditionModel` by `preprocess_model_parameters`.
`load_sd14_ttnn(device, loras=...)` folds each LoRA into that CPU model first, so
`W' = W + scale * (alpha / rank) * (up @ down)` is baked into the weights the device receives.
The UNet half goes through diffusers' kohya converter. The text-encoder half
(`lora_te_*` keys) is merged by `lora._merge_text_encoder`, because diffusers' own text-encoder
loader finds no matching modules on the transformers 5.x CLIP layout.

## What loads and what does not

| Kind | Loads? | Why |
|---|---|---|
| SD 1.x style LoRA, kohya format (`lora_unet_*`, `lora_te_*`) | Yes | Same layers as the SD 1.4 UNet and CLIP encoder. |
| LoRA for another base model (SDXL, Flux, Krea-2) | No. `merge_loras` raises `ValueError` | Diffusers loads the file but matches no UNet weight. The check for "changed nothing" exists because that outcome otherwise looks like success. |
| `guoyww/animatediff-motion-lora-*` (zoom, pan, tilt, roll) | Not supported | These patch the MotionAdapter's motion modules and do not touch the SD UNet. They need a separate merge into the motion adapter, which this change does not do. |

SD 1.5 LoRAs are applied to SD 1.4 weights, which is this repo's base. The layer shapes are
identical so the merge succeeds, and the three results below look as intended.

## Hardware results (2026-10-07, one Blackhole chip of a P300c pair)

8 frames, 20 steps, 512x512, seed fixed per prompt, `--device-id 0`. Every run in a row
shares prompt and seed with its baseline.

| Run | LoRA tensors merged (UNet + text encoder) | Model load | Denoise, 8 frames |
|---|---|---|---|
| Lighthouse, baseline | none | 8.0 s | 123.7 s (cold kernel compile after a chip reset) |
| Lighthouse, pixel art @1.0 | 192 + 0 | 6.2 s | 19.8 s |
| Lighthouse, storybook anime @1.0 | 192 + 72 | 6.5 s | 18.5 s |
| Crypt, baseline | none | 5.5 s | 18.4 s |
| Crypt, pixel art @1.0 | 192 + 0 | 6.2 s | 18.5 s |
| Crypt, storybook anime @0.85 | 192 + 72 | 6.3 s | 18.2 s |
| Crypt, anime @0.85 + pixel art @0.6 | 384 + 72 | 7.5 s | 18.3 s |

Warm denoise time is 2.3 to 2.5 s per frame with or without a LoRA. The 123.7 s baseline is
the first run after a chip reset and includes kernel compilation; it says nothing about LoRAs.
These are single runs, so differences of a few tenths of a second are within noise.

Observations:

* Pixel art changes the palette and edges strongly even without its trigger phrase.
* Storybook anime at 1.0 without its trigger phrase gives a warm, painterly look. At 0.85 with the
  trigger phrase the crypt becomes an illustrated dungeon with a lit candle. The author recommends 0.75 to 0.95.
* Stacking works. The stacked crypt differs from both single-LoRA crypts: it keeps the anime
  lighting and picks up the pixel-art LoRA's blocky floor and water.
* The figures in the prompts ("a pale figure waving", "a hooded figure holding a lantern") do not
  appear in any run, with or without a LoRA. SD 1.4 does not render them reliably; LoRAs do not fix that.

### Lighthouse: baseline, pixel art, storybook anime (top to bottom)

Prompt: *a candlelit abandoned lighthouse on black cliffs, thick fog creeping over the sea, a
pale figure waving from the highest window, eerie, mysterious, midnight*. Seed 1313.

![Lighthouse with no LoRA, pixel art and storybook anime](assets/lora/lighthouse-base-pixel-anime.jpg)

### Crypt: baseline, pixel art, storybook anime, both stacked (top to bottom)

Prompt: *a moonlit crypt beneath a drowned chapel, a hooded figure holding a single lantern,
whispering shadows gathering along the walls, ghostly mist, ancient runes glowing faintly,
haunting, mysterious*, with each LoRA's trigger phrase in front ("Pixel Art, ",
"storybook anime illustration, "). Seed 666.

![Crypt with no LoRA, pixel art, storybook anime and both stacked](assets/lora/crypt-base-pixel-anime-stacked.jpg)

Each image shows frames 1, 4 and 8 of the 8-frame GIF. With the default Phase 2.5 path the
frames differ little from each other.

## Checked before anything loads

`inspect_lora()` reads only the safetensors header, so it costs milliseconds and needs no device. The
CLI runs it before claiming a chip and `merge_loras` runs it before loading a model. It refuses, with the
reason, an SDXL LoRA (two text encoders, or SDXL metadata), a LoHa or LoKr file, a DoRA file, and anything
without kohya `lora_down` / `lora_up` tensors (diffusers-format and motion LoRAs). It also prints the most
common training tags from the file's `ss_tag_frequency` metadata. These are hints for a trigger phrase and
are not required; many LoRAs have none. On the files used here it reported 192 UNet and 72 text-encoder
modules for storybook anime, which matches what the merge changed, and listed `storybook anime illustration`,
the trigger phrase from the author's card.

## Does the merged LoRA survive the trip to the device?

The pictures above show a LoRA changes the output. They do not show the device computes what the CPU would
with the same merge, and the device weights are bfloat16 while a LoRA delta is small.
`scripts/lora_device_check.py` runs one UNet step (same latent, timestep 500, same text embedding) on the
CPU and on one Blackhole chip, with and without the LoRA, and compares the **effect** of the LoRA
(`with - without`) on each. The first two columns can look good even if the LoRA did nothing on the device,
because the LoRA is a small change to a large signal. The last column cannot.

| LoRA | Effect / signal (CPU) | PCC device vs CPU, no LoRA | PCC device vs CPU, with LoRA | PCC of the effect |
|---|---|---|---|---|
| Pixel art @1.0 | 0.100 | 0.9994 | 0.9995 | 0.935 |
| Storybook anime @0.85, UNet half | 0.028 | 0.9994 | 0.9995 | 0.690 |
| LCM-LoRA @1.0 (LoCon, conv targets) | 0.232 | 0.9994 | 0.9888 | 0.864 |

The effects reach the device in every case. A small effect is partly buried in bfloat16 noise: the storybook
LoRA changes the output by 2.8% of its norm and the device reproduces that change with correlation 0.69.
The LCM-LoRA changes convolution weights as well as attention, and those survive (0.86), but the overall
device-to-CPU match drops from 0.9994 to 0.9888. One seed and one timestep each, so these show that the
effect arrives. They do not measure what it costs in quality.

## Faster sampling with the LCM-LoRA

`latent-consistency/lcm-lora-sdv1-5` is a LoRA that turns the UNet into a consistency model. Merged next to
a style LoRA (the deltas add) and paired with `LCMScheduler`, it needs about 4 steps at guidance 1.5 instead
of 25 steps at 7.5. It is a rank-64 LoCon with 278 targets, including 3x3 and 1x1 convolutions and
`time_emb_proj`, so it exercises more of the merge than the two style LoRAs. Checked three ways:

* On the CPU, all 278 tensors change, and a 3x3 conv, a 1x1 conv and a linear target match
  `W + alpha/rank * up @ down` to about 1e-8.
* On the chip, the numeric check above.
* End to end, images and times below. 8 frames, seed 666, one chip.

| Run | Denoise, 8 frames | Per frame |
|---|---|---|
| Pixel art, 20 steps, guidance 7.5 | 18.6 s | 2.3 s |
| Pixel art + LCM, 4 steps, guidance 1.5 | 9.2 s | 1.2 s |
| Pixel art + LCM, 8 steps | 11.3 s | 1.4 s |
| LCM only, 4 steps | 9.2 s | 1.2 s |

End to end that is about 2x. The UNet work falls by 5x, and the remainder (VAE decode, the CPU
temporal blend, the first-call kernel load) does not shrink. Served warm, the same 4-step variant took 4.2 s
for 8 frames against 21.6 s at 25 steps.

![The same seed: pixel art at 20 steps, pixel art + LCM at 4 and 8 steps, LCM only at 4 steps](assets/lora/lcm-comparison.jpg)

The compositions differ from the 20-step result, as they do for any change of sampler. The LCM-only panel has
grid lines. They do not depend on `--temporal-alpha` (alpha 0 looks the same) and get stronger at 8 steps. With
a prompt that does not contain "Pixel Art" the 4-step output has no lines (below), so the lines come from the
prompt, which asks the base model for a pixel grid, and not from LCM or the device.

![The same seed with a neutral prompt: 20 steps at guidance 7.5, and LCM at 4 steps at guidance 1.5](assets/lora/lcm-neutral-control.jpg)

Fewer steps also means less cross-frame blending. Over the served runs, the first and last frames of a
clip differed by 43.9 (mean absolute pixel difference) with LCM at 4 steps and 19.4 with the 25-step style
variant, so LCM clips are less coherent from frame to frame. Try 8 steps, and compare against the
25-step result, before using it for a clip that has to loop. `--lcm-lora` cannot be combined with
`--lightning` or `--motion-adapter`.

## Serving LoRAs

The served path (`animatediff_ttnn.server.app`, which `tt-model serve` runs) takes a **catalog** of named
variants from the operator. Each is a full model set resident on the chip, so choosing one per request
costs nothing.

```bash
ANIMATEDIFF_LORAS='{
  "pixel":      "artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5:PixelArtRedmond15V-PixelArt-PIXARFK.safetensors",
  "pixel-fast": ["artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5:PixelArtRedmond15V-PixelArt-PIXARFK.safetensors",
                 "latent-consistency/lcm-lora-sdv1-5:pytorch_lora_weights.safetensors"]
}' python -m uvicorn --lifespan on animatediff_ttnn.server.app:app
curl -s localhost:8000/v1/videos/generations -H 'content-type: application/json' \
  -d '{"prompt": "a moonlit crypt", "lora": "pixel-fast", "num_frames": 8}'
```

* A value is one spec string or a list (stacked). Names are letters, digits, `.`, `_`, `-`; `base` is refused.
* `GET /v1/models` lists the names under `loras`. An unknown name is a 400 that lists the valid ones, and is
  refused before the device lock, so it never waits behind a generation.
* A request names a variant and never a repo id or a path. The operator decides which files the server loads.
* A variant that includes the LCM-LoRA samples with LCM and defaults to 4 steps and guidance 1.5 when the
  request leaves them out. Values the caller sends are used as sent.
* Variants share the plain model's VAE. Each costs one UNet's weights on the device, plus about 7 s at startup.
  Switching between the plain model and two variants, then back, reproduced the plain model's frames exactly.
* A bad catalog or an incompatible file fails startup, before readiness, and the chip is released.
* `prompt_schedule` (a list of `{"frame": n, "prompt": "..."}`) is accepted too and encoded with the chosen
  variant's text encoder. With keyframes at frames 0 and 7 the first and last frames differed by 82.7 against
  43.9 without a schedule.

With tt-model, pass the catalog as `tt-model serve --env ANIMATEDIFF_LORAS=...`. That was not run through
`tt-model` here, only through uvicorn on the chip. The container image needs `peft`, which this release adds
to `tt_model_package.yaml` and its lock, and `extra_code.ref` must point at a tag that contains this code.
LoRA files are fetched from the Hub at startup, so a container without network access needs them in its
Hugging Face cache first.

![The same prompt and seed served three ways: no LoRA, pixel art, storybook anime](assets/lora/served-variants.jpg)

## Which LoRAs to try next

Tried so far: two style LoRAs (pixel art, storybook anime), their stack, and the LCM-LoRA. Not tried, and
what each would show:

* **A character or concept LoRA with a trigger token.** Tests the text-encoder half and the trigger-word hint
  against a LoRA that does nothing without its token.
* **A detail slider LoRA used at negative and large scales** (the "add detail" family). The scale parser
  accepts signs, and nobody has checked that a negative scale behaves on the device.
* **A third LoRA stacked on a style plus LCM.** The CPU merge of anime, pixel art and LCM works with no leftover
  adapter layers on the container's versions; the device result and the quality of the stack are untried.
* **A SD 1.5 base.** Most LoRAs are trained on 1.5 and this repo serves 1.4. Same architecture, but a style may
  match better on 1.5.
* **LyCORIS LoHa / LoKr** files are refused with a clear message. Supporting them would mean a separate merge.
* **Motion LoRAs** (zoom, pan, roll) patch the motion adapter and need the Phase 3 path, which does not run an LCM
  sampler and is much slower.

Sources for candidate files (checked only as listings, not downloaded): the
[detail and noise-offset files](https://huggingface.co/ffxvs/lora-effects/tree/main) appear on third-party
mirrors, so check the author and license before use.

## Limits and caveats

* **Phase 3 and CPU modes.** `--lora` and `--lcm-lora` work in blackhole and sim modes with the default
  (Phase 2.5) loop. The served path takes LoRAs through the catalog above. `--mode cpu`, `--motion-adapter`
  and `session.ensure_blackhole` do not.
* **Not pinned.** A LoRA repo you name is downloaded at its default branch (`revision_for` returns
  `None` for repos this package does not pin). Only `.safetensors` files are read.
* **Merged weights are not cached.** Each run re-merges. At under a second this does not need a cache.
* **Validated on diffusers 0.39 and peft 0.21.** `requirements.txt` still allows diffusers 0.32.1;
  the merge uses `unet.fuse_lora` and `unet.unload_lora`, which were not tried there.
* **Licenses.** The pixel art LoRA is `license:other`, the anime LoRA is CreativeML OpenRAIL-M, and the
  Krea-2 LoRA (which does not load) is `license:other`. Check each before redistributing output.

## Reproduce

```bash
gozer run --chips 1 --who "claude:lora" --reason "lora demo" -- \
  python examples/generate.py --device-id 0 --frames 8 --steps 20 --seed 666 \
  --prompt "Pixel Art, a moonlit crypt beneath a drowned chapel, ghostly mist, haunting" \
  --lora "artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5:PixelArtRedmond15V-PixelArt-PIXARFK.safetensors"
```

Open one device per process. Launching three at once failed in firmware init
(`risc_firmware_initializer`) and left the chips needing a reset.
