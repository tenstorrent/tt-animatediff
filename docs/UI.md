# Gradio UI

## Local launch (Blackhole hardware)

```bash
# Install UI dependency
pip install -e ".[ui]"

# Activate tt-metal env (required for blackhole/sim modes)
source ~/tt-metal/python_env/bin/activate

# Launch
python app.py
# Open http://localhost:7860
```

The UI caches the Blackhole device and loaded models across generations —
only the first generation pays the model-load cost (~7s) and kernel compilation
(~2–3 min first run, cached after).

## Local launch (CPU only — no hardware)

```bash
pip install -e ".[ui]"
python app.py
# Switch mode to "cpu" in the UI
```

## Local launch (ttsim — no hardware)

```bash
# Download ttsim binary first
mkdir -p ~/sim
wget -O ~/sim/libttsim_bh.so \
    https://github.com/tenstorrent/ttsim/releases/download/v1.7.0/libttsim_bh.so

pip install -e ".[ui]"
source ~/tt-metal/python_env/bin/activate
python app.py
# Switch mode to "sim" in the UI, set sim path if needed
```

## HuggingFace Spaces deployment

The `spaces/` directory is the Space entrypoint — `spaces/app.py` sets
`SPACE_MODE=sim` and then imports the shared `app.py` from the repo root.

**Important:** The Space repo must contain the entire tt-animatediff repository
tree (not just the contents of `spaces/`). `spaces/app.py` does
`sys.path.insert(0, Path(__file__).parent.parent)` to locate `app.py` one
level up — this only works when the full repo is present.

1. Create a new Space at https://huggingface.co/spaces (SDK: Gradio)
2. Push the full tt-animatediff repo to the Space (e.g. `git push space main`)
3. The `spaces/README.md` Space card already sets `app_file: spaces/app.py`
4. The Space detects `SPACE_MODE=sim` and runs in sim mode automatically

Note: ttsim requires a Linux x86_64 runner. Upload `libttsim_bh.so` as a
Space file or add a setup script to download it at startup.

## Parameters

| Parameter | Range | Default | Notes |
|---|---|---|---|
| Mode | cpu / blackhole / sim | blackhole | sim on HF Spaces |
| Prompt | text | — | See [Prompt Guide](../README.md#prompt-guide) |
| Negative prompt | text | standard | Excludes unwanted content |
| Frames | 2–24 | 8 | 2–4 recommended for sim |
| Steps | 4–50 | 25 | 4 recommended for sim |
| Seed | integer | 42 | -1 not supported; use any integer |
| Temporal alpha | 0.0–1.0 | 0.35 | Ignored in cpu mode |
| Lightning | checkbox | off | Switches to Euler solver; ~6× faster on CPU, same step count on Blackhole/sim |
| Lightning steps | 2 / 4 / 8 | 4 | CPU only — must match the distilled adapter checkpoint; Blackhole/sim ignores this |
| Sim binary path | file path | ~/sim/libttsim_bh.so | sim mode only |
