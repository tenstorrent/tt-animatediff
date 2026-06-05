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

1. Create a new Space at https://huggingface.co/spaces (SDK: Gradio)
2. Copy the contents of `spaces/` into the Space repo root
3. Copy `app.py` and `animatediff_ttnn/` into the Space repo root
4. The Space runs in `sim` mode automatically (no hardware required)

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
| Sim binary path | file path | ~/sim/libttsim_bh.so | sim mode only |
