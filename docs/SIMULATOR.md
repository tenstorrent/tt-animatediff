# Running on ttsim — Blackhole Simulator

[ttsim](https://github.com/tenstorrent/ttsim) is Tenstorrent's full-system simulator
for Wormhole, Blackhole, and Quasar. It exports a single `libttsim_bh.so` that
TT-Metalium loads instead of the real PCIe driver, giving you a virtual Blackhole
device on any Linux/x86_64 machine.

**No Tenstorrent hardware required.** Useful for:
- CI pipelines that need to verify TTNN kernel correctness
- Development on machines without a Blackhole board
- Demonstrating the pipeline end-to-end at a conference or in a walkthrough

ttsim is **bit-exact** with silicon for supported operations. Output GIFs produced
by the simulator are numerically identical to hardware output at the same seed.

---

## Performance expectations

The simulator is roughly 10–100× slower than silicon depending on the operation.
A single TTNN UNet forward pass that takes ~15 s on a P300C takes several minutes
on the simulator. Plan accordingly:

| Mode | Frames | Steps | Approx. wall-clock |
|---|---|---|---|
| Smoke test | 2 | 4 | ~15–45 min (machine-dependent) |
| Short run | 4 | 8 | ~1–2 h |
| Full run (silicon parity) | 8 | 25 | ~4–8 h |

For CI, use 2 frames × 4 steps with a timeout. For exploration, use whatever your
patience allows.

---

## Setup

### 1. Build tt-metal (one-time)

```bash
git clone https://github.com/tenstorrent/tt-metal.git ~/tt-metal
cd ~/tt-metal
./build_metal.sh
source python_env/bin/activate
```

### 2. Download the ttsim Blackhole binary

Get the latest release from <https://github.com/tenstorrent/ttsim/releases>.

```bash
TTSIM_VERSION="v1.7.0"   # check releases page for latest
mkdir -p ~/sim

wget -O ~/sim/libttsim_bh.so \
    https://github.com/tenstorrent/ttsim/releases/download/${TTSIM_VERSION}/libttsim_bh.so

# ttsim requires its SOC descriptor to be in the same directory as the .so
cp ~/tt-metal/tt_metal/soc_descriptors/blackhole_140_arch.yaml \
    ~/sim/soc_descriptor.yaml
```

> The `.so` and `soc_descriptor.yaml` **must be in the same directory**.
> ttsim locates the descriptor relative to the library's path at runtime.

### 3. Install this project

```bash
source ~/tt-metal/python_env/bin/activate
pip install -e /path/to/tt-animatediff[dev]
```

### 4. Download model weights

```bash
cd ~/tt-animatediff
bash weights/download_weights.sh
# or: huggingface-cli download CompVis/stable-diffusion-v1-4
```

---

## Running the simulator example

`examples/generate.py --mode sim` (or the `generate_sim.py` shim) runs the same
TTNN pipeline as the Blackhole hardware path, redirected to a ttsim virtual device.
The script handles all required env vars automatically; no `TT_METAL_*` prefixing
in your shell is required.

```bash
# Quick smoke test (2 frames × 4 steps — shim default)
python examples/generate_sim.py

# Equivalent using generate.py directly
python examples/generate.py --mode sim --frames 2 --steps 4

# Custom prompt, more frames
python examples/generate.py --mode sim \
    --prompt "neon city rain at midnight, cyberpunk aesthetic" \
    --frames 4 --steps 8 \
    --output output/sim_4frame.gif

# Supply ttsim path as a flag instead of env var
python examples/generate.py --mode sim \
    --sim ~/sim/libttsim_bh.so \
    --frames 2 --steps 4

# Full silicon-parity run (slow)
python examples/generate.py --mode sim \
    --frames 8 --steps 25 \
    --output output/sim_full.gif
```

---

## Required environment variables

| Variable | Value | Why |
|---|---|---|
| `TT_METAL_SIMULATOR` | path to `libttsim_bh.so` | Activates simulator — TT-Metalium checks this before opening PCIe |
| `TT_METAL_SLOW_DISPATCH_MODE` | `1` | Fast dispatch on the simulator has not been characterized for determinism |
| `TT_METAL_DISABLE_SFPLOADMACRO` | `1` | SFPLOADMACRO is not implemented in the ttsim SFPU |

`generate.py --mode sim` sets `SLOW_DISPATCH_MODE` and `DISABLE_SFPLOADMACRO` automatically
as defaults. They can be overridden in your shell if needed.

---

## How it works

`generate.py --mode sim` differs from `--mode blackhole` in exactly two ways:

1. **Sets `TT_METAL_SIMULATOR`** before any tt-metal import — this is the only
   change needed to redirect all TTNN dispatch to the virtual device.

2. **Skips the hwmon sentinel check** in `setup_blackhole()`. The sentinel check
   reads `/sys/class/hwmon/hwmon*/temp1_input` to detect dead-ARC chips; that
   path does not exist on a machine without Blackhole silicon and would produce
   spurious warnings. The simulator always presents a healthy virtual device.

Everything else — model loading, CLIP encoding, PNDM scheduler, cross-frame
attention, VAE decode — is identical to the hardware path.

---

## Verifying bit-exactness with silicon

If you have access to both a simulator and a Blackhole board, you can verify that
outputs match by running at the same seed:

```bash
# On the simulator
python examples/generate.py --mode sim \
    --seed 42 --frames 2 --steps 4 --output output/sim_ref.gif

# On hardware (same seed)
python examples/generate.py --mode blackhole \
    --seed 42 --frames 2 --steps 4 --output output/hw_ref.gif
```

The GIFs should be pixel-identical (ttsim is bit-exact for all operations used
by the SD 1.4 TTNN UNet). Any divergence indicates an operation hitting an
`UnimplementedFunctionality` or `UndefinedBehavior` in the simulator.

---

## Known simulator limitations

- **No multi-chip**: ttsim presents one virtual device. `--mode sim` always
  opens `device_ids=[0]` — this matches `--mode blackhole` which is also
  restricted to one chip due to the TTNN UNet's `ttnn.to_torch()` constraint.
- **SFPLOADMACRO unsupported**: requires `TT_METAL_DISABLE_SFPLOADMACRO=1`.
- **Fast dispatch not validated**: use slow dispatch mode.
- **Speed**: expect 10–100× slower than silicon per operation.

For the current list of ttsim limitations, see the upstream repo:
<https://github.com/tenstorrent/ttsim#known-issues>

---

## CI integration

To run the smoke test in CI (no hardware runners needed):

```yaml
# .github/workflows/ci.yml addition
- name: Download ttsim
  run: |
    mkdir -p ~/sim
    wget -q -O ~/sim/libttsim_bh.so \
      https://github.com/tenstorrent/ttsim/releases/download/v1.7.0/libttsim_bh.so
    cp ~/tt-metal/tt_metal/soc_descriptors/blackhole_140_arch.yaml \
      ~/sim/soc_descriptor.yaml

- name: Smoke test on simulator (2 frames × 4 steps)
  run: |
    python examples/generate.py --mode sim \
      --sim $HOME/sim/libttsim_bh.so \
      --frames 2 --steps 4 --output output/ci_smoke.gif
    test -f output/ci_smoke.gif
```

This gives you end-to-end TTNN correctness verification on every PR without
requiring a Blackhole runner.
