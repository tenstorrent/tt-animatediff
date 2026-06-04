# Hardware Compatibility Notes

## tt-metal version

Tested with:
- Firmware bundle: 19.8.0 (KMD 2.8.0)
- Note: firmware 19.5.0 was the last "fully tested" version per tt-metal's own warning; 19.8.0 works with the caveats below.

## SD model path reorganization

Between tt-metal ≤19.5.0 and 19.8.0, the Stable Diffusion 1.4 demo moved:

**Old path (broken):**
```
models.demos.wormhole.stable_diffusion.*
```

**New path (current):**
```
models.demos.vision.generative.stable_diffusion.wormhole.*
```

All imports in this repo have been updated. If you see `ModuleNotFoundError: No module named 'models.demos.wormhole'`, your tt-metal is on the old layout — either upgrade tt-metal or revert the import paths in:
- `animatediff_ttnn/ttnn_pipeline.py`
- `animatediff_ttnn/temporal_attention.py`
- `examples/generate.py`

## Ethernet core hang recovery

If tt-metal throws `Timed out while waiting for active ethernet core`, run:

```bash
tt-smi -r 0 1 2 3
sleep 8
```

This clears hung ethernet cores from prior incomplete teardowns. Safe to run any time no process is actively using the hardware.

## Concurrent device contexts

The Blackhole runtime does not support opening multiple MeshDevice contexts simultaneously in separate processes. All generation in this repo is single-device, sequential.

## Motherboard warning

You may see warnings like:

```
Unknown motherboard 'B850M-C' for chip_id=0 — defaulting tray_id to 0
```

This is cosmetic — the B850M-C hasn't been added to tt-metal's `mobo_to_bus_ids` table. It does not affect functionality.
