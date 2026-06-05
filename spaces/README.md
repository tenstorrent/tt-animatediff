---
title: tt-animatediff
emoji: 🌌
colorFrom: teal
colorTo: indigo
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
license: apache-2.0
short_description: AnimateDiff on Tenstorrent Blackhole (ttsim)
---

# tt-animatediff — HuggingFace Space

Generates animated GIFs using SD 1.4 with cross-frame temporal attention,
running on the **ttsim** Blackhole simulator (bit-exact with real hardware).

No Tenstorrent silicon required — ttsim runs on standard CPU x86_64.
Slower than real hardware (~10–100× per operation); use 2 frames × 4 steps
for a quick smoke test.

**Hardware path:** [github.com/tenstorrent/tt-animatediff](https://github.com/tenstorrent/tt-animatediff)
**Showcase:** [tenstorrent.github.io/tt-animatediff](https://tenstorrent.github.io/tt-animatediff/)
