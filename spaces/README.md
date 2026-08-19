---
title: tt-animatediff
emoji: 🌌
colorFrom: teal
colorTo: indigo
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: apache-2.0
short_description: AnimateDiff on Tenstorrent Blackhole — capped CPU demo
---

# tt-animatediff — CPU reference demo

Loads [`episod/tt-animatediff`](https://huggingface.co/episod/tt-animatediff) and runs
the **CPU** path with a distilled 4-step Lightning checkpoint, capped to 4 frames at
384×384 for free-tier hardware.

Tenstorrent Blackhole is not reachable from Hugging Face infrastructure, so nothing here
reflects hardware performance: a P300C runs the same model at **~12.5 s/frame**, 25
steps, 512×512. The gallery in the app shows real Blackhole output.

**Code:** [github.com/tenstorrent/tt-animatediff](https://github.com/tenstorrent/tt-animatediff)
