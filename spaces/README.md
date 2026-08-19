---
title: tt-animatediff
emoji: 🌌
colorFrom: green
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
the **CPU** path with a distilled Lightning checkpoint (2 or 4 steps), capped to 4
frames at 512×512 for free-tier hardware. A 4-frame run takes several minutes on
free-tier CPU — that wait is expected, not a hang.

Tenstorrent Blackhole is not reachable from Hugging Face infrastructure, so nothing here
reflects hardware performance: a P300C runs the same model at **~12.5 s/frame**, 25
steps, 512×512. The gallery in the app shows real Blackhole output.

**Deploying this Space:** the model repo (`episod/tt-animatediff`) must either be
**public**, or the Space needs an `HF_TOKEN` secret with read access to it — a Space
gets no implicit credential for a private model repo, so without one of these it will
build and reach "Running" and then 401 on a visitor's first click.

**Code:** [github.com/tenstorrent/tt-animatediff](https://github.com/tenstorrent/tt-animatediff)
