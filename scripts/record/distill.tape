# distill.tape — VHS recording of the AnimateDiff-Lightning distillation process.
#
# This tape records a DEMO run (50 steps) to show the process.
# For the full run (~10 hours), use: bash scripts/record/run_distill.sh
#
# Prerequisites:
#   pip install vhs  (or: brew install charmbracelet/tap/vhs)
#   vhs scripts/record/distill.tape
#
# Output: scripts/record/recordings/distill.gif

Output scripts/record/recordings/distill.gif

Set FontSize 13
Set Width 200
Set Height 55
Set Theme "Dracula"
Set Padding 12

# ── intro ───────────────────────────────────────────────────────────
Sleep 500ms
Type "# AnimateDiff-Lightning — LCM distillation on Tenstorrent Blackhole"
Sleep 800ms
Enter
Sleep 300ms
Type "# Distilling SD 1.4 UNet from 25 steps → 4 steps and 8 steps"
Sleep 800ms
Enter
Sleep 300ms
Type "# Hardware: 4× Blackhole P300C · 16-core x86 · 249 GB RAM"
Sleep 1s
Enter
Sleep 1s

# ── check hardware ──────────────────────────────────────────────────
Type "tt-smi -s | python3 -c \""
Enter
Sleep 200ms
Type "import sys, json; d=json.load(sys.stdin)"
Enter
Sleep 200ms
Type "chips = d.get('device_info', [])"
Enter
Sleep 200ms
Type "print(f'  {len(chips)} Blackhole chip(s) online')"
Enter
Sleep 200ms
Type "[(print(f'  chip {c[\\\"device_id\\\"]}: {c[\\\"board_type\\\"]}, AICLK {c[\\\"tt_flash_version\\\"]}')) for c in chips]\""
Enter
Sleep 2s

# ── phase 1: LCM UNet (demo: 50 steps) ─────────────────────────────
Enter
Type "echo ''"
Enter
Type "echo '╔══════════════════════════════════════════════════════════════'"
Enter
Type "echo '║  Phase 1: LCM UNet Distillation (demo — 50 steps)'"
Enter
Type "echo '║  Full run: --num_train_steps 5000 (~10 hours on CPU)'"
Enter
Type "echo '╚══════════════════════════════════════════════════════════════'"
Enter
Sleep 500ms
Type "python3 scripts/distill_lcm.py --steps 8 --num_train_steps 50"
Enter
Sleep 120s

# ── phase 2: MotionAdapter (demo: 30 steps) ─────────────────────────
Type "echo ''"
Enter
Type "echo '╔══════════════════════════════════════════════════════════════'"
Enter
Type "echo '║  Phase 2: MotionAdapter Distillation (demo — 30 steps)'"
Enter
Type "echo '║  Full run: --num_train_steps 3000 (~45 min on CPU)'"
Enter
Type "echo '╚══════════════════════════════════════════════════════════════'"
Enter
Sleep 500ms
Type "python3 scripts/distill_motion_adapter.py --steps 8 --unet weights/unet_lcm_8step.pt --num_train_steps 30"
Enter
Sleep 60s

# ── done ─────────────────────────────────────────────────────────────
Type "echo ''"
Enter
Type "echo '╔══════════════════════════════════════════════════════════════'"
Enter
Type "echo '║  Weights produced:'"
Enter
Type "echo '║    weights/unet_lcm_8step.pt'"
Enter
Type "echo '║    weights/motion_adapter_lcm_8step.pt'"
Enter
Type "echo '║'"
Enter
Type "echo '║  Full run: bash scripts/record/run_distill.sh'"
Enter
Type "echo '║  Download pre-built: bash weights/download_weights.sh'"
Enter
Type "echo '╚══════════════════════════════════════════════════════════════'"
Enter
Sleep 3s
