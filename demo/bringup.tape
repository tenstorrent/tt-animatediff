Output demo/bringup.gif
Output demo/bringup.mp4

Set Shell "bash"
Set FontSize 13
Set Width 1760
Set Height 800
Set Theme "Dracula"
Set TypingSpeed 60ms
Set Padding 20

# ── hidden bootstrap ──────────────────────────────────────────────────────────
Hide
Env TF_CPP_MIN_LOG_LEVEL "3"
Env PYTHONWARNINGS "ignore"
Env PYTHONDONTWRITEBYTECODE "1"
Env PAGER "cat"
Env BAT_PAGER "cat"
Env MICRO_TRUECOLOR "1"
Type "bash /home/ttuser/code/tt-animatediff/demo/setup_demo_files.sh && cd /tmp/tt-animatediff-demo && clear"
Enter
Sleep 1s
Show

# ─────────────────────────────────────────────────────────────────────────────
# 1. Orient ourselves — clone the reference, look around
# ─────────────────────────────────────────────────────────────────────────────

Type "git clone --depth 1 https://github.com/guoyww/AnimateDiff.git"
Enter
Sleep 12s

Type "ls AnimateDiff/"
Enter
Sleep 2s

Type "ls AnimateDiff/animatediff/"
Enter
Sleep 2s

Type "ls AnimateDiff/animatediff/models/"
Enter
Sleep 3s

# ─────────────────────────────────────────────────────────────────────────────
# 2. Open the file that matters
# ─────────────────────────────────────────────────────────────────────────────

Sleep 1s
Type "micro AnimateDiff/animatediff/models/motion_module.py"
Enter
Sleep 3s

# scroll down to VanillaTemporalModule — line ~130
PageDown 8
Sleep 4s

# scroll back up a bit to read the forward() method
PageUp
Sleep 5s

Ctrl+Q
Sleep 1s

# ─────────────────────────────────────────────────────────────────────────────
# 3. Start our project
# ─────────────────────────────────────────────────────────────────────────────

Sleep 800ms
Type "mkdir -p animatediff_ttnn"
Enter
Sleep 1s

Type "touch animatediff_ttnn/__init__.py"
Enter
Sleep 800ms

Type "ls animatediff_ttnn/"
Enter
Sleep 1s

# ─────────────────────────────────────────────────────────────────────────────
# 4. Write temporal_module.py in micro
# ─────────────────────────────────────────────────────────────────────────────

Sleep 800ms
Type "micro animatediff_ttnn/temporal_module.py"
Enter
Sleep 2s

# File is pre-populated by setup script — scroll through it
PageUp 8
Sleep 2s
PageDown
Sleep 3s
PageDown
Sleep 3s
PageDown
Sleep 3s
PageDown 8
Sleep 3s

Ctrl+Q
Sleep 1s

# ─────────────────────────────────────────────────────────────────────────────
# 5. Run the shape test
# ─────────────────────────────────────────────────────────────────────────────

Sleep 800ms
Type "python3 test_shapes.py"
Enter
Sleep 10s

# ─────────────────────────────────────────────────────────────────────────────
# 6. Open pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

Sleep 800ms
Type "micro animatediff_ttnn/pipeline.py"
Enter
Sleep 2s

PageUp 8
Sleep 2s
PageDown
Sleep 3s
PageDown 8
Sleep 3s

Ctrl+Q
Sleep 1s

# ─────────────────────────────────────────────────────────────────────────────
# 7. Run end-to-end
# ─────────────────────────────────────────────────────────────────────────────

Sleep 800ms
Type "python3 run_demo.py"
Enter
Sleep 12s

# ─────────────────────────────────────────────────────────────────────────────
# 8. Check what we built
# ─────────────────────────────────────────────────────────────────────────────

Sleep 1s
Type "ls -lh animatediff_ttnn/"
Enter
Sleep 2s

Type "ls -lh *.py"
Enter
Sleep 2s

Sleep 1s
Type "# done. swap synthetic_weights() for from_checkpoint() on real hardware"
Enter
Sleep 3s
