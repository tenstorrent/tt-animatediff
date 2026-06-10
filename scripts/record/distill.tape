# distill.tape — VHS recording script for the full distillation run.
# Play back: vhs scripts/record/distill.tape
# Produces:  scripts/record/recordings/distill.gif

Output scripts/record/recordings/distill.gif
Set FontSize 14
Set Width 220
Set Height 50

# Open tmux split: left = training, right = hardware monitor
Type "tmux new-session -d -s distill -x 220 -y 50 && tmux split-window -h -t distill 'watch -n 2 tt-smi -s' && tmux attach -t distill"
Enter
Sleep 3s

# Switch to left pane and start training
Type "tmux select-pane -t distill:0.0"
Enter
Sleep 1s
Type "cd /home/ttuser/code/tt-animatediff && bash scripts/record/run_distill.sh"
Enter
