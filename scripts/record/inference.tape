# inference.tape — VHS recording for the short inference demo.
# Play back: vhs scripts/record/inference.tape
# Produces:  scripts/record/recordings/inference.gif

Output scripts/record/recordings/inference.gif
Set FontSize 14
Set Width 160
Set Height 30

Type "cd /home/ttuser/code/tt-animatediff"
Enter
Sleep 1s
Type "bash scripts/record/run_inference.sh"
Enter
