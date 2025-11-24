import subprocess, sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = sys.executable

print("Training...")
subprocess.check_call([PY, os.path.join(ROOT, "src", "train", "train.py")])

print("Evaluating...")
subprocess.check_call([PY, os.path.join(ROOT, "src", "eval", "evaluate.py")])

print("Demo complete. Check experiments/results/demo_run")