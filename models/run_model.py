"""Top-level runner kept for compatibility. Use `train.run()` to start training.
Usage:
    QUICK_DEBUG=1 python run_model.py   # quick smoke test
    python run_model.py                # full run (may take long)
"""
import sys
import os

# Ensure repo root is on sys.path so absolute imports like `models.*` resolve
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from models.train import run

if __name__ == '__main__':
    max_images = None
    if len(sys.argv) > 1:
        try:
            max_images = int(sys.argv[1])
        except Exception:
            max_images = None
    run(max_images_per_class=max_images)
