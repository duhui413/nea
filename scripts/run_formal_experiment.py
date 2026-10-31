"""Run or resume the eight-arm paper protocol in one batch directory."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from nea.pipeline.formal_entry import main

if __name__ == "__main__":
    raise SystemExit(main(repo_root=REPO_ROOT))
