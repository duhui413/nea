"""Run or resume registered controlled diagnosis perturbations in an existing run."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    from nea.pipeline.formal_entry import main
    raise SystemExit(main(["--phase", "perturb", *sys.argv[1:]], repo_root=REPO_ROOT))
