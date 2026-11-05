#!/usr/bin/env python3
"""Canonical entrypoint for the refactored BEVDet thesis experiments."""

import sys
from pathlib import Path


# Match the legacy entrypoint's repository-local import behavior without
# depending on the caller's working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from bevdet_experiment.runner import main


if __name__ == '__main__':
    main()
