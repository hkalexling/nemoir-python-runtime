"""``python -m nemoir_runtime`` entry point for the ``nemotrace`` verifier CLI."""

from __future__ import annotations

import sys

from nemoir_runtime.cli import main

if __name__ == "__main__":
    sys.exit(main())
