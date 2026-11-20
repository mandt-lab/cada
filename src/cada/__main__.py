"""Run the CADA command-line interface with ``python -m cada``."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
