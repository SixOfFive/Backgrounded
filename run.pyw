#!/usr/bin/env pythonw
"""Launcher. Use run.pyw (pythonw) for no console window, or

    python run.pyw --help

for the same entry point with output attached."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backgrounded.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
