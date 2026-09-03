"""EmoPyLab Desktop Workstation Launcher (PySide6 / Qt6).

Entry point for the EmoPyLab Desktop GUI Application.
Can be executed as:
    python EmoPyLab.py
    python emopylabgui.py
    emopylab-gui
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import main execution loop from the primary window implementation
from emopylab_app import main as _main


def main() -> int:
    """Launches the EmoPyLab Desktop Workstation."""
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
