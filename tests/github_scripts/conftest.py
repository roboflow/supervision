"""Make the CI helper scripts importable by the tests that exercise them.

The scripts in `.github/scripts` are standalone entry points rather than an installed
package, so the directory holding them has to be on `sys.path` before the test modules
can import them by name.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / ".github" / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
