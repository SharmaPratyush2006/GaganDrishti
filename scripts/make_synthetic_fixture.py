#!/usr/bin/env python
"""Generate the synthetic shadow fixture.

Thin CLI wrapper so the generator can be run straight from a clone without
installing the package:

    python scripts/make_synthetic_fixture.py --config configs/default.yaml

Once the package is installed (``pip install -e .``) the equivalent command is:

    depthwizard-make-fixture --config configs/default.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a clone without `pip install -e .`.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from depthwizard.ingest.synthetic import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
