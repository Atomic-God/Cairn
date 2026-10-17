"""Root pytest configuration.

Ensures the repository root is on ``sys.path`` so that ``pytest`` works on a
fresh clone even before the package is installed (``pip install -e .`` is
still the recommended path). Harmless when the installed package shadows
the source tree: in that case the import resolves to the installed copy
first only if it appears earlier on ``sys.path`` — inserting at position 0
keeps the source tree authoritative for the test run, which is what we
want when testing a checkout.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
