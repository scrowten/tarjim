import sys
from pathlib import Path

# Ensure the project root (one level up from tests/) is on sys.path so
# imports like `import tarjim.src.core...` work when running pytest from
# different working directories or CI.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
