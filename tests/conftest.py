import sys
from pathlib import Path

# Ensure the project root is importable as 'src' / 'api' when running pytest
# from any working directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))