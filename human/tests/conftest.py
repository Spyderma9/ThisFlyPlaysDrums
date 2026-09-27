import sys
from pathlib import Path

# human/ scripts import each other by bare name (they're run as `python human/x.py`), so tests do the same.
HUMAN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HUMAN))
