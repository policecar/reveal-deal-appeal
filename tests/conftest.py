import sys

import pytest
from pathlib import Path

# Scripts import modules as `from config import ...` and run with src/ on the
# path (python src/refine.py); mirror that for tests.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent.parent / "data"
