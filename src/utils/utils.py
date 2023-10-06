from pathlib import Path
from yaml import safe_load

PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"


def get_config():
    with open(PROJECT_DIR / "config.yml", 'r') as f:
        return safe_load(f)
