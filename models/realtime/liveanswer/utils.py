import os
from typing import Optional
from pathlib import Path

# Load environment variables from .env if available
from dotenv import load_dotenv  # type: ignore

# Load from the project root .env file
project_root = Path(__file__).resolve().parents[4]  # Go up 4 levels to vera_dev
env_path = project_root / ".env"
load_dotenv(env_path)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name, default)
    return value


