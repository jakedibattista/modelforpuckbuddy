from __future__ import annotations

"""I/O utilities.

Provides simple JSON file loading used by multiple scripts.
"""

from pathlib import Path
from typing import Any, Dict, Union
import json


PathLike = Union[str, Path]


def load_json_file(path: PathLike) -> Dict[str, Any]:
    """Load a JSON file into a dictionary.

    Args:
        path: Path to a JSON file.

    Returns:
        Parsed JSON as a dictionary.
    """
    p = Path(path)
    with p.open("r") as f:
        return json.load(f)


