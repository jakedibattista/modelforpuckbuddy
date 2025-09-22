from __future__ import annotations

"""Configuration helpers.

Handles optional loading of a .env file and retrieving required env vars.
"""

from typing import Optional
import os


def load_env() -> None:
    """Load environment variables from a .env file if python-dotenv is installed.

    This is a no-op if python-dotenv is not available.
    """
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        # Quietly ignore if dotenv is not installed or fails; env may be set elsewhere
        pass


def get_required_env(key: str) -> str:
    """Get a required environment variable or raise a helpful error.

    Args:
        key: The environment variable name to retrieve.

    Returns:
        The environment variable value.

    Raises:
        RuntimeError: If the environment variable is not set.
    """
    value: Optional[str] = os.getenv(key)
    if not value:
        raise RuntimeError(f"Required environment variable not set: {key}")
    return value


