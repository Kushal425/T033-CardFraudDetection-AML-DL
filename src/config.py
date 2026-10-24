"""
src/config.py
=============
Loads the project configuration from ``configs/config.yaml`` and exposes it as
a plain Python ``dict`` (``CFG``).  Also provides a tiny helper that returns
the repo root so every other module can build absolute paths without hard-coding
them.

All other modules should import ``from src.config import CFG`` rather than
reading the YAML file themselves.
"""

from pathlib import Path
import yaml

# ── Locate the repo root ────────────────────────────────────────────────────
# This file lives at  <repo>/src/config.py, so root is two levels up.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent


def get_repo_root() -> Path:
    """Return the absolute path to the repository root directory."""
    return REPO_ROOT


# ── Load config ─────────────────────────────────────────────────────────────
_CONFIG_PATH: Path = REPO_ROOT / "configs" / "config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    """
    Read and return the YAML configuration as a nested Python dict.

    Parameters
    ----------
    path : Path, optional
        Explicit path to the YAML file.  Defaults to ``configs/config.yaml``
        relative to the repository root.

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at *path*.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Expected 'configs/config.yaml' in the repository root."
        )
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# Module-level singleton so callers can simply do ``from src.config import CFG``
CFG: dict = load_config()
