"""Data shipped with i4b: the TABULA building catalogue, weather series,
internal-gain profiles and grid signals.

Everything here is addressed through :func:`path`, which resolves against the
installed package. Callers used to build these paths themselves by joining a
repository root onto ``i4b/data/...``; that worked from a checkout and silently
did the wrong thing anywhere else, because the join was relative when no root
was given.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

__all__ = ["path"]


def path(*parts: str, repo_filepath: str | Path = "") -> Path:
    """Return the location of a file or directory inside the packaged data.

    Parameters
    ----------
    *parts : str
        Path segments below ``i4b/data``, e.g. ``"weather", "weather_Mannheim.csv"``.
    repo_filepath : str or Path, optional
        Resolve against this repository root instead of the installed package.
        Retained because callers outside this repo pass it; leaving it empty is
        correct and works from any working directory.
    """
    if repo_filepath:
        return Path(repo_filepath, "i4b", "data", *parts)
    return Path(str(files("i4b.data").joinpath(*parts)))
