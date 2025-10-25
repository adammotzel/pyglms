"""
Configure global settings for package.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("turtles")
except PackageNotFoundError:
    __version__ = "0.0.0"

# modules
__all__ = [
    "preprocess",
    "stats",
    "plotting"
]
