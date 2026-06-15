"""Persistence: the ``Store`` contract + backends (SQLite + sqlite-vec)."""

from ..core import Store
from .backends.sqlite import SqliteStore

__all__ = ["Store", "SqliteStore"]
