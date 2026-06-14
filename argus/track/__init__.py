"""Tracking-by-detection: the ``Tracker`` contract (from core) + its backends."""

from ..core import Track, Tracker
from .backends.bytetrack import ByteTrackTracker

__all__ = ["Tracker", "Track", "ByteTrackTracker"]
