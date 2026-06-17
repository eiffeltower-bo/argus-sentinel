"""COCO class taxonomy shared by detection, tracking, and peek.

Lives in ``core`` (not in any one domain) so every layer imports it downward — no
backend or pipeline module owns these maps.
"""

from __future__ import annotations

# COCO classes we care about, and how they roll up into categories.
COCO_LABELS: dict[int, str] = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CATEGORY_BY_CLASS: dict[int, str] = {
    0: "person",
    2: "vehicle",
    3: "vehicle",
    5: "vehicle",
    7: "vehicle",
}
TARGET_CLASSES: dict[str, list[int]] = {"person": [0], "vehicle": [2, 3, 5, 7]}


def classes_for(targets: tuple[str, ...]) -> list[int]:
    """Resolve target category names (``"person"``/``"vehicle"``) to COCO class ids."""
    classes: list[int] = []
    for t in targets:
        if t not in TARGET_CLASSES:
            raise ValueError(f"unknown target {t!r}; choose from {list(TARGET_CLASSES)}")
        classes.extend(TARGET_CLASSES[t])
    return classes
