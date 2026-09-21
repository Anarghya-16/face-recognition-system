"""Filesystem helpers for identity-per-folder datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def load_identity_folders(root: str | Path) -> Dict[str, List[Path]]:
    """Read a dataset laid out as `root/<person_name>/<image files>`."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    people: Dict[str, List[Path]] = {}
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = list_images(person_dir)
        if imgs:
            people[person_dir.name] = imgs
    if not people:
        raise ValueError(
            f"No identity sub-folders with images under {root}. "
            "Expected layout: <root>/<person_name>/<image>.jpg"
        )
    return people
