from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

DEFAULT_TILE_TRIGGER = 2800
DEFAULT_TILE_SIZE = 2200
DEFAULT_TILE_OVERLAP = 0.15


@dataclass(frozen=True)
class Tile:
    index: int
    box: tuple[int, int, int, int]
    image: Image.Image


def calculate_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1.0 - overlap)))
    count = max(2, math.ceil((length - tile_size) / stride) + 1)
    starts: list[int] = []
    for index in range(count):
        start = min(index * stride, length - tile_size)
        if not starts or start != starts[-1]:
            starts.append(start)
    return starts


def create_tiles(
    image: Image.Image,
    tile_trigger: int = DEFAULT_TILE_TRIGGER,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: float = DEFAULT_TILE_OVERLAP,
) -> list[Tile]:
    width, height = image.size
    if max(width, height) <= tile_trigger:
        return [Tile(1, (0, 0, width, height), image.copy())]
    x_starts = calculate_starts(width, tile_size, overlap) if width > tile_trigger else [0]
    y_starts = calculate_starts(height, tile_size, overlap) if height > tile_trigger else [0]
    crop_width = min(width, tile_size) if width > tile_trigger else width
    crop_height = min(height, tile_size) if height > tile_trigger else height
    tiles: list[Tile] = []
    index = 1
    for y in y_starts:
        for x in x_starts:
            box = (x, y, min(width, x + crop_width), min(height, y + crop_height))
            tiles.append(Tile(index, box, image.crop(box)))
            index += 1
    return tiles


def save_tiles(tiles: list[Tile], target_dir: str | Path, stem: str) -> list[Path]:
    """Speichert jede Kachel als eigenständige PNG-Datei und liefert deren Pfade."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for tile in tiles:
        path = target_dir / f"{stem}_tile{tile.index:03d}.png"
        tile.image.save(path)
        paths.append(path)
    return paths
