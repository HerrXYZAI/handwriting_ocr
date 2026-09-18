from __future__ import annotations

from pathlib import Path

import pymupdf
from PIL import Image

DEFAULT_PDF_DPI = 300


def is_pdf(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".pdf"


def _render_page(page: pymupdf.Page, dpi: int) -> Image.Image:
    zoom = dpi / 72
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB, alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def pdf_page_count(pdf_path: str | Path) -> int:
    with pymupdf.open(pdf_path) as document:
        return document.page_count


def render_pdf_page(pdf_path: str | Path, page_index: int, dpi: int = DEFAULT_PDF_DPI) -> Image.Image:
    """Rendert eine PDF-Seite (0-basiert) als RGB-Bild."""
    with pymupdf.open(pdf_path) as document:
        if not 0 <= page_index < document.page_count:
            raise ValueError(f"Seite {page_index + 1} existiert nicht (Dokument hat {document.page_count} Seiten).")
        return _render_page(document.load_page(page_index), dpi)


def extract_pdf_pages(pdf_path: str | Path, dpi: int = DEFAULT_PDF_DPI, output_dir: str | Path | None = None) -> list[Path]:
    """Rastert alle Seiten eines PDFs zu PNG-Dateien und liefert deren Pfade.

    Bereits gerenderte Seiten werden wiederverwendet, solange sie neuer als
    das PDF sind, damit ein wiederholter Aufruf nicht erneut rendert.
    """
    pdf_path = Path(pdf_path).resolve()
    target_dir = Path(output_dir).resolve() if output_dir else pdf_path.with_name(f"{pdf_path.stem}_pages")
    target_dir.mkdir(parents=True, exist_ok=True)
    pdf_mtime = pdf_path.stat().st_mtime

    paths: list[Path] = []
    with pymupdf.open(pdf_path) as document:
        for index in range(document.page_count):
            page_path = target_dir / f"{pdf_path.stem}_p{index + 1:03d}.png"
            if not page_path.exists() or page_path.stat().st_mtime < pdf_mtime:
                _render_page(document.load_page(index), dpi).save(page_path)
            paths.append(page_path)
    return paths
