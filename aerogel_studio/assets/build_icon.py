"""Render the code-authored SVG and build the Windows multi-resolution icon."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QRectF
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer


def main() -> None:
    app = QGuiApplication.instance() or QGuiApplication([])
    folder = Path(__file__).resolve().parent
    renderer = QSvgRenderer(str(folder / "app_icon.svg"))
    if not renderer.isValid():
        raise ValueError("The application SVG is invalid")
    canvas = QImage(1024, 1024, QImage.Format.Format_RGBA8888)
    canvas.fill(0)
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter, QRectF(0, 0, 1024, 1024))
    painter.end()
    bitmap = Image.frombytes("RGBA", (1024, 1024), bytes(canvas.constBits()))
    bitmap = bitmap.resize((256, 256), Image.Resampling.LANCZOS)
    bitmap.save(folder / "app_icon.png")
    bitmap.save(
        folder / "app_icon.ico",
        format="ICO",
        sizes=[(size, size) for size in (16, 24, 32, 48, 64, 128, 256)],
    )
    print("Created app_icon.png and app_icon.ico")
    _ = app


if __name__ == "__main__":
    main()
