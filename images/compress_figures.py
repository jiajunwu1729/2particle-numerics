"""Make the report versions of the two Lyapunov figures.

The full-size PNGs are written to images/ by lyapunov_poincare_section.py and
lyapunov_section_periodic_orbits.py (they are not kept in the repository).
This script turns them into the JPEGs used by report/report.tex: 1800 px wide
(about 285 dpi at the A4 text width), quality 90.

Run from the repository root, after the two Lyapunov scripts:

    python images/compress_figures.py
"""

from pathlib import Path

from PIL import Image

IMAGES = Path(__file__).resolve().parents[1] / "images"
WIDTH = 1800

for name in ("lyapunov_poincare_section_reduced", "lyapunov_section_periodic_orbits"):
    source = IMAGES / f"{name}.png"
    target = IMAGES / f"{name}_report.jpg"
    image = Image.open(source)
    background = Image.new("RGB", image.size, "white")
    background.paste(image, mask=image.getchannel("A") if image.mode == "RGBA" else None)
    height = round(image.height * WIDTH / image.width)
    background.resize((WIDTH, height), Image.LANCZOS).save(
        target, quality=90, optimize=True, dpi=(285, 285)
    )
    print(f"{source.name}: {source.stat().st_size / 1e6:.1f} MB -> "
          f"{target.name}: {target.stat().st_size / 1e6:.2f} MB")
