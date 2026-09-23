"""Keep the author's layered structural artwork; replace only its text layers.

GIMP 3 exports temporary artwork; Matplotlib adds the shared thesis typography.
The XCF files are never saved or modified. Canvas widths follow the existing
TeX widths, giving exactly the same printed font scale as the numerical plots.
"""
import hashlib
import json
import os
import subprocess
import tempfile

from PIL import Image
from style import *

jobs = [
    ("fig2.1", THESIS / "fig2.1/fig2.1_6k.xcf", 8.0),
    ("fig2.3", THESIS / "fig2.3/fig2.3_4k.xcf", 8.0),
    ("fig2.4", THESIS / "fig2.4/fig2.4.xcf", 8.0),
    ("fig2.7", THESIS / "fig2.7/fig2.7_from2.6.xcf", 7.5),
    ("S2.17", ROOT / "figures_collection/S2.17/S2.17_template.xcf", 7.5),
]
box = dict(boxstyle="round", facecolor="white",
           edgecolor=plt.rcParams["legend.edgecolor"],
           alpha=plt.rcParams["legend.framealpha"])
manifest = {}
with tempfile.TemporaryDirectory(prefix="thesis-structure-") as temporary:
    work = Path(temporary)
    (work / "jobs.json").write_text(json.dumps([
        {"name": name, "source": str(source)} for name, source, _ in jobs]))
    environment = os.environ.copy()
    environment["THESIS_STRUCTURE_TEMP"] = temporary
    helper = OUT / "export_structure_layers.py"
    subprocess.run(["gimp", "-n", "-i", "-d", "-f",
                    "--batch-interpreter=python-fu-eval",
                    "-b", f"exec(open({str(helper)!r}).read())", "--quit"],
                   env=environment, check=True)
    for name, source, width in jobs:
        pixels = Image.open(work / (name + ".png"))
        w, h = pixels.size
        labels = json.loads((work / (name + ".json")).read_text())
        bottom_space = .07 if name == "fig2.7" else 0
        fig = plt.figure(figsize=(width, width * h * (1 + bottom_space) / w))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(pixels, interpolation="none")
        if bottom_space:
            ax.set_ylim(h * (1 + bottom_space), -.5)
        ax.set_axis_off()
        for label in labels:
            text = label["text"]
            if not text:
                continue
            x, y, align = label["x"], label["y"], "left"
            is_title = text.startswith("(") or "states along" in text
            if name in ["fig2.3", "fig2.4"]:
                column = 0 if text in ["(a)", "(c)"] else 1
                row = 0 if text in ["(a)", "(b)"] else 1
                x = (0.025 + 0.5 * column) * w
                y = 0.025 * h + row * (2400 if name == "fig2.3" else 3000)
            if name in ["fig2.7", "S2.17"]:
                if "states along" in text:
                    text = text.replace(" along T-Y", "")
                    x = 0.07 * w
                elif text in ["spin-up", "spin-down"]:
                    x = (0.22 if text == "spin-up" else 0.52) * w
                    y = (0.445 if y < h / 2 else 0.925) * h
                    if name == "fig2.7" and y > h / 2:
                        y = 1.008 * h
                    align = "center"
                elif text in ["(a)", "(b)"]:
                    x = (0.035 if text == "(a)" else 0.71) * w
            ax.text(x, y, text, fontsize=12 if is_title else 14, ha=align,
                    va="top", bbox=box if is_title else None)
        save(fig, name + ".pdf")
        manifest[name] = {
            "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "canvas_width_inches": width, "tex_width_fraction": width / 10,
            "labels": labels,
        }
(OUT / "structure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
