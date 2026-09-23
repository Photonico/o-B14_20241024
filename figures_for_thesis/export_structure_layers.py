"""Run inside GIMP: export visible artwork without altering the source XCF."""
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from gi.repository import Gimp, Gio

work = Path(os.environ["THESIS_STRUCTURE_TEMP"])
jobs = json.loads((work / "jobs.json").read_text())
for job in jobs:
    image = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE,
                           Gio.File.new_for_path(job["source"]))
    labels = []

    def remove_text(layers):
        for layer in layers:
            if not layer.get_visible():
                continue
            if layer.is_text_layer():
                _, x, y = layer.get_offsets()
                text = layer.get_text()
                if text is None:
                    text = "".join(ET.fromstring("<root>" + layer.get_markup() + "</root>").itertext())
                labels.append({"text": text, "x": x, "y": y,
                               "width": layer.get_width(), "height": layer.get_height()})
                layer.set_visible(False)
            elif layer.is_group():
                remove_text(layer.get_children())

    remove_text(image.get_layers())
    procedure = Gimp.get_pdb().lookup_procedure("file-png-export")
    config = procedure.create_config()
    config.set_property("run-mode", Gimp.RunMode.NONINTERACTIVE)
    config.set_property("image", image)
    config.set_property("file", Gio.File.new_for_path(str(work / (job["name"] + ".png"))))
    result = procedure.run(config)
    assert result.index(0) == Gimp.PDBStatusType.SUCCESS, job["name"]
    (work / (job["name"] + ".json")).write_text(json.dumps(labels))
    image.delete()
