"""Record the original labelled PNG artwork used directly by the thesis."""
import hashlib
import json
from PIL import Image
from style import *

# %% Original structural artwork, included in the thesis without PDF conversion.
artwork = [
    ('fig2.1', THESIS/'fig2.1/fig2.1_6k.png', 8),
    ('fig2.3', THESIS/'fig2.3/fig2.3_4k.png', 8),
    ('fig2.4', THESIS/'fig2.4/fig2.4_6k.png', 8),
    ('fig2.7', THESIS/'fig2.7/fig2.7_from2.6.png', 7.5),
    ('S2.17', THESIS/'S2.17/S2.17.png', 7.5),
]
manifest = {}
for name, path, width in artwork:
    with Image.open(path) as pixels:
        dimensions = list(pixels.size)
    manifest[name] = {'source': str(path),
                      'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'pixel_dimensions': dimensions,
                      'thesis_width_fraction': width/10,
                      'method': 'Original labelled PNG included directly; no conversion.'}
(OUT/'structure_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
