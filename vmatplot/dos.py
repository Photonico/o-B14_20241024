#### Declarations of process functions for PDoS with vectorized programming
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

# Necessary packages invoking
import xml.etree.ElementTree as ET
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from vmatplot.commons import extract_fermi
from vmatplot.output_settings import color_sampling, canvas_setting

