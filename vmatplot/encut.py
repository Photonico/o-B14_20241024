#### ENCUT
# energy versus cutoff energy
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0914

import xml.etree.ElementTree as ET
import os

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import ScalarFormatter
from vmatplot.commons import check_vasprun
from vmatplot.output import canvas_setting, color_sampling

def identify_kpoints(directory="."):