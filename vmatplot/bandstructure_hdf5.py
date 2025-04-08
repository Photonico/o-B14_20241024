#### Bandstructure using HDF5
# pylint: disable = C0103, C0114, C0116, C0301, C0302, C0321, R0913, R0914, R0915, W0612, W0105

# Necessary packages invoking
import xml.etree.ElementTree as ET
import os
import h5py
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from vmatplot.output_settings import color_sampling, canvas_setting
from vmatplot.algorithms import transpose_matrix
from vmatplot.commons import extract_fermi, get_atoms_count, process_boundary, get_or_default
from vmatplot.dos import extract_dos
from vmatplot.pdos import extract_dict_pdos, create_matters_pdos

global_tolerance = 1e-4

