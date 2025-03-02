#### Phonon dispersion
# pylint: disable = C0103, C0114, C0116, C0301, C0302, C0321, R0913, R0914, R0915, W0612, W0105

# Necessary packages invoking
import xml.etree.ElementTree as ET
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from vmatplot.output_settings import color_sampling, canvas_setting
from vmatplot.algorithms import transpose_matrix
from vmatplot.commons import extract_fermi, get_atoms_count, process_boundary, get_or_default

from vmatplot.bandstructure import extract_kpath, kpoints_index, kpoints_path, high_symmetry_path, is_kpoints_returning
from vmatplot.dos import extract_dos
from vmatplot.pdos import extract_dict_pdos, create_matters_pdos

global_tolerance = 1e-4

def is_qpoints_returning(directory):
    qpoints_file_path = os.path.join(directory, "QPOINTS")
    qpoints_opt_path = os.path.join(directory, "QPOINTS_OPT")
    qpoints_file = None

    # Determine which file to use
    if os.path.exists(qpoints_opt_path):
        qpoints_file = qpoints_opt_path
    elif os.path.exists(qpoints_file_path):
        qpoints_file = qpoints_file_path
    else: return False
    try:
        with open(qpoints_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
        # Ensure it's a line-mode QPOINTS file
        if lines[2][0].lower() != "l":
            return False
        # Extract high symmetry points
        high_symmetry_points = []
        for line in lines[4:]:
            tokens = line.strip().split()
            if tokens and tokens[-1].isalpha():  # Check if the last token is a label
                high_symmetry_points.append(tokens[-1])
        # Check if the first and last points are the same
        return high_symmetry_points and high_symmetry_points[0] == high_symmetry_points[-1]
    except Exception:
        return False

def extract_phonon_high_sym(directory):
    # Open and read the QPOINTS file
    qpoints_file_path = os.path.join(directory, "QPOINTS")
    qpoints_opt_path = os.path.join(directory, "QPOINTS_OPT")
    if os.path.exists(qpoints_opt_path):
        qpoints_file = qpoints_opt_path
    elif os.path.exists(qpoints_file_path):
        qpoints_file = qpoints_file_path
    else:
        raise FileNotFoundError("QPOINTS file not found in the directory.")
    with open(qpoints_file, "r", encoding="utf-8") as file:
        QPOINTS = file.readlines()
    # Check if the QPOINTS file is in line-mode
    if QPOINTS[2][0] not in ("l", "L"):
        raise ValueError(f"Expected 'L' on the third line of QPOINTS file, got: {QPOINTS[2]}")
    # Initialize a list to store high symmetry points
    high_symmetry_points = []
    # Read the high symmetry points from the QPOINTS file
    for i in range(4, len(QPOINTS)):
        tokens = QPOINTS[i].strip().split()
        if tokens and tokens[-1].isalpha():
            high_symmetry_points.append(tokens[-1])
    # Remove duplicates except for the first and last points
    if len(high_symmetry_points) > 2:
        unique_points = [high_symmetry_points[0]]       # Keep the first point
        seen = set(unique_points)
        for point in high_symmetry_points[1:-1]:        # Process middle points
            if point not in seen:
                unique_points.append(point)
                seen.add(point)
        unique_points.append(high_symmetry_points[-1])  # Keep the last point
    else: unique_points = high_symmetry_points            # If only two points, return as is
    return unique_points

def extract_phonon_high_sym_details(directory):
    # Construct the full path to the vasprun.xml file
    xml_file = os.path.join(directory, "vasprun.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # Initialize a list to store the k-point coordinates
    qpoints = []
    # These elements contain the k-point coordinates
    qpoints_file_path = os.path.join(directory, "QPOINTS")
    qpoints_opt_path = os.path.join(directory, "QPOINTS_OPT")
    # HSE06 algorithms
    if os.path.exists(qpoints_opt_path):
        varray_nodes = root.findall("./calculation/eigenvalues_qpoints_opt[@comment='qpoints_opt']/qpoints/varray[@name='kpointlist']")
        if varray_nodes:
            last_varray = varray_nodes[-1]
            for kpoint in last_varray.findall("./v"):
                coords = [float(x) for x in kpoint.text.split()]
                qpoints.append(coords)
    # GGA-PBE algorithms
    elif os.path.exists(qpoints_file_path):
        varray_nodes = root.findall("./calculation/qpoints/varray[@name='kpointlist']")
        if varray_nodes:
            last_varray = varray_nodes[-1]
            for kpoint in last_varray.findall("./v"):
                coords = [float(x) for x in kpoint.text.split()]
                qpoints.append(coords)
    # Return the list of k-point coordinates
    return qpoints
