#### Phonon dispersion
# pylint: disable = C0103, C0114, C0116, C0301, C0302, C0321, R0913, R0914, R0915, W0612, W0105

# Necessary packages invoking
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from vmatplot.output_settings import color_sampling, canvas_setting
from vmatplot.algorithms import transpose_matrix
from vmatplot.commons import extract_fermi, get_atoms_count, process_boundary, get_or_default

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
    outcar_file = os.path.join(directory, "OUTCAR")
    q_coords = []
    path = []
    prev_coords = None
    total_distance = 0.0
    null_count = 0

    with open(outcar_file, 'r') as f:
        for line in f:
            if not line.strip(): null_count += 1
            else: null_count = 0
            if null_count >= 20: break
            if "q-point No." in line:
                coord_line = next(f, "").strip()
                if not coord_line: continue
                parts = coord_line.split()
                try:
                    coords = [float(parts[1]), float(parts[2]), float(parts[3])]
                except (IndexError, ValueError):
                    continue
                q_coords.append(coords)
                if prev_coords is None:
                    total_distance = 0.0
                else:
                    dx = coords[0] - prev_coords[0]
                    dy = coords[1] - prev_coords[1]
                    dz = coords[2] - prev_coords[2]
                    d = np.sqrt(dx**2 + dy**2 + dz**2)
                    total_distance += d
                path.append(total_distance)
                prev_coords = coords
    return {"q_coords": q_coords, "path": path}

def extract_phonon_reciprocal_weights(directory):
    # Read CONTCAR file
    contcar_path = f"{directory}/CONTCAR"
    with open(contcar_path, "r") as file:
        lines = file.readlines()
    # Extract lattice vectors
    lattice_vectors = np.array([list(map(float, line.split())) for line in lines[2:5]])
    # Calculate reciprocal lattice vectors
    volume = np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2]))
    reciprocal_lattice_vectors = 2 * np.pi * np.array([
        np.cross(lattice_vectors[1], lattice_vectors[2]) / volume,
        np.cross(lattice_vectors[2], lattice_vectors[0]) / volume,
        np.cross(lattice_vectors[0], lattice_vectors[1]) / volume
    ])
    # Compute the lengths of the reciprocal lattice vectors
    reciprocal_lengths = [np.linalg.norm(vec) for vec in reciprocal_lattice_vectors]
    return reciprocal_lengths

def extract_qpath(directory):
    # Extract q-points and reciprocal weights
    qpoints = extract_phonon_high_sym_details(directory)["q_coords"]
    reciprocal_weights = extract_phonon_reciprocal_weights(directory)
    # Initialize cumulative distances
    cumulative_distances = [0]
    for i in range(1, len(qpoints)):
        # Compute the vector difference between two q-points
        delta_k = np.array(qpoints[i]) - np.array(qpoints[i-1])
        # Apply the reciprocal lattice weight
        weighted_distance = np.sqrt(sum((delta_k[j] * reciprocal_weights[j]) ** 2 for j in range(3)))
        cumulative_distances.append(cumulative_distances[-1] + weighted_distance)
    return cumulative_distances

def extract_eigenvalues_qpoints(directory):
    return 0

def extract_phonon_bands():
    return 0

def create_matters_phonons():
    return 0

def plot_phonons(title, matters_list=None, eigen_range=None, legend_loc=False):
    return 0