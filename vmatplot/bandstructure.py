#### Band structure
# This module provides VASP bandstructure plotting utilities.
# High-symmetry labels are read directly from KPOINTS/KPOINTS_OPT and preserve Unicode labels such as Γ, K′, and M′.
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

import matplotlib as mpl

mpl.rcParams["lines.solid_capstyle"] = "round"
mpl.rcParams["lines.dash_capstyle"]  = "round"
mpl.rcParams["lines.solid_joinstyle"] = "round"
mpl.rcParams["lines.dash_joinstyle"]  = "round"

global_tolerance = 1e-4

# Band structure plotting type aliases.
# These helpers make the plotting routines handle spin/non-spin aliases consistently.
_MONOCOLOR_TYPES = {
    "monocolor",
    "monocolor nonpolarized",
    "monocolor spin up",
    "spin up monocolor",
    "monocolor spin down",
    "spin down monocolor",
}

_BANDS_TYPES = {
    "bands",
    "bands nonpolarized",
    "bands spin up",
    "spin up bands",
    "bands spin down",
    "spin down bands",
}


def _normalize_bstype(bstype):
    """Normalize a bandstructure style string for robust comparisons."""
    return str(bstype).strip().lower()


def _is_monocolor_type(bstype):
    """Return True for all monocolor bandstructure style aliases."""
    return _normalize_bstype(bstype) in _MONOCOLOR_TYPES


def _is_bands_type(bstype):
    """Return True for all conduction/valence bandstructure style aliases."""
    return _normalize_bstype(bstype) in _BANDS_TYPES


def _clean_kpoints_label(label):
    """Clean a KPOINTS label while preserving Unicode labels such as Γ, K′, and M′.

    The function intentionally does not remap Greek names such as Gamma -> Γ.
    The preferred input style is to write Unicode labels directly in KPOINTS.
    Only prime notation is normalized so that K', K′, and K$^\\prime$ are handled consistently.
    """
    if label is None:
        return ""

    s = str(label).strip().strip('"').strip()

    # Normalize common prime notations to the Unicode prime symbol.
    s = s.replace("$^{\\prime}$", "′")
    s = s.replace("$^\\prime$", "′")
    s = s.replace("^{\\prime}", "′")
    s = s.replace("^\\prime", "′")
    s = s.replace("\\prime", "′")
    s = s.replace("'", "′")

    # If the label is a simple math wrapper after prime normalization, strip the wrapper.
    # Example: $M′$ -> M′.  Do not strip wrappers such as $\\Gamma$.
    if s.startswith("$") and s.endswith("$") and s.count("$") == 2:
        inner = s[1:-1].strip()
        if "\\" not in inner:
            s = inner

    return s.strip()

def _split_kpoints_coord_label(line):
    """Parse one KPOINTS line and return (coords, label).
    Supported examples:
        0.0 0.0 0.0 Γ
        0.5 0.0 0.0 K'
        0.5 0.0 0.0 K'
        0.5 0.0 0.0 ! K'
    Returns (None, None) when the line does not contain a valid coordinate-label pair.
    """
    tokens = line.strip().split()
    if len(tokens) < 4:
        return None, None
    try:
        coords = (float(tokens[0]), float(tokens[1]), float(tokens[2]))
    except ValueError:
        return None, None
    if tokens[3] == "!":
        label = " ".join(tokens[4:]).strip() if len(tokens) > 4 else ""
    else:
        label = tokens[3]
    label = _clean_kpoints_label(label)
    if not label:
        return None, None

    return coords, label

def _select_kpoints_file(directory):
    """
    Select the k-point path file.
    For HSE06 band structure, KPOINTS_OPT contains the line-mode path.
    For ordinary GGA/PBE band structure, KPOINTS contains the line-mode path.
    """
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    kpoints_path = os.path.join(directory, "KPOINTS")

    if os.path.exists(kpoints_opt_path):
        return kpoints_opt_path

    if os.path.exists(kpoints_path):
        return kpoints_path

    return None

# extract bands

def extract_bandgap_outcar(directory="."):
    """
    Extract the bandgap, LUMO, and HOMO values from the OUTCAR file and return as a dictionary.
    Parameters:
        directory (str): Path to the directory containing the VASP output files (default is current directory).
    Returns:
        dict: A dictionary containing:
            - "bandgap": The bandgap value in eV.
            - "HOMO index": The index of the HOMO band.
            - "HOMO energy": The energy of the HOMO band in eV.
            - "LUMO index": The index of the LUMO band.
            - "LUMO energy": The energy of the LUMO band in eV.
        str: Error message if required data is missing or cannot be processed.
    """
    outcar_path = os.path.join(directory, "OUTCAR")
    # Check if OUTCAR exists
    if not os.path.exists(outcar_path):
        return "Error: OUTCAR not found in the specified directory."
    try:
        with open(outcar_path, "r") as file:
            lines = file.readlines()
        # Extract NELECT and NKPTS
        nelect = None
        nkpts = None
        for line in lines:
            if "NELECT" in line:
                nelect = float(line.split()[2]) / 2  # HOMO band index
            elif "NKPTS" in line:
                nkpts = int(line.split()[3])  # Total k-points
        if nelect is None or nkpts is None:
            return "Error: Could not extract NELECT or NKPTS from OUTCAR."
        # Calculate HOMO and LUMO band indices
        homo_band = int(nelect)
        lumo_band = homo_band + 1
        # Extract HOMO and LUMO energies
        homo_energies = []
        lumo_energies = []
        for line in lines:
            if f"{homo_band:5d}" in line:  # Strictly match HOMO band
                try:
                    homo_energies.append(float(line.split()[1]))
                except (ValueError, IndexError):
                    pass
            elif f"{lumo_band:5d}" in line:  # Strictly match LUMO band
                try:
                    lumo_energies.append(float(line.split()[1]))
                except (ValueError, IndexError):
                    pass
        if not homo_energies or not lumo_energies:
            return "Error: Could not extract HOMO or LUMO energies from OUTCAR."
        # Sort HOMO energies and take the last (maximum)
        homo_energy = sorted(homo_energies)[-1]
        # Sort LUMO energies and take the first (minimum)
        lumo_energy = sorted(lumo_energies)[0]
        # Calculate bandgap
        bandgap = lumo_energy - homo_energy
        return {
            "bandgap": bandgap,
            "HOMO index": homo_band,
            "HOMO energy": homo_energy,
            "HOMO": homo_energy,
            "LUMO index": lumo_band,
            "LUMO energy": lumo_energy,
            "LUMO": lumo_energy,
        }
    except Exception as e:
        return f"Error: {str(e)}"

def extract_bandgap_OUTCAR(*args):
    return extract_bandgap_outcar(*args)

def is_kpoints_returning(directory):
    """
    Check if the last high symmetry point in the KPOINTS or KPOINTS_OPT file
    returns to the starting point.
    Args:
        directory (str): The directory path containing the KPOINTS or KPOINTS_OPT file.
    Returns:
        bool: True if the last high symmetry point returns to the starting point, False otherwise.
    """
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    kpoints_file = None

    # Determine which file to use
    if os.path.exists(kpoints_opt_path):
        kpoints_file = kpoints_opt_path
    elif os.path.exists(kpoints_file_path):
        kpoints_file = kpoints_file_path
    else: return False
    try:
        with open(kpoints_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
        # Ensure it's a line-mode KPOINTS file
        if lines[2][0].lower() != "l":
            return False
        # Extract high-symmetry labels.  Keep Unicode labels such as Γ, K′, and M′.
        high_symmetry_points = []
        for line in lines[4:]:
            _coords, label = _split_kpoints_coord_label(line)
            if label:
                high_symmetry_points.append(label)
        # Check if the first and last points are the same
        return high_symmetry_points and high_symmetry_points[0] == high_symmetry_points[-1]
    except Exception:
        return False

# def extract_reciprocal_weights(directory):
#     """
#     Extract reciprocal lattice weights from the CONTCAR file in the given directory.
#     Args:
#     directory (str): The directory containing the CONTCAR file.
#     Returns:
#     list: A list of weights representing the relative lengths of the reciprocal lattice vectors.
#     """
#     # Read CONTCAR file
#     contcar_path = f"{directory}/CONTCAR"
#     with open(contcar_path, "r") as file:
#         lines = file.readlines()
#     # Extract lattice vectors
#     lattice_vectors = np.array([list(map(float, line.split())) for line in lines[2:5]])
#     # Calculate reciprocal lattice vectors
#     volume = np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2]))
#     reciprocal_lattice_vectors = 2 * np.pi * np.array([
#         np.cross(lattice_vectors[1], lattice_vectors[2]) / volume,
#         np.cross(lattice_vectors[2], lattice_vectors[0]) / volume,
#         np.cross(lattice_vectors[0], lattice_vectors[1]) / volume
#     ])
#     # Compute the lengths of the reciprocal lattice vectors
#     reciprocal_lengths = [np.linalg.norm(vec) for vec in reciprocal_lattice_vectors]
#     return reciprocal_lengths

def extract_reciprocal_weights(directory):
    """
    Backward-compatible helper.

    Returns only |b1|, |b2|, |b3|.  Do not use this for exact k-path distances
    in non-orthogonal cells.
    """
    reciprocal_lattice_vectors = extract_reciprocal_lattice(directory)
    reciprocal_lengths = np.linalg.norm(reciprocal_lattice_vectors, axis=1)
    return reciprocal_lengths.tolist()

def extract_high_sym(directory):
    """
    Extracts the high symmetry lines from the KPOINTS file in a VASP calculation directory,
    removing duplicate points except for the first and last.
    """
    # Open and read the KPOINTS file
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    if os.path.exists(kpoints_opt_path):
        kpoints_file = kpoints_opt_path
    elif os.path.exists(kpoints_file_path):
        kpoints_file = kpoints_file_path
    else:
        raise FileNotFoundError("KPOINTS file not found in the directory.")
    with open(kpoints_file, "r", encoding="utf-8") as file:
        KPOINTS = file.readlines()
    # Check if the KPOINTS file is in line-mode
    if KPOINTS[2][0] not in ("l", "L"):
        raise ValueError(f"Expected 'L' on the third line of KPOINTS file, got: {KPOINTS[2]}")
    # Initialize a list to store high symmetry points
    high_symmetry_points = []
    # Read the high-symmetry labels from the KPOINTS file.
    for i in range(4, len(KPOINTS)):
        _coords, label = _split_kpoints_coord_label(KPOINTS[i])
        if label:
            high_symmetry_points.append(label)
    # Remove duplicates except for the first and last points
    if len(high_symmetry_points) > 2:
        unique_points = [high_symmetry_points[0]]   # Keep the first point
        seen = set(unique_points)
        for point in high_symmetry_points[1:-1]:        # Process middle points
            if point not in seen:
                unique_points.append(point)
                seen.add(point)
        unique_points.append(high_symmetry_points[-1])  # Keep the last point
    else:
        unique_points = high_symmetry_points            # If only two points, return as is
    return unique_points

def extract_reciprocal_lattice(directory):
    """
    Extract full reciprocal lattice vectors from CONTCAR, falling back to POSCAR.
    Returns:
        numpy.ndarray with shape (3, 3), whose rows are b1, b2, b3.
        A fractional reciprocal coordinate q = (q1, q2, q3) is converted by:

            q_cart = q @ reciprocal_lattice_vectors
    """
    contcar_path = os.path.join(directory, "CONTCAR")
    poscar_path = os.path.join(directory, "POSCAR")
    if os.path.exists(contcar_path):
        cell_path = contcar_path
    elif os.path.exists(poscar_path):
        cell_path = poscar_path
    else:
        raise FileNotFoundError("Neither CONTCAR nor POSCAR found in the directory.")
    with open(cell_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    scale = float(lines[1].split()[0])
    lattice_vectors = np.array([
        list(map(float, lines[2].split()[:3])),
        list(map(float, lines[3].split()[:3])),
        list(map(float, lines[4].split()[:3])),
    ], dtype=float)
    # VASP scale factor.
    # Positive scale means ordinary multiplicative scale.
    # Negative scale means target cell volume.
    if scale > 0:
        lattice_vectors *= scale
    else:
        current_volume = abs(np.linalg.det(lattice_vectors))
        target_volume = abs(scale)
        lattice_vectors *= (target_volume / current_volume) ** (1.0 / 3.0)
    a1, a2, a3 = lattice_vectors
    volume = np.dot(a1, np.cross(a2, a3))
    reciprocal_lattice_vectors = 2 * np.pi * np.array([
        np.cross(a2, a3) / volume,
        np.cross(a3, a1) / volume,
        np.cross(a1, a2) / volume,
    ])
    # return extract_reciprocal_weights
    return reciprocal_lattice_vectors

def extract_high_sym_details_xml(directory):
    """
    Extracts the list of k-point coordinates from the vasprun.xml file of a VASP calculation.

    Args:
    directory (str): The directory path that contains the VASP vasprun.xml file.
    
    Returns:
    list: A list of k-point coordinates where each k-point is represented as a list of its coordinates.
    
    The function parses the vasprun.xml file to find the k-point coordinates used in the calculation.
    It looks for the 'varray' XML element with the attribute name set to 'kpointlist', which contains
    the k-point data. Each k-point is then extracted and appended to a list, which is returned.
    """
    # Construct the full path to the vasprun.xml file
    xml_file = os.path.join(directory, "vasprun.xml")
    # Parse the XML file
    tree = ET.parse(xml_file)
    # Get the root of the XML tree
    root = tree.getroot()
    # Initialize a list to store the k-point coordinates
    kpoints = []
    # Find all 'v' elements within 'varray' elements that have a 'name' attribute equal to 'kpointlist'
    # These elements contain the k-point coordinates
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        for kpoint in root.findall("./calculation/eigenvalues_kpoints_opt[@comment='kpoints_opt']/kpoints/varray[@name='kpointlist']/v"):
            # Split the text content of the 'v' element to get the individual coordinate strings
            # Convert each coordinate string to a float and create a list of coordinates
            coords = [float(x) for x in kpoint.text.split()]
            # Append the list of coordinates to the kpoints list
            kpoints.append(coords)
    # GGA-PBE algorithms
    elif os.path.exists(kpoints_file_path):
        for kpoint in root.findall(".//varray[@name='kpointlist']/v"):
            # Split the text content of the 'v' element to get the individual coordinate strings
            # Convert each coordinate string to a float and create a list of coordinates
            coords = [float(x) for x in kpoint.text.split()]
            # Append the list of coordinates to the kpoints list
            kpoints.append(coords)
    # Return the list of k-point coordinates
    return kpoints

def extract_high_sym_details_hdf5(directory):
    """
    Extract k-point coordinates from the HDF5 file.
    Data is assumed to be stored in '/calculation/kpoints/kpointlist' as a 2D array (nkpoints x 3).
    """
    h5_path = os.path.join(directory, "vaspout.h5")
    with h5py.File(h5_path, "r") as f:
        kp_data = f["calculation/kpoints/kpointlist"][:]
    kpoints = [row.tolist() for row in kp_data]
    return kpoints

def extract_kpath_no_weight(directory):
    """
    Calculates the cumulative distances along a path through k-points in reciprocal space.

    Args:
    directory (str): The directory path that contains the VASP vasprun.xml file.

    Returns:
    list: A list of cumulative distances for the path through the k-points.

    This function uses the list of k-point coordinates extracted from the vasprun.xml file
    and computes the Euclidean distance between successive k-points. These distances are
    then summed cumulatively to provide a measure of the total path length traversed up to
    each k-point in the list.

    The resulting cumulative distances serve as the x-axis values (k-points) in a bandstructure plot.
    """
    # Extract the list of k-point coordinates
    # changing flag
    kpoints = extract_high_sym_details_xml(directory)
    # Initialize the list for cumulative distances with the starting point (0 distance)
    cumulative_distances = [0]
    # Iterate over the list of k-points to calculate the path distances
    for i in range(1, len(kpoints)):
        # Calculate the Euclidean distance between successive k-points
        distance = np.linalg.norm(np.array(kpoints[i]) - np.array(kpoints[i-1]))
        # Add the distance from the previous total to get the new cumulative distance
        cumulative_distances.append(cumulative_distances[-1] + distance)
    # Return the list of cumulative distances
    return cumulative_distances

def _parse_line_mode_kpoints_segments(directory):
    """
    Parse VASP KPOINTS/KPOINTS_OPT in line-mode and return
    (n_per_segment, segments).
    segments:
        [((label_start, [kx, ky, kz]), (label_end, [kx, ky, kz])), ...]
    Returns (None, None) if no line-mode k-point path is found.
    """
    kpoints_file = _select_kpoints_file(directory)
    if kpoints_file is None:
        return None, None
    with open(kpoints_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) < 4:
        return None, None
    try:
        n_per = int(lines[1].strip().split()[0])
    except Exception:
        return None, None
    mode = lines[2].strip().lower()
    if not mode.startswith("l"):
        return None, None
    endpoints = []
    for line in lines[4:]:
        coords, label = _split_kpoints_coord_label(line)
        if coords is None or not label:
            continue
        endpoints.append((label, list(coords)))
    if len(endpoints) < 2:
        return None, None
    segments = []
    for i in range(0, len(endpoints) - 1, 2):
        segments.append((endpoints[i], endpoints[i + 1]))
    return n_per, segments

def _segment_break_indices_from_kpoints(directory, nk_from_vasprun, tol=1e-10):
    """Return indices where the next segment restarts from a different point.
    The returned indices are positions in the concatenated k-point list (as VASP emits for line-mode):
    insert a break *before* that index (between index-1 and index).
    """
    n_per, segments = _parse_line_mode_kpoints_segments(directory)
    if n_per is None or segments is None:
        return []
    expected_nk = n_per * len(segments)
    if nk_from_vasprun is not None and expected_nk != nk_from_vasprun:
        # If mismatch, do not trust segment arithmetic (e.g., non-standard producer). Fall back later.
        return []
    breaks = []
    for si in range(len(segments) - 1):
        _start_i, end_i = segments[si]
        start_next, _end_next = segments[si + 1]
        if np.linalg.norm(np.array(end_i[1]) - np.array(start_next[1])) > tol:
            breaks.append((si + 1) * n_per)
    return breaks

def _jump_break_indices_from_klist(kpoints, reciprocal_lattice=None, jump_factor=5.0, tol=1e-12):
    """Heuristic fallback: detect discontinuities by unusually large steps."""
    k = np.asarray(kpoints, dtype=float)
    if len(k) < 2:
        return []
    dk = np.diff(k, axis=0)
    if reciprocal_lattice is not None:
        B = np.asarray(reciprocal_lattice, dtype=float)
        dk_cart = dk @ B
    else:
        dk_cart = dk
    steps = np.linalg.norm(dk_cart, axis=1)
    nz = steps[steps > tol]
    if nz.size == 0:
        return []
    typical = float(np.median(nz))
    if typical < tol:
        return []
    return (np.where(steps > jump_factor * typical)[0] + 1).tolist()

def _apply_breaks_insert_nan(path, breaks, *band_groups):
    """Insert a separator point at each break: duplicate x and insert NaN in y.
    This yields a visual line break while preserving both segment endpoints.
    Args:
      path: list[float]
      breaks: list[int] indices in the *original* arrays.
      band_groups: each is list[list[float]] e.g. bands, conduction_bands, valence_bands
    Returns:
      (path_new, *groups_new)
    """
    if not breaks:
        return (path, *band_groups)
    path_new = list(path)
    groups_new = []
    for g in band_groups:
        groups_new.append([list(b) for b in g])
    # Insert from back to front so indices stay valid
    for b in sorted(breaks, reverse=True):
        if b <= 0:
            xval = path_new[0] if path_new else 0.0
        else:
            xval = path_new[b - 1]
        path_new.insert(b, xval)
        for g in groups_new:
            for band in g:
                band.insert(b, float("nan"))
    return (path_new, *groups_new)

def extract_kpath(directory, return_breaks=False, jump_factor=5.0):
    """
    Calculate cumulative reciprocal-space distances along a k-point path.
    This version uses the full reciprocal lattice metric, so it works correctly
    for non-orthogonal cells and is compatible with KPOINTS_OPT-based HSE06
    calculations.
    """
    kpoints = extract_high_sym_details_xml(directory)

    reciprocal_lattice_vectors = extract_reciprocal_lattice(directory)
    nk = len(kpoints)
    breaks = _segment_break_indices_from_kpoints(directory, nk_from_vasprun=nk)
    if not breaks:
        breaks = _jump_break_indices_from_klist(
            kpoints,
            reciprocal_lattice=reciprocal_lattice_vectors,
            jump_factor=jump_factor,
        )
    break_set = set(breaks)
    cumulative_distances = [0.0]
    for i in range(1, nk):
        delta_frac = np.array(kpoints[i], dtype=float) - np.array(kpoints[i - 1], dtype=float)
        # fractional reciprocal coordinate -> Cartesian reciprocal vector
        delta_cart = delta_frac @ reciprocal_lattice_vectors
        distance = np.linalg.norm(delta_cart)
        if i in break_set:
            distance = 0.0

        cumulative_distances.append(cumulative_distances[-1] + float(distance))
    if return_breaks:
        return cumulative_distances, breaks
    return cumulative_distances

def extract_high_symlines(directory):
    """
    Extracts the high symmetry lines from the KPOINTS file in a VASP calculation directory.

    Args:
    directory (str): The directory path that contains the VASP KPOINTS file.
    
    Returns:
    tuple: A tuple containing the kpoints format, number of high symmetry lines, 
           a set of high symmetry points, and a list of limit points for each line.
    
    This function opens the KPOINTS file and reads the high symmetry lines specified within it.
    It checks for the expected format and extracts the high symmetry points and their limits.
    """
    # Open and read the KPOINTS file
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    if os.path.exists(kpoints_opt_path):
        kpoints_file = kpoints_opt_path
    elif os.path.exists(kpoints_file_path):
        kpoints_file = kpoints_file_path

    with open(kpoints_file, "r", encoding="utf-8") as file:
        KPOINTS = file.readlines()
    # Check if the KPOINTS file is in line-mode
    if KPOINTS[2][0] not in ("l", "L"):
        raise ValueError(f"Expected 'L' on the third line of KPOINTS file, got: {KPOINTS[2]}")
    # Determine the format of the kpoints (cartesian or reciprocal)
    kpoints_format = "cartesian" if KPOINTS[3][0] in ["c", "C"] else "reciprocal"
    # Initialize a set to store unique high symmetry points
    high_symmetry_points = set()
    # Read the high-symmetry labels from the KPOINTS file.
    for i in range(4, len(KPOINTS)):
        _coords, label = _split_kpoints_coord_label(KPOINTS[i])
        if label:
            high_symmetry_points.add(label)
    # The number of unique high symmetry lines
    lines = len(high_symmetry_points)
    # The set of high symmetry points
    sets = high_symmetry_points
    # Extract non-empty lines from the KPOINTS file
    non_empty_lines = [line.split() for line in KPOINTS[4:] if line.strip()]
    # Extract the start and end points for each high symmetry line
    limits = []
    for i in range(0, len(non_empty_lines), 2):
        start = non_empty_lines[i]
        end = non_empty_lines[i+1]
        limits.append([start, end])
    # Return the kpoints format, number of lines, set of high symmetry points, and their limits
    return kpoints_format, lines, list(sets), limits

def extract_kpoints_eigenval(directory):
    """
    Extracts k-point coordinates from a VASP EIGENVAL file.

    Args:
    directory (str): The directory path that contains the VASP EIGENVAL file.

    Returns:
    numpy.ndarray: An array of k-point coordinates.

    The function reads the EIGENVAL file, which contains the eigenvalues for each k-point and band.
    It extracts the k-point coordinates from this file and returns them as a NumPy array.
    """
    # Open the EIGENVAL file
    with open(os.path.join(directory, "EIGENVAL"), "r", encoding="utf-8") as file:
        lines = file.readlines()
    # Initialize the list for k-points
    kpoints_list = []
    # Get the total number of bands and k-points from the sixth line of the file
    try:
        num_bands = int(lines[5].split()[2])
        num_kpoints = int(lines[5].split()[1])
    except IndexError as exc:
        # If the expected format is not found, raise an error
        raise ValueError("The EIGENVAL file does not have the expected format.") from exc
    # Calculate the number of lines in each k-point block (including the k-point line itself)
    block_size = num_bands + 1
    # Iterate over the EIGENVAL file to extract k-point coordinates
    # The k-point blocks start from line 7 (index 6) and are spaced by the block size
    for i in range(6, 6 + num_kpoints * block_size, block_size):
        # Extract the k-point coordinates from the first line of each block
        kpoint_line = lines[i].strip().split()
        # Check if there are enough elements in the line to represent a k-point
        if len(kpoint_line) < 4:
            continue  # Skip lines that don't have enough elements
        # Take the first three values as k-point coordinates (ignoring the weight)
        kpoint_coords = [float(kpoint_line[j]) for j in range(3)]
        kpoints_list.append(kpoint_coords)
    # Convert the k-point list to a NumPy array for efficiency
    kpoints_array = np.array(kpoints_list)
    return kpoints_array

def extract_weight_xml(directory):
    xml_file = os.path.join(directory, "vasprun.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")

    weight_list = []
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        for weight in root.findall(".//eigenvalues_kpoints_opt[@comment='kpoints_opt']/kpoints/varray[@name='weights']/v"): # <varray name="weights" >
            weight_list.append(float(weight.text))
    # GGA-PBE algorithms
    elif os.path.exists(kpoints_file_path):
        for weight in root.findall(".//varray[@name='weights']/v"): # <varray name="weights" >
            weight_list.append(float(weight.text))
    return weight_list

def extract_weight_hdf5(directory):
    h5_file = os.path.join(directory, "vaspout.h5")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    weight_list = []
    with h5py.File(h5_file, "r") as f:
        if os.path.exists(kpoints_opt_path):
            # HSE06 algorithm: read weights from the corresponding group
            weight_data = f["/calculation/eigenvalues_kpoints_opt/kpoints/weights"][:]
            weight_list = weight_data.tolist()
        elif os.path.exists(kpoints_file_path):
            # GGA-PBE algorithm: read weights from the corresponding group
            weight_data = f["/calculation/kpoints/weights"][:]
            weight_list = weight_data.tolist()
    return weight_list

def extract_kpoints_count_xml(directory):
    xml_file = os.path.join(directory, "vasprun.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    # Find the kpoints varray
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        kpoints_varray = root.find("./calculation/eigenvalues_kpoints_opt[@comment='kpoints_opt']/kpoints/varray[@name='kpointlist']")
    # GGA-PBE algorithms
    elif os.path.exists(kpoints_file_path):
        kpoints_varray = root.find(".//kpoints/varray[@name='kpointlist']")
    # Check if the varray exists
    if kpoints_varray is not None:
        # The number of kpoints is the number of <v> tags within the varray
        num_kpoints = len(kpoints_varray.findall("./v"))
        return num_kpoints
    else:
        print("The kpointlist section does not exist in the provided XML file.")
        return None

def extract_kpoints_count_hdf5(directory):
    h5_path = os.path.join(directory, "vaspout.h5")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    with h5py.File(h5_path, "r") as f:
        if os.path.exists(kpoints_opt_path):
            try:
                kpointlist_ds = f["/calculation/eigenvalues_kpoints_opt/kpoints/kpointlist"]
            except KeyError:
                print("Dataset '/calculation/eigenvalues_kpoints_opt/kpoints/kpointlist' not found.")
                return None
        elif os.path.exists(kpoints_file_path):
            try:
                kpointlist_ds = f["/calculation/kpoints/kpointlist"]
            except KeyError:
                print("Dataset '/calculation/kpoints/kpointlist' not found.")
                return None
        else:
            print("Neither KPOINTS_OPT nor KPOINTS files exist in the directory.")
            return None

        num_kpoints = kpointlist_ds.shape[0]
        return num_kpoints

def extract_bands_count(directory):
    eigen_lines = extract_eigenvalues_bands_nonpolarized(directory)
    return len(eigen_lines)

def kpoints_coordinate(directory):
    """Return a dictionary mapping KPOINTS labels to coordinates.

    The parser preserves Unicode labels such as Γ, K′, and M′.  Repeated labels map to
    the last occurrence, which matches the historical behavior of this helper.  For
    repeated high-symmetry boundaries in plots, use kpoints_path_lists instead.
    """
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")

    if os.path.exists(kpoints_opt_path):
        kpoints_file = kpoints_opt_path
    elif os.path.exists(kpoints_file_path):
        kpoints_file = kpoints_file_path
    else:
        raise FileNotFoundError("KPOINTS file not found in the directory.")

    high_symmetry_points = {}
    with open(kpoints_file, "r", encoding="utf-8") as file:
        for line in file.readlines()[4:]:
            coords, label = _split_kpoints_coord_label(line)
            if coords is not None and label:
                high_symmetry_points[label] = coords

    return high_symmetry_points

def kpoints_index(directory):
    # Retrieve the coordinates of the high symmetry points
    high_symmetry_points = kpoints_coordinate(directory)
    # Retrieve the list of kpoints
    # changing flag
    kpoints_list = extract_high_sym_details_xml(directory)
    # Initialize a dictionary to store the indices of the high symmetry points
    high_symmetry_indices = {}
    # For each high symmetry point, find the closest kpoint
    for label, coord in high_symmetry_points.items():
        # Initialize a minimum distance to a very large number so any actual distance will be smaller
        min_distance = float("inf")
        min_index = None
        # Iterate over the kpoint list to find the kpoint closest to the current high symmetry point coordinates
        for index, kpoint in enumerate(kpoints_list):
            # Calculate the Euclidean distance
            distance = sum((c - k) ** 2 for c, k in zip(coord, kpoint)) ** 0.5
            # If this distance is the smallest so far, update the minimum distance and index
            if distance < min_distance:
                min_distance = distance
                min_index = index
        # Store the index of the closest kpoint
        high_symmetry_indices[label] = min_index
    return high_symmetry_indices

def kpoints_path(directory):
    """
    Maps high symmetry points to cumulative distances along the k-points path.
    Args:
    directory (str): Path to the directory containing the VASP files.
    Returns:
    dict: A dictionary mapping high symmetry point labels to path distances.
    """
    # Extract high symmetry point indices and cumulative distances
    high_symmetry_indices = kpoints_index(directory)
    path_distances = extract_kpath(directory)
    # Map high symmetry points to their cumulative distances
    high_symmetry_paths = {
        label: path_distances[index] for label, index in high_symmetry_indices.items()
    }
    return high_symmetry_paths


def extract_kpoints_high_sym_boundaries(directory, return_coords=False):
    """Parse line-mode KPOINTS/KPOINTS_OPT and return high-symmetry boundaries.

    This implementation keeps repeated labels and labels with prime symbols, such as
    Γ, K′, and M′.  It does not use str.isalpha(), because prime symbols are not
    alphabetic and would be incorrectly filtered out.

    If return_coords is False, return [label, ...].
    If return_coords is True, return [(label, (kx, ky, kz)), ...].
    """
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")

    if os.path.exists(kpoints_opt_path):
        kpoints_file = kpoints_opt_path
    elif os.path.exists(kpoints_file_path):
        kpoints_file = kpoints_file_path
    else:
        raise FileNotFoundError("KPOINTS file not found in the directory.")

    with open(kpoints_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 4 or lines[2].strip()[:1].lower() != "l":
        raise ValueError(
            f"Expected a line-mode KPOINTS file (3rd line starts with 'L'), got: "
            f"{lines[2] if len(lines) > 2 else '<missing>'}"
        )

    endpoints = []
    for line in lines[4:]:
        coords, label = _split_kpoints_coord_label(line)
        if coords is None:
            continue
        endpoints.append((label, coords))

    if not endpoints:
        return []

    # Pair endpoints into line segments: (start, end), (start, end), ...
    segments = []
    for i in range(0, len(endpoints) - 1, 2):
        segments.append((endpoints[i], endpoints[i + 1]))

    if not segments:
        return []

    boundaries = [segments[0][0]]
    for si, (_start, end) in enumerate(segments):
        boundaries.append(end)

        # If the next segment starts from a different k-point, keep that restart point.
        # This supports branched paths and disconnected path pieces.
        if si + 1 < len(segments):
            next_start = segments[si + 1][0]
            if np.linalg.norm(np.array(next_start[1]) - np.array(end[1])) > 1e-10:
                boundaries.append(next_start)

    if return_coords:
        return boundaries

    return [label for (label, _coords) in boundaries]

def kpoints_path_lists(directory):
    """Return (positions, labels) for x-ticks along the bandstructure path.
    Unlike kpoints_path(), this keeps repeated labels (e.g., Γ ... Γ) and handles branched paths
    by matching boundary points monotonically along the k-point list extracted from vasprun.xml.
    """
    boundaries = extract_kpoints_high_sym_boundaries(directory, return_coords=True)
    if not boundaries:
        return [], []
    kpoints_list = extract_high_sym_details_xml(directory)
    path_distances = extract_kpath(directory)
    if not kpoints_list or not path_distances:
        return [], []
    positions = []
    labels = []
    last_search_start = 0
    for label, hs_coord in boundaries:
        min_dist = float("inf")
        best_idx = None
        hs = np.array(hs_coord, dtype=float)
        for index, kp in enumerate(kpoints_list[last_search_start:], start=last_search_start):
            d = np.linalg.norm(hs - np.array(kp, dtype=float))
            if d < min_dist:
                min_dist = d
                best_idx = index
                if min_dist < 1e-12:
                    break
        if best_idx is None or best_idx >= len(path_distances):
            continue
        pos = path_distances[best_idx]
        # Merge if multiple boundaries map to the same position (segment boundary duplication)
        if positions and abs(pos - positions[-1]) < 1e-10:
            if labels[-1] != label:
                labels[-1] = f"{labels[-1]}|{label}"
        else:
            positions.append(pos)
            labels.append(label)
        last_search_start = best_idx + 1
    return positions, labels

def high_symmetry_coordinates(directory):
    """
    This function extracts the coordinates of high symmetry points from the KPOINTS file.

    Args:
    directory (str): The directory path that contains the VASP KPOINTS file.
    
    Returns:
    list: A list of coordinates for the high symmetry points in the Brillouin zone.
    """
    # Retrieve the coordinates of the high symmetry points
    high_symmetry_points = kpoints_coordinate(directory)
    # Extract the coordinates from the dictionary and store them in a list
    coordinates_list = list(high_symmetry_points.values())
    return coordinates_list

def high_symmetry_path(directory):
    """
    This function extracts the x-axis values (cumulative path distances) for the high symmetry points
    in a bandstructure plot.

    Args:
    directory (str): The directory path that contains the VASP output files.
    Returns:
    list: A list of x-axis values for the high symmetry points in the bandstructure plot.
    """
    # Get the indices of high symmetry points in the k-point list
    high_symmetry_indices = kpoints_index(directory)
    # Calculate the cumulative path distances for the k-points
    path = extract_kpath(directory)
    # Initialize a list to store the x-axis values for the high symmetry points
    high_sym_path = []
    # Iterate over the high symmetry points and their indices
    for index in high_symmetry_indices.values():
        # Append the corresponding path distance to the list
        high_sym_path.append(path[index])
    # Return the list of x-axis values
    return high_sym_path

def clean_kpoints(kpoints_list, tol=1e-10):
    kpoints_list[np.isclose(kpoints_list, 0, atol=tol)] = 0
    return kpoints_list

def extract_eigenvalues_kpoints_xml(directory, spin_label):
    """
    Extracts the eigenvalues for each k-point from a VASP vasprun.xml file considering spin polarization.
    Args:
        directory (str): The directory path that contains the VASP vasprun.xml file.
        spin_label (str): The spin channel label ('spin1' or 'spin2').
    Returns:
        list of lists: A matrix where each sublist contains the eigenvalues for a specific k-point and spin channel.
    This function parses the vasprun.xml file to extract the electronic energy levels (eigenvalues)
    at each k-point in the reciprocal lattice for the material being studied, considering the specified spin channel.
    These eigenvalues are crucial for analyzing the material's electronic structure, such as plotting band structures.
    """
    xml_file = os.path.join(directory, "vasprun.xml")
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # Initialize a list to store the eigenvalues for each k-point
    eigenvalues_matrix = []
    # Find the eigenvalues section in the XML tree
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        eigenvalues_section = root.find("./calculation/eigenvalues_kpoints_opt[@comment='kpoints_opt']/eigenvalues")
    # GGA-PBE algorithms
    elif os.path.exists(kpoints_file_path):
        eigenvalues_section = root.find("./calculation/eigenvalues")
    if eigenvalues_section is not None:
        # Find all k-point <set> elements within the eigenvalues section
        # kpoint_sets = eigenvalues_section.findall(".//set/set/set")
        kpoint_sets = eigenvalues_section.findall(f".//set/set[@comment='{spin_label}']/set")
        if kpoint_sets:
            # Iterate over each k-point set to extract eigenvalues
            for kpoint_set in kpoint_sets:
                kpoint_eigenvalues = []
                # Iterate over each band's eigenvalue within the current k-point set
                for r in kpoint_set.findall("./r"):
                    # The energy eigenvalue is the first number in the <r> tag's text
                    energy = float(r.text.split()[0])
                    kpoint_eigenvalues.append(energy)
                # Append the list of eigenvalues for this k-point to the matrix
                eigenvalues_matrix.append(kpoint_eigenvalues)
        else:
            # Handle the case where no k-point <set> elements are found
            print("No k-point <set> elements found in the eigenvalues section.")
    else:
        # Handle the case where the eigenvalues section is missing
        print("Eigenvalues section not found in the XML file.")
    # Return the matrix of eigenvalues
    return eigenvalues_matrix

def extract_eigenvalues_kpoints_hdf5(directory, spin_label):
    """
    Extract eigenvalues for each k-point from the HDF5 file for the specified spin channel.
    Data is assumed to be stored in '/calculation/eigenvalues/{spin_label}' with shape (nkpoints x nbands).
    """
    h5_path = os.path.join(directory, "vaspout.h5")
    with h5py.File(h5_path, "r") as f:
        try:
            eigenvalues_data = f["calculation/eigenvalues"][spin_label][:]
        except KeyError:
            print(f"No eigenvalues data found for spin label: {spin_label}")
            return []
    return eigenvalues_data.tolist()

def extract_eigenvalues_kpoints_nonpolarized(directory):
    # changing flag
    return extract_eigenvalues_kpoints_xml(directory, "spin 1")

def extract_eigenvalues_kpoints_spinUp(directory):
    # changing flag
    return extract_eigenvalues_kpoints_xml(directory, "spin 1")

def extract_eigenvalues_kpoints_spinDown(directory):
    # changing flag
    return extract_eigenvalues_kpoints_xml(directory, "spin 2")

def extract_eigenvalues_bands(directory, spin_label):
    """
    Extracts and transposes the eigenvalues for each band from a VASP calculation.

    This function is designed to work with data from VASP (Vienna Ab initio Simulation Package) calculations. 
    It extracts the eigenvalues associated with each k-point for a given spin orientation (either 'spin1' or 'spin2'), 
    and then transposes the matrix so that each row represents a band and each column represents a k-point.
    Args:
        directory (str): The directory path that contains the VASP vasprun.xml file.
        spin_label (str): The spin label ('spin 1' or 'spin 2') for which the eigenvalues are to be extracted. 
                          'spin 1' typically refers to spin-up and 'spin 2' to spin-down in spin-polarized calculations.
    Returns:
        list of lists: A transposed matrix of eigenvalues where each row represents a band and each column represents a k-point. 
                       This format is useful for plotting band structures and analyzing the electronic properties of materials.
    Example:
        # Extract eigenvalues for 'spin 1' (spin-up) orientation
        directory = "/path/to/vasp/output"
        spin_label = "spin1"
        bands_matrix = extract_eigenvalues_bands(directory, spin_label)
        # 'bands_matrix' now contains the eigenvalues with bands as rows and k-points as columns
    """
    # Extract the eigenvalues for each k-point
    # changing flag
    eigenvalues_matrix = extract_eigenvalues_kpoints_xml(directory, spin_label)
    # Transpose the matrix so that bands are rows and k-points are columns
    transposed_eigenvalues_matrix = transpose_matrix(eigenvalues_matrix)
    # Return the transposed matrix of eigenvalues
    return transposed_eigenvalues_matrix

def extract_eigenvalues_bands_nonpolarized(directory):
    return extract_eigenvalues_bands(directory, "spin 1")

def extract_eigenvalues_bands_spinUp(directory):
    return extract_eigenvalues_bands(directory, "spin 1")

def extract_eigenvalues_bands_spinDown(directory):
    return extract_eigenvalues_bands(directory, "spin 2")

def extract_eigenvalues_conductionBands(directory, spin_label, TOLERANCE = global_tolerance):
    eigenvalues_matrix = extract_eigenvalues_bands(directory, spin_label)
    conduction_bands = []
    current_LUMO = extract_bandgap_outcar(directory)["LUMO energy"]
    current_HOMO = extract_bandgap_outcar(directory)["HOMO energy"]
    for eigenvalues_bands in eigenvalues_matrix:
        if np.min(eigenvalues_bands) >= current_LUMO-TOLERANCE:
            conduction_bands.append(eigenvalues_bands)
    return conduction_bands

def extract_eigenvalues_valenceBands(directory, spin_label, TOLERANCE = global_tolerance):
    eigenvalues_matrix = extract_eigenvalues_bands(directory, spin_label)
    valence_bands = []
    current_LUMO = extract_bandgap_outcar(directory)["LUMO energy"]
    current_HOMO = extract_bandgap_outcar(directory)["HOMO energy"]
    for eigenvalues_bands in eigenvalues_matrix:
        if np.max(eigenvalues_bands) <= current_HOMO+TOLERANCE:
            valence_bands.append(eigenvalues_bands)
    return valence_bands

def extract_eigenvalues_conductionBands_nonpolarized(directory, TOLERANCE):
    return extract_eigenvalues_conductionBands(directory, "spin 1", TOLERANCE)

def extract_eigenvalues_valenceBands_nonpolarized(directory, TOLERANCE):
    return extract_eigenvalues_valenceBands(directory, "spin 1", TOLERANCE)

def extract_eigenvalues_conductionBands_spinUp(directory, TOLERANCE):
    return extract_eigenvalues_conductionBands(directory, "spin 1", TOLERANCE)

def extract_eigenvalues_valenceBands_spinUp(directory, TOLERANCE):
    return extract_eigenvalues_valenceBands(directory, "spin 1", TOLERANCE)

def extract_eigenvalues_conductionBands_spinDown(directory, TOLERANCE):
    return extract_eigenvalues_conductionBands(directory, "spin 2", TOLERANCE)

def extract_eigenvalues_valenceBands_spinDown(directory, TOLERANCE):
    return extract_eigenvalues_valenceBands(directory, "spin 2", TOLERANCE)

def extract_high_sym_intersections(directory, spin_label):
    """
    Extracts the eigenvalues at high symmetry points for each band in a VASP bandstructure calculation,
    using the path value (x) in the bandstructure plot and the corresponding eigenvalue (y).
    Args:
        directory (str): The directory path that contains the VASP output files.
        spin_label (str): The spin label ('spin1' or 'spin2') for which the eigenvalues are to be extracted.
    Returns:
        dict: A dictionary where the keys are high symmetry points (e.g., 'Gamma', 'K', 'M'), and the values 
              are lists of tuples, where each tuple represents (path, eigenvalue) coordinates of intersection 
              points at that high symmetry point. 'path' is the x-coordinate, and 'eigenvalue' is the y-coordinate.
    """
    # Extract the eigenvalues (bands as rows, k-points as columns)
    eigenvalues_bands = extract_eigenvalues_bands(directory, spin_label)

    # Get the path distances for each k-point along the bandstructure path
    path = extract_kpath(directory)

    # Extract the high symmetry points and their positions in the bandstructure
    high_symmetry_indices = kpoints_index(directory)  # Get the indices of high symmetry points
    high_symmetry_labels = extract_high_sym(directory)  # Get high symmetry labels

    # Initialize a dictionary to store the intersection points
    intersections = {}

    # Iterate over each high symmetry point and its corresponding index
    for label, index in high_symmetry_indices.items():
        # Initialize a list to store (path, eigenvalue) coordinates at this high symmetry point
        intersections[label] = []
        for band in eigenvalues_bands:
            # x is the path value (cumulative distance along the bandstructure path)
            x = path[index]  # The x-axis value is the path (cumulative distance) at the high symmetry point
            # y is the eigenvalue at that point
            y = band[index]  # The y-axis value is the eigenvalue at that k-point
            # Append the (path, eigenvalue) coordinates to the list for this high symmetry point
            intersections[label].append((x, y))
    return intersections

def extract_high_sym_intersections_with_fermi(directory, spin_label):
    """
    Extracts the eigenvalues at high symmetry points for each band in a VASP bandstructure calculation,
    using the path value (x) in the bandstructure plot and the corresponding eigenvalue (y) minus the Fermi level.

    Args:
        directory (str): The directory path that contains the VASP output files.
        spin_label (str): The spin label ('spin1' or 'spin2') for which the eigenvalues are to be extracted.

    Returns:
        dict: A dictionary where the keys are high symmetry points (e.g., 'Gamma', 'K', 'M'), and the values 
              are lists of tuples, where each tuple represents (path, eigenvalue - Fermi) coordinates of intersection 
              points at that high symmetry point. 'path' is the x-coordinate, and 'eigenvalue' is the y-coordinate minus Fermi.
    """
    # Extract the eigenvalues (bands as rows, k-points as columns)
    eigenvalues_bands = extract_eigenvalues_bands(directory, spin_label)

    # Get the Fermi energy
    fermi_energy = extract_fermi(directory)

    # Get the path distances for each k-point along the bandstructure path
    path = extract_kpath(directory)

    # Extract the high symmetry points and their positions in the bandstructure
    high_symmetry_indices = kpoints_index(directory)    # Get the indices of high symmetry points
    high_symmetry_labels = extract_high_sym(directory)  # Get high symmetry labels

    # Initialize a dictionary to store the intersection points
    intersections = {}

    # Iterate over each high symmetry point and its corresponding index
    for label, index in high_symmetry_indices.items():
        # Initialize a list to store (path, eigenvalue - Fermi) coordinates at this high symmetry point
        intersections[label] = []
        for band in eigenvalues_bands:
            # x is the path value (cumulative distance along the bandstructure path)
            x = path[index]  # The x-axis value is the path (cumulative distance) at the high symmetry point
            # y is the eigenvalue at that point, minus the Fermi energy
            y = band[index] - fermi_energy  # The y-axis value is the eigenvalue at that k-point minus the Fermi level
            # Append the (path, eigenvalue - Fermi) coordinates to the list for this high symmetry point
            intersections[label].append((x, y))

    return intersections

def extract_high_sym_valence_intersections(directory, spin_label):
    """
    Extracts the valence band eigenvalues (y < 0) at high symmetry points for each band in a VASP 
    bandstructure calculation, using the path value (x) in the bandstructure plot and the corresponding 
    eigenvalue (y) minus the Fermi level.

    Args:
        directory (str): The directory path that contains the VASP output files.
        spin_label (str): The spin label ('spin1' or 'spin2') for which the eigenvalues are to be extracted.

    Returns:
        dict: A dictionary where the keys are high symmetry points (e.g., 'Gamma', 'K', 'M'), and the values 
              are lists of tuples, where each tuple represents (path, eigenvalue - Fermi) coordinates of valence 
              band intersection points (where eigenvalue - Fermi < 0).
    """
    intersections_with_fermi = extract_high_sym_intersections_with_fermi(directory, spin_label)
    valence_intersections = {}

    # Filter for valence band intersections (y < 0)
    for label, points in intersections_with_fermi.items():
        valence_intersections[label] = [point for point in points if point[1] < 0]

    return valence_intersections

def extract_high_sym_conduction_intersections(directory, spin_label):
    """
    Extracts the conduction band eigenvalues (y > 0) at high symmetry points for each band in a VASP 
    bandstructure calculation, using the path value (x) in the bandstructure plot and the corresponding 
    eigenvalue (y) minus the Fermi level.

    Args:
        directory (str): The directory path that contains the VASP output files.
        spin_label (str): The spin label ('spin1' or 'spin2') for which the eigenvalues are to be extracted.
    Returns:
        dict: A dictionary where the keys are high symmetry points (e.g., 'Gamma', 'K', 'M'), and the values 
              are lists of tuples, where each tuple represents (path, eigenvalue - Fermi) coordinates of conduction 
              band intersection points (where eigenvalue - Fermi > 0).
    """
    intersections_with_fermi = extract_high_sym_intersections_with_fermi(directory, spin_label)
    conduction_intersections = {}

    # Filter for conduction band intersections (y > 0)
    for label, points in intersections_with_fermi.items():
        conduction_intersections[label] = [point for point in points if point[1] > 0]

    return conduction_intersections

def extract_high_sym_min_conduction_intersections(directory, spin_label):
    """
    Extracts the lowest conduction band eigenvalue (minimum y > 0) at high symmetry points for each band 
    in a VASP bandstructure calculation, using the path value (x) in the bandstructure plot and the corresponding 
    eigenvalue (y) minus the Fermi level.

    Args:
        directory (str): The directory path that contains the VASP output files.
        spin_label (str): The spin label ('spin1' or 'spin2') for which the eigenvalues are to be extracted.
    Returns:
        dict: A dictionary where the keys are high symmetry points (e.g., 'Gamma', 'K', 'M'), and the values 
              are tuples representing (path, min_eigenvalue - Fermi) for the lowest conduction band intersection 
              point (where min_eigenvalue - Fermi > 0).
    """
    intersections_with_fermi = extract_high_sym_conduction_intersections(directory, spin_label)
    min_conduction_intersections = {}

    # Find the minimum conduction band eigenvalue (y > 0) for each high symmetry point
    for label, points in intersections_with_fermi.items():
        if points:  # If there are conduction intersections
            min_point = min(points, key=lambda point: point[1])  # Get the point with the minimum y
            min_conduction_intersections[label] = min_point

    return min_conduction_intersections

def extract_high_sym_max_valence_intersections(directory, spin_label):
    """
    Extracts the highest valence band eigenvalue (maximum y < 0) at high symmetry points for each band 
    in a VASP bandstructure calculation, using the path value (x) in the bandstructure plot and the corresponding 
    eigenvalue (y) minus the Fermi level.

    Args:
        directory (str): The directory path that contains the VASP output files.
        spin_label (str): The spin label ('spin1' or 'spin2') for which the eigenvalues are to be extracted.
    Returns:
        dict: A dictionary where the keys are high symmetry points (e.g., 'Gamma', 'K', 'M'), and the values 
              are tuples representing (path, max_eigenvalue - Fermi) for the highest valence band intersection 
              point (where max_eigenvalue - Fermi < 0).
    """
    intersections_with_fermi = extract_high_sym_valence_intersections(directory, spin_label)
    max_valence_intersections = {}

    # Find the maximum valence band eigenvalue (y < 0) for each high symmetry point
    for label, points in intersections_with_fermi.items():
        if points:  # If there are valence intersections
            max_point = max(points, key=lambda point: point[1])  # Get the point with the maximum y
            max_valence_intersections[label] = max_point

    return max_valence_intersections

# extract bands with weights

def extract_weights_kpoints_xml(directory, spin_label, start_label=None, end_label=None):
    """
    Extracts the projected weight of eigenvalues for different orbitals (s, p, d) for specified spin electrons from a VASP calculation.

    This function parses the 'vasprun.xml' file to extract the projected weight of eigenvalues
    for each orbital type (s, p, and d orbitals) at each k-point for specified spin electrons. 
    The weights are summed over a range of atoms if specified.

    Args:
    directory (str): The directory path containing the VASP output files, specifically 'vasprun.xml'.
    spin_label (str): The label of spin ('spin1' for spin-up, 'spin2' for spin-down).
    start_label (int, optional): The starting index of atoms to be included in the sum. Defaults to the first atom.
    end_label (int, optional): The ending index of atoms to be included in the sum. Defaults to the last atom.
    Returns:
    tuple of lists: Contains multiple lists, each representing the weight of eigenvalues for a specific orbital type
    across all k-points. The order is s, py, pz, px, dxy, dyz, dz2, dx2y2, total d, and total p orbitals.
    """
    # Construct the path to the vasprun.xml file and parse it
    xml_file = os.path.join(directory, "vasprun.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Atoms count
    atom_count = get_atoms_count(directory)

    # Initialize matrices to store the weight of eigenvalues for each orbital
    weights_kpoints_s = []
    weights_kpoints_py, weights_kpoints_pz, weights_kpoints_px = [], [], []
    weights_kpoints_dxy, weights_kpoints_dyz, weights_kpoints_dz2, weights_kpoints_dx2y2 = [], [], [], []
    weights_kpoints_d, weights_kpoints_p = [], []

    # Find the projected weight of eigenvalues section in the XML tree
    projected_section = root.find(".//projected/array")
    if projected_section is not None:
        # Find all k-point <set> elements within the projected section
        kpoint_sets = projected_section.findall(f".//set[@comment='{spin_label}']/set")
        for kpoint_set in kpoint_sets:
            weights_s, weights_py, weights_pz, weights_px = [], [], [], []
            weights_dxy, weights_dyz, weights_dz2, weights_dx2y2 = [], [], [], []
            for band_set in kpoint_set.findall(".//set"):
                r_elements = band_set.findall("./r")
                if r_elements:
                    # Extract and sum the weights for each orbital
                    if start_label is None:
                        start = 0
                    else: start = start_label
                    if end_label is None:
                        end = atom_count
                    else: end = end_label
                    weights_s.append(sum(float(r.text.split()[0]) for r in r_elements[start:end]))
                    weights_py.append(sum(float(r.text.split()[1]) for r in r_elements[start:end]))
                    weights_pz.append(sum(float(r.text.split()[2]) for r in r_elements[start:end]))
                    weights_px.append(sum(float(r.text.split()[3]) for r in r_elements[start:end]))
                    weights_dxy.append(sum(float(r.text.split()[4]) for r in r_elements[start:end]))
                    weights_dyz.append(sum(float(r.text.split()[5]) for r in r_elements[start:end]))
                    weights_dz2.append(sum(float(r.text.split()[6]) for r in r_elements[start:end]))
                    weights_dx2y2.append(sum(float(r.text.split()[7]) for r in r_elements[start:end]))
            # Sum of p and d orbitals for each k-point
            weights_d_kpoint = [sum(x) for x in zip(weights_dxy, weights_dyz, weights_dz2, weights_dx2y2)]
            weights_p_kpoint = [sum(x) for x in zip(weights_py, weights_pz, weights_px)]
            # Append weights for each orbital type
            weights_kpoints_s.append(weights_s)
            weights_kpoints_py.append(weights_py)
            weights_kpoints_pz.append(weights_pz)
            weights_kpoints_px.append(weights_px)
            weights_kpoints_dxy.append(weights_dxy)
            weights_kpoints_dyz.append(weights_dyz)
            weights_kpoints_dz2.append(weights_dz2)
            weights_kpoints_dx2y2.append(weights_dx2y2)
            weights_kpoints_d.append(weights_d_kpoint)
            weights_kpoints_p.append(weights_p_kpoint)
    else:
        print("Projected weight section not found in the XML file.")
    return (weights_kpoints_s,                                                                      # 0
            weights_kpoints_py, weights_kpoints_pz, weights_kpoints_px,                             # 1, 2, 3
            weights_kpoints_dxy, weights_kpoints_dyz, weights_kpoints_dz2, weights_kpoints_dx2y2,   # 4, 5, 6, 7
            weights_kpoints_d,                                                                      # -2
            weights_kpoints_p)                                                                      # -1

def extract_weights_kpoints_hdf5(directory, spin_label, start_label=None, end_label=None):
    """
    Extract orbital projection weights from the HDF5 file for the specified spin channel.
    Data is assumed to be stored in '/calculation/projected/array/{spin_label}' with shape
    (nkpoints, nbands, natoms, 8), where the 8 channels correspond to:
    [s, py, pz, px, dxy, dyz, dz2, dx2y2].
    The function sums the weights over atoms (or a specified atom range).
    """
    h5_path = os.path.join(directory, "vaspout.h5")
    with h5py.File(h5_path, "r") as f:
        try:
            proj_data = f["calculation/projected/array"][spin_label][:]
        except KeyError:
            print(f"No projection data found for spin label: {spin_label}")
            return ([], [], [], [], [], [], [], [], [], [])
    if start_label is None:
        start = 0
    else:
        start = start_label
    if end_label is None:
        end = proj_data.shape[2]
    else:
        end = end_label
    proj_subset = proj_data[:, :, start:end, :]  # Shape: (nkpoints, nbands, natoms_subset, 8)
    proj_sum = np.sum(proj_subset, axis=2)         # Sum over atoms -> Shape: (nkpoints, nbands, 8)
    weights_s = proj_sum[:, :, 0].tolist()
    weights_py = proj_sum[:, :, 1].tolist()
    weights_pz = proj_sum[:, :, 2].tolist()
    weights_px = proj_sum[:, :, 3].tolist()
    weights_dxy = proj_sum[:, :, 4].tolist()
    weights_dyz = proj_sum[:, :, 5].tolist()
    weights_dz2 = proj_sum[:, :, 6].tolist()
    weights_dx2y2 = proj_sum[:, :, 7].tolist()
    weights_d = np.add.reduce([proj_sum[:, :, 4], proj_sum[:, :, 5], proj_sum[:, :, 6], proj_sum[:, :, 7]]).tolist()
    weights_p = np.add.reduce([proj_sum[:, :, 1], proj_sum[:, :, 2], proj_sum[:, :, 3]]).tolist()
    return (weights_s, weights_py, weights_pz, weights_px,
            weights_dxy, weights_dyz, weights_dz2, weights_dx2y2,
            weights_d, weights_p)

def extract_weights_kpoints_nonpolarized(directory, start_label=None, end_label=None):
    # changing flag
    return extract_weights_kpoints_xml(directory, "spin1", start_label, end_label)

def extract_weights_kpoints_spinUp(directory, start_label=None, end_label=None):
    # changing flag
    return extract_weights_kpoints_xml(directory, "spin1", start_label, end_label)

def extract_weights_kpoints_spinDown(directory, start_label=None, end_label=None):
    # changing flag
    return extract_weights_kpoints_xml(directory, "spin2", start_label, end_label)

def extract_weights_bands(directory, spin_label, start_label=None, end_label=None):
    """
    Extracts and transposes the weight of eigenvalues for different orbitals across bands.

    This function is designed to work with VASP calculation outputs. It extracts the projected weight of eigenvalues
    for different orbitals (s, p, d) across bands for specified spin states (spin-up or spin-down). The function
    allows for the selection of a specific range of atoms by specifying start and end labels.

    Args:
    - directory (str): The directory path containing the 'vasprun.xml' file from a VASP calculation.
    - spin_label (str): Specifies the spin state. Use "spin1" for spin-up and "spin2" for spin-down.
    - start_label (int, optional): The starting index of atoms to consider for weight extraction. Defaults to None, which considers the first atom.
    - end_label (int, optional): The ending index of atoms to consider for weight extraction. Defaults to None, which considers up to the last atom.
    Returns:
    - tuple of lists: Each list within the tuple represents the transposed weight of eigenvalues for a specific orbital type across all bands. The order is:
        0: s orbital
        1: py orbital
        2: pz orbital
        3: px orbital
        4: dxy orbital
        5: dyz orbital
        6: dz2 orbital
        7: d(x2-y2) orbital
        -2: Total weight for all d orbitals
        -1: Total weight for all p orbitals
    
    Example Usage:
    # Extracting weights for spin-up electrons across all atoms
    weights_for_bands = extract_weights_bands("/path/to/directory", "spin1")
    s_orbital_weights = weights_for_bands[0]  # Weights for s orbital across bands
    """
    # changing flag
    weights_kpoints = extract_weights_kpoints_xml(directory, spin_label, start_label, end_label)
    weights_bands_s = transpose_matrix(weights_kpoints[0])
    weights_bands_py = transpose_matrix(weights_kpoints[1])
    weights_bands_pz = transpose_matrix(weights_kpoints[2])
    weights_bands_px = transpose_matrix(weights_kpoints[3])
    weights_bands_dxy = transpose_matrix(weights_kpoints[4])
    weights_bands_dyz = transpose_matrix(weights_kpoints[5])
    weights_bands_dz2 = transpose_matrix(weights_kpoints[6])
    weights_bands_dx2y2 = transpose_matrix(weights_kpoints[7])
    weights_bands_d = transpose_matrix(weights_kpoints[-2])
    weights_bands_p = transpose_matrix(weights_kpoints[-1])
    return (weights_bands_s,                                                                                # 0
            weights_bands_py, weights_bands_pz, weights_bands_px,                                           # 1, 2, 3
            weights_bands_dxy, weights_bands_dyz, weights_bands_dz2, weights_bands_dx2y2,                   # 4, 5, 6, 7
            weights_bands_d,                                                                                # -2
            weights_bands_p                                                                                 # -1
            )

def extract_weights_bands_nonpolarized(directory, start_label=None, end_label=None):
    return extract_weights_bands(directory, "spin1", start_label, end_label)

def extract_weights_bands_spinUp(directory, start_label=None, end_label=None):
    return extract_weights_bands(directory, "spin1", start_label, end_label)

def extract_weights_bands_spinDown(directory, start_label=None, end_label=None):
    return extract_weights_bands(directory, "spin2", start_label, end_label)

# plot bandstructure
def create_matters_bs(matters_list):
    # Ensure input is a list of lists
    if isinstance(matters_list, list) and matters_list and not any(isinstance(i, list) for i in matters_list):
        source_data = matters_list[:]
        matters_list.clear()
        matters_list.append(source_data)
    matters = []
    for current_matter in matters_list:
        bstype, label, directory, *optional = current_matter
        # Set default values using get_or_default
        color = get_or_default(optional[0] if len(optional) > 0 else None, "default")
        lstyle = get_or_default(optional[1] if len(optional) > 1 else None, "solid")
        weight = get_or_default(optional[2] if len(optional) > 2 else None, 1.5)
        alpha = get_or_default(optional[3] if len(optional) > 3 else None, 1.0)
        current_tolerance = get_or_default(optional[4] if len(optional) > 4 else None, 0)
        # Band structure plotting style: monocolor
        if bstype.lower() in ["monocolor", "monocolor nonpolarized"]:
            fermi_energy = extract_fermi(directory)
            kpath, breaks = extract_kpath(directory, return_breaks=True)
            bands = extract_eigenvalues_bands_nonpolarized(directory)
            kpath, bands = _apply_breaks_insert_nan(kpath, breaks, bands)
            matters.append([bstype, label, fermi_energy, kpath, bands, color, lstyle, weight, alpha, current_tolerance])
        elif bstype.lower() in ["monocolor spin up", "spin up monocolor"]:
            fermi_energy = extract_fermi(directory)
            kpath, breaks = extract_kpath(directory, return_breaks=True)
            bands = extract_eigenvalues_bands_spinUp(directory)
            kpath, bands = _apply_breaks_insert_nan(kpath, breaks, bands)
            matters.append([bstype, label, fermi_energy, kpath, bands, color, lstyle, weight, alpha, current_tolerance])
        elif bstype.lower() in ["monocolor spin down", "spin down monocolor"]:
            fermi_energy = extract_fermi(directory)
            kpath, breaks = extract_kpath(directory, return_breaks=True)
            bands = extract_eigenvalues_bands_spinDown(directory)
            kpath, bands = _apply_breaks_insert_nan(kpath, breaks, bands)
            matters.append([bstype, label, fermi_energy, kpath, bands, color, lstyle, weight, alpha, current_tolerance])
        # Band structure plotting style: bands
        elif bstype.lower() in ["bands", "bands nonpolarized"]:
            fermi_energy = extract_fermi(directory)
            kpath, breaks = extract_kpath(directory, return_breaks=True)
            conduction_bands = extract_eigenvalues_conductionBands_nonpolarized(directory, current_tolerance)
            valence_bands = extract_eigenvalues_valenceBands_nonpolarized(directory, current_tolerance)
            kpath, conduction_bands, valence_bands = _apply_breaks_insert_nan(kpath, breaks, conduction_bands, valence_bands)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, color, lstyle, weight, alpha, current_tolerance])
        elif bstype.lower() in ["bands spin up", "spin up bands"]:
            fermi_energy = extract_fermi(directory)
            kpath, breaks = extract_kpath(directory, return_breaks=True)
            conduction_bands = extract_eigenvalues_conductionBands_spinUp(directory, current_tolerance)
            valence_bands = extract_eigenvalues_valenceBands_spinUp(directory, current_tolerance)
            kpath, conduction_bands, valence_bands = _apply_breaks_insert_nan(kpath, breaks, conduction_bands, valence_bands)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, color, lstyle, weight, alpha, current_tolerance])
        elif bstype.lower() in ["bands spin down", "spin down bands"]:
            fermi_energy = extract_fermi(directory)
            kpath, breaks = extract_kpath(directory, return_breaks=True)
            conduction_bands = extract_eigenvalues_conductionBands_spinDown(directory, current_tolerance)
            valence_bands = extract_eigenvalues_valenceBands_spinDown(directory, current_tolerance)
            kpath, conduction_bands, valence_bands = _apply_breaks_insert_nan(kpath, breaks, conduction_bands, valence_bands)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, color, lstyle, weight, alpha, current_tolerance])
    return matters

def plot_bandstructure(title, matters_list=None, eigen_range=None, legend_loc=False):
    # Help information
    help_info = """
    Usage: plot_bandstructure
        arg[0]: title;
        arg[1]: the range of eigenvalues, from -arg[1] to arg[1];
        arg[2]: matters list;
        arg[3]: legend location;
    """
    if title in ["help", "Help"]:
        print(help_info)
        return

    # Figure settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Colors calling
    fermi_color = color_sampling("Violet")
    annotate_color = color_sampling("Grey")

    # Data calling and plotting
    matters = create_matters_bs(matters_list)
    for matter in matters:
        current_label = matter[1]
        if _is_monocolor_type(matter[0]):
            fermi = matter[2]
            for bands_index in range(0, len(matter[4])):
                current_band = [eigenvalue - fermi for eigenvalue in matter[4][bands_index]]
                if bands_index == 0:
                    plt.plot(matter[3], current_band, c=color_sampling(matter[5])[1], linestyle=matter[6], lw=matter[7], alpha=matter[8], label=f"{current_label}", zorder=4)
                else:
                    plt.plot(matter[3], current_band, c=color_sampling(matter[5])[1], linestyle=matter[6], lw=matter[7], alpha=matter[8], zorder=4)
        elif _is_bands_type(matter[0]):
            fermi = matter[2]
            for bands_index in range(0, len(matter[4])):
                current_conduction_band = [eigenvalue - fermi for eigenvalue in matter[4][bands_index]]
                if bands_index == 0:
                    plt.plot(matter[3], current_conduction_band, c=color_sampling(matter[6])[2], linestyle=matter[7], lw=matter[8], alpha=matter[9], label=f"Conduction bands for {current_label}", zorder=4)
                else:
                    plt.plot(matter[3], current_conduction_band, c=color_sampling(matter[6])[2], linestyle=matter[7], lw=matter[8], alpha=matter[9], zorder=4)
            for bands_index in range(0, len(matter[5])):
                current_valence_band = [eigenvalue - fermi for eigenvalue in matter[5][bands_index]]
                if bands_index == 0:
                    plt.plot(matter[3], current_valence_band, c=color_sampling(matter[6])[0], linestyle=matter[7], lw=matter[8], alpha=matter[9], label=f"Valence bands for {current_label}", zorder=4)
                else:
                    plt.plot(matter[3], current_valence_band, c=color_sampling(matter[6])[0], linestyle=matter[7], lw=matter[8], alpha=matter[9], zorder=4)
        kpath_start = matter[3][0]
        kpath_end = matter[3][-1]
        fermi_last = matter[2]

    # Fermi energy as a horizon line
    plt.axhline(y = 0, color=fermi_color[0], alpha=0.8, linestyle="--", label="Fermi energy", zorder=2)
    efermi = fermi_last
    kpath_range = kpath_end-kpath_start
    # fermi_energy_text = f"Fermi energy\n{efermi:.3f} (eV)"
    # plt.text(kpath_start+kpath_range*0.98, eigen_range*0.02, fermi_energy_text, fontsize=10, c=fermi_color[0], rotation=0, va = "bottom", ha="right", zorder=5)

    # Title
    plt.title(f"{title}")
    plt.ylabel("Energy (eV)")

    # y-axis range
    demo_boundary = process_boundary(eigen_range)
    if demo_boundary[0] is None:
        plt.ylim(demo_boundary[1]*(-1), demo_boundary[1])
    else: plt.ylim(demo_boundary[0], demo_boundary[1])

    # x-axis range
    plt.xlim(kpath_start, kpath_end)

    # High symmetry path
    directory = matters_list[-1][2]
    high_symmetry_positions, high_symmetry_labels = kpoints_path_lists(directory)
    plt.xticks(high_symmetry_positions, high_symmetry_labels)

    for k_loc in high_symmetry_positions[1:-1]:
        plt.axvline(x=k_loc, color=annotate_color[1], linestyle="--", alpha=0.8, zorder=1)

    # Legend
    if legend_loc is True:
        plt.legend(loc=legend_loc)
    elif legend_loc is None or legend_loc is False:
        # Do not display the legend
        pass

    plt.tight_layout()

# plot bandstructure with DoS
def create_matters_bsdos(matters_list):
    # Ensure input is a list of lists
    if isinstance(matters_list, list) and matters_list and not any(isinstance(i, list) for i in matters_list):
        source_data = matters_list[:]
        matters_list.clear()
        matters_list.append(source_data)
    matters = []
    for current_matter in matters_list:
        # Unpack inputs and optional values
        bstype, label, bs_dir, dos_dir, *optional = current_matter
        # Set defaults using get_or_default
        color = get_or_default(optional[0] if len(optional) > 0 else None, "default")
        lstyle = get_or_default(optional[1] if len(optional) > 1 else None, "solid")
        weight = get_or_default(optional[2] if len(optional) > 2 else None, 1.5)
        alpha = get_or_default(optional[3] if len(optional) > 3 else None, 1.0)
        current_tolerance = get_or_default(optional[4] if len(optional) > 4 else None, 0)
        # Common operations for extracting data
        fermi_energy = extract_fermi(bs_dir)
        kpath, breaks = extract_kpath(bs_dir, return_breaks=True)
        dos = extract_dos(dos_dir)
        # Handle different bandstructure types
        if bstype.lower() in ["monocolor", "monocolor nonpolarized"]:
            bands = extract_eigenvalues_bands_nonpolarized(bs_dir)
            kpath_plot, bands = _apply_breaks_insert_nan(kpath, breaks, bands)
            matters.append([bstype, label, fermi_energy, kpath_plot, bands, dos, color, lstyle, weight, alpha, current_tolerance])
        elif bstype.lower() in ["monocolor spin up", "spin up monocolor"]:
            bands = extract_eigenvalues_bands_spinUp(bs_dir)
            kpath_plot, bands = _apply_breaks_insert_nan(kpath, breaks, bands)
            matters.append([bstype, label, fermi_energy, kpath_plot, bands, dos, color, lstyle, weight, alpha, current_tolerance])
        elif bstype.lower() in ["monocolor spin down", "spin down monocolor"]:
            bands = extract_eigenvalues_bands_spinDown(bs_dir)
            kpath_plot, bands = _apply_breaks_insert_nan(kpath, breaks, bands)
            matters.append([bstype, label, fermi_energy, kpath_plot, bands, dos, color, lstyle, weight, alpha, current_tolerance])
        elif bstype.lower() in ["bands", "bands nonpolarized"]:
            conduction_bands = extract_eigenvalues_conductionBands_nonpolarized(bs_dir, current_tolerance)
            valence_bands = extract_eigenvalues_valenceBands_nonpolarized(bs_dir, current_tolerance)
            kpath_plot, conduction_bands, valence_bands = _apply_breaks_insert_nan(kpath, breaks, conduction_bands, valence_bands)
            matters.append([bstype, label, fermi_energy, kpath_plot, conduction_bands, valence_bands, dos, color, lstyle, weight, alpha, current_tolerance])
        elif bstype.lower() in ["bands spin up", "spin up bands"]:
            conduction_bands = extract_eigenvalues_conductionBands_spinUp(bs_dir, current_tolerance)
            valence_bands = extract_eigenvalues_valenceBands_spinUp(bs_dir, current_tolerance)
            kpath_plot, conduction_bands, valence_bands = _apply_breaks_insert_nan(kpath, breaks, conduction_bands, valence_bands)
            matters.append([bstype, label, fermi_energy, kpath_plot, conduction_bands, valence_bands, dos, color, lstyle, weight, alpha, current_tolerance])
        elif bstype.lower() in ["bands spin down", "spin down bands"]:
            conduction_bands = extract_eigenvalues_conductionBands_spinDown(bs_dir, current_tolerance)
            valence_bands = extract_eigenvalues_valenceBands_spinDown(bs_dir, current_tolerance)
            kpath_plot, conduction_bands, valence_bands = _apply_breaks_insert_nan(kpath, breaks, conduction_bands, valence_bands)
            matters.append([bstype, label, fermi_energy, kpath_plot, conduction_bands, valence_bands, dos, color, lstyle, weight, alpha, current_tolerance])
    return matters

def plot_bsDoS(suptitle, matters_list=None, eigen_range=None, dos_range=None, legend_loc=False):
    # Figure setting
    fig_setting = canvas_setting(12, 6)
    params = fig_setting[2]; plt.rcParams.update(params)

    fig = plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Colors calling
    bs_fermi_color = color_sampling("Violet")
    annotate_color = color_sampling("Grey")

    # Data calling and plotting
    matters = create_matters_bsdos(matters_list)

    # Title
    fig.suptitle(f"{suptitle}", fontsize=fig_setting[3][0], y=1.00)

    # ax1 Band structure
    ax1.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    ax1.set_title("Band structure", fontsize=fig_setting[3][1])

    for matter in matters:
        # print(matter[7], matter[8], matter[9], matter[10])
        bs_current_label = matter[1]
        if _is_monocolor_type(matter[0]):
            bs_label = "mono"
            bs_fermi = matter[2]
            for bands_index in range(0, len(matter[4])):
                current_band = [eigenvalue - bs_fermi for eigenvalue in matter[4][bands_index]]
                if bands_index == 0:
                    ax1.plot(matter[3], current_band, c=color_sampling(matter[6])[1], linestyle=matter[7], lw=matter[8], alpha=matter[9], label=f"{bs_current_label}", zorder=4)
                else:
                    ax1.plot(matter[3], current_band, c=color_sampling(matter[6])[1], linestyle=matter[7], lw=matter[8], alpha=matter[9], zorder=4)
        elif _is_bands_type(matter[0]):
            bs_fermi = matter[2]
            bs_label = "bands"
            for bands_index in range(0, len(matter[4])):
                current_conduction_band = [eigenvalue - bs_fermi for eigenvalue in matter[4][bands_index]]
                if bands_index == 0:
                    ax1.plot(matter[3], current_conduction_band, c=color_sampling(matter[7])[2], linestyle=matter[8], lw=matter[9], alpha=matter[10], label=f"Conduction bands {bs_current_label}", zorder=4)
                else:
                    ax1.plot(matter[3], current_conduction_band, c=color_sampling(matter[7])[2], linestyle=matter[8], lw=matter[9], alpha=matter[10], zorder=4)
            for bands_index in range(0, len(matter[5])):
                current_valence_band = [eigenvalue - bs_fermi for eigenvalue in matter[5][bands_index]]
                if bands_index == 0:
                    ax1.plot(matter[3], current_valence_band, c=color_sampling(matter[7])[0], linestyle=matter[8], lw=matter[9], alpha=matter[10], label=f"Valence bands {bs_current_label}", zorder=4)
                else:
                    ax1.plot(matter[3], current_valence_band, c=color_sampling(matter[7])[0], linestyle=matter[8], lw=matter[9], alpha=matter[10], zorder=4)
        kpath_start = matter[3][0]
        kpath_end = matter[3][-1]
        bs_fermi_last = matter[2]

    # Fermi energy as a horizon line
    ax1.axhline(y = 0, color=bs_fermi_color[0], alpha=0.8, linestyle="--", label="Fermi energy", zorder=2)
    bs_efermi = bs_fermi_last
    kpath_range = kpath_end-kpath_start
    # bs_fermi_energy_text = f"Fermi energy\n{bs_efermi:.3f} (eV)"
    # ax1.text(kpath_start+kpath_range*0.98, eigen_range*0.02, bs_fermi_energy_text, fontsize=10, c=bs_fermi_color[0], rotation=0, va = "bottom", ha="right", zorder=5)

    # y-axis
    ax1.set_ylabel("Energy (eV)")
    demo_boundary = process_boundary(eigen_range)
    if demo_boundary[0] is None:
        ax1.set_ylim(demo_boundary[1]*(-1), demo_boundary[1])
    else: ax1.set_ylim(demo_boundary[0], demo_boundary[1])

    # x-axis
    ax1.set_xlim(kpath_start, kpath_end)

    bs_direction = (matters_list[-1])[2]
    high_symmetry_positions, high_symmetry_labels = kpoints_path_lists(bs_direction)
    ax1.set_xticks(high_symmetry_positions)
    ax1.set_xticklabels(high_symmetry_labels)

    for k_loc in high_symmetry_positions[1:-1]:
        ax1.axvline(x=k_loc, color=annotate_color[1], linestyle="--", alpha=0.8, zorder=1)

    # ax2 DoS
    ax2.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    ax2.set_title("DoS (a.u.)", fontsize=fig_setting[3][1])
    for matter in matters:
        DoS_current_label = matter[1]
        if _is_monocolor_type(matter[0]):
            dos_efermi = matter[5][0]
            plt.plot(matter[5][6], matter[5][5], c=color_sampling(matter[6])[1], lw=matter[8], alpha=matter[9], label=f"Total DoS {DoS_current_label}", zorder = 2)

        elif _is_bands_type(matter[0]):
            dos_efermi = matter[6][0]
            # plt.plot(matter[6][6], matter[6][5], c=color_sampling(matter[7])[1], label=f"Total DoS {current_label}", zorder = 2)
            dos_data = matter[6][6]
            energy_data = matter[6][5]

            conduction_dos = [dos for dos, energy in zip(dos_data, energy_data) if energy > 0]
            conduction_energy = [energy for energy in energy_data if energy > 0]
            valence_dos = [dos for dos, energy in zip(dos_data, energy_data) if energy < 0]
            valence_energy = [energy for energy in energy_data if energy < 0]

            if conduction_dos and conduction_energy:
                ax2.plot(conduction_dos, conduction_energy, c=color_sampling(matter[7])[2], lw=matter[9], alpha=matter[10])
            if valence_dos and valence_energy:
                ax2.plot(valence_dos, valence_energy, c=color_sampling(matter[7])[0], lw=matter[9], alpha=matter[10])

    ax2.set_ylim(eigen_range*(-1), eigen_range)
    ax2.set_xlim(0, dos_range)

    ax2.set_xticks([0, dos_range/2, dos_range])
    ax2.set_xticklabels(["0", f"{dos_range/2:.1f}", f"{dos_range:.1f}"])
    ax2.set_yticks([])

    shift = dos_efermi
    ax2.axhline(y = dos_efermi-shift, color=bs_fermi_color[0], alpha=0.8, linestyle="--", label="Fermi energy", zorder=2)

    # legend
    if legend_loc is True:
        ax1.legend(loc=legend_loc)
        ax2.legend(loc=legend_loc)
    elif legend_loc is None or legend_loc is False:
        pass
    plt.tight_layout()

# plot bandstructure with PDoS
def create_matters_bsPDoS(bs_list, pdos_list):
    """
    Merge BS (Band Structure) and PDoS (Projected Density of States) configurations
    into a structured pair for combined plotting.

    Parameters:
        bs_list (list): BS configuration list. Each item should be in the format:
            [bstype, label, bs_directory, (optional: color, lstyle, weight, alpha, tolerance)]
            For example: ["monocolor", "", "4.1_PDoS/o-B14_K20"]
        pdos_list (list): PDoS configuration list. Each item should be in the format:
            [pdos_label, pdos_directory, atoms, orbital, (optional: line_color, line_style, line_weight, line_alpha)]
            For example: ["$p$ for Group 1", "4.1_PDoS/o-B14_K20", index_g1, "p", "blue", "solid", 2.0, 1.0]

    Returns:
        tuple: (bs_matters, pdos_matters) where:
            - bs_matters is produced by calling create_matters_bs(bs_list)
            - pdos_matters is produced by calling create_matters_pdos(pdos_list)
    """
    # Use your previously defined functions for BS and PDoS
    bs_matters = create_matters_bs(bs_list)
    pdos_matters = create_matters_pdos(pdos_list)
    return bs_matters, pdos_matters

def plot_bsPDoS(title, bs_list, pdos_list, eigen_range, dos_range, legend_loc=False):
    """
    Plot a combined figure with BS (Band Structure) on the left and PDoS on the right.
    
    Parameters:
        title (str): Plot title.
        bs_list (list): BS configuration list (see create_matters_bs for details).
        pdos_list (list): PDoS configuration list (see create_matters_pdos for details).
        eigen_range (float): Energy range for the BS plot (y-axis from -eigen_range to eigen_range).
        dos_range (float): x-axis range for the PDoS plot (from 0 to dos_range).
        legend_loc: Legend location (e.g., "upper right") or False to hide legends.
    """
    # Set up figure using predefined canvas settings
    fig_setting = canvas_setting(12, 6)
    plt.rcParams.update(fig_setting[2])
    fig = plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Color settings for Fermi energy and annotations
    bs_fermi_color = color_sampling("Violet")
    annotate_color = color_sampling("Grey")

    # Set the overall title
    fig.suptitle(title, fontsize=fig_setting[3][0], y=1.00)

    # Get BS and PDoS matters using the helper function
    bs_matters, pdos_matters = create_matters_bsPDoS(bs_list, pdos_list)

    # Plot bandstructure
    ax1.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    ax1.set_title("Band structure", fontsize=fig_setting[3][1])

    for matter in bs_matters:
        bs_current_label = matter[1]
        if _is_monocolor_type(matter[0]):
            bs_label = "mono"
            bs_fermi = matter[2]
            for bands_index in range(0, len(matter[4])):
                current_band = [eigenvalue - bs_fermi for eigenvalue in matter[4][bands_index]]
                if bands_index == 0:
                    ax1.plot(matter[3], current_band, c=color_sampling(matter[5])[1], linestyle=matter[6], lw=matter[7], alpha=matter[8], label=f"{bs_current_label}", zorder=4)
                else:
                    ax1.plot(matter[3], current_band, c=color_sampling(matter[5])[1], linestyle=matter[6], lw=matter[7], alpha=matter[8], zorder=4)
        elif _is_bands_type(matter[0]):
            bs_fermi = matter[2]
            bs_label = "bands"
            for bands_index in range(0, len(matter[4])):
                current_conduction_band = [eigenvalue - bs_fermi for eigenvalue in matter[4][bands_index]]
                if bands_index == 0:
                    ax1.plot(matter[3], current_conduction_band, c=color_sampling(matter[6])[2], linestyle=matter[7], lw=matter[8], alpha=matter[9], label=f"Conduction bands {bs_current_label}", zorder=4)
                else:
                    ax1.plot(matter[3], current_conduction_band, c=color_sampling(matter[6])[2], linestyle=matter[7], lw=matter[8], alpha=matter[9], zorder=4)
            for bands_index in range(0, len(matter[5])):
                current_valence_band = [eigenvalue - bs_fermi for eigenvalue in matter[5][bands_index]]
                if bands_index == 0:
                    ax1.plot(matter[3], current_valence_band, c=color_sampling(matter[6])[0], linestyle=matter[7], lw=matter[8], alpha=matter[9], label=f"Valence bands {bs_current_label}", zorder=4)
                else:
                    ax1.plot(matter[3], current_valence_band, c=color_sampling(matter[6])[0], linestyle=matter[7], lw=matter[8], alpha=matter[9], zorder=4)
        kpath_start = matter[3][0]
        kpath_end = matter[3][-1]
        bs_fermi_last = matter[2]

    # Fermi energy as a horizon line
    ax1.axhline(y = 0, color=bs_fermi_color[0], alpha=0.8, linestyle="--", label="Fermi energy", zorder=2)
    bs_efermi = bs_fermi_last
    kpath_range = kpath_end-kpath_start
    # bs_fermi_energy_text = f"Fermi energy\n{bs_efermi:.3f} (eV)"
    # ax1.text(kpath_start+kpath_range*0.98, eigen_range*0.02, bs_fermi_energy_text, fontsize=10, c=bs_fermi_color[0], rotation=0, va = "bottom", ha="right", zorder=5)

    # y-axis
    ax1.set_ylabel("Energy (eV)")
    demo_boundary = process_boundary(eigen_range)
    if demo_boundary[0] is None:
        ax1.set_ylim(demo_boundary[1]*(-1), demo_boundary[1])
    else: ax1.set_ylim(demo_boundary[0], demo_boundary[1])

    # x-axis
    ax1.set_xlim(kpath_start, kpath_end)
    bs_direction = (bs_list[-1])[2]
    high_symmetry_positions, high_symmetry_labels = kpoints_path_lists(bs_direction)
    ax1.set_xticks(high_symmetry_positions)
    ax1.set_xticklabels(high_symmetry_labels)

    for k_loc in high_symmetry_positions[1:-1]:
        ax1.axvline(x=k_loc, color=annotate_color[1], linestyle="--", alpha=0.8, zorder=1)

    # Plot PDoS
    ax2.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    ax2.set_title("PDoS (a.u.)", fontsize=fig_setting[3][1])

    dos_efermi = None
    for pdos_matter in pdos_matters:
        pdos_label = pdos_matter[0]
        pdos_data = pdos_matter[1]
        orbital = pdos_matter[3]
        ax2.plot(pdos_data[orbital], pdos_data["pdos_shifted_energy"],
                 color=color_sampling(pdos_matter[4])[0],
                 linestyle=pdos_matter[5],
                 linewidth=pdos_matter[6],
                 alpha=pdos_matter[7],
                 label=pdos_label, zorder=2)
        dos_efermi = pdos_data["efermi"]
    ax2.set_ylim(-eigen_range, eigen_range)
    ax2.set_xlim(0, dos_range)
    ax2.set_xticks([0, dos_range/2, dos_range])
    ax2.set_xticklabels(["0", f"{dos_range/2:.1f}", f"{dos_range:.1f}"])
    ax2.set_yticks([])
    if dos_efermi is not None:
        ax2.axhline(y=0, linestyle="--", color=bs_fermi_color[0], alpha=0.8,
                    label="Fermi energy", zorder=2)
    
    # Legend settings
    if legend_loc:
        ax1.legend(loc=legend_loc)
        ax2.legend(loc=legend_loc)
    
    plt.tight_layout()
    plt.show()

# plot spin-polarized bandstructure: FM / AFM
def _bs_parse_pair(value, default_pair, duplicate_scalar=True):
    """Return a two-item tuple from tuple/list or strings such as "(blue,purple)" and "(solid/dotted)"."""
    if value is None:
        return tuple(default_pair)

    if isinstance(value, (tuple, list)):
        if len(value) == 0:
            return tuple(default_pair)
        if len(value) == 1:
            return (value[0], value[0]) if duplicate_scalar else tuple(default_pair)
        return (value[0], value[1])

    value_str = str(value).strip()
    if value_str == "":
        return tuple(default_pair)

    if value_str.lower() == "default":
        return tuple(default_pair)

    if value_str.startswith("(") and value_str.endswith(")"):
        value_str = value_str[1:-1].strip()

    if "," in value_str:
        parts = [part.strip() for part in value_str.split(",") if part.strip()]
    elif "/" in value_str:
        parts = [part.strip() for part in value_str.split("/") if part.strip()]
    else:
        parts = [value_str]

    if len(parts) >= 2:
        return (parts[0], parts[1])
    if len(parts) == 1 and duplicate_scalar:
        return (parts[0], parts[0])
    return tuple(default_pair)

def _bs_resolve_color(color, shade_index=1):
    """Resolve a vmatplot color name through color_sampling; fall back to raw Matplotlib color."""
    try:
        sampled = color_sampling(color)
        if isinstance(sampled, (list, tuple)) and len(sampled) > shade_index:
            return sampled[shade_index]
    except Exception:
        pass
    return color

def _bs_normalize_matters_list(matters_list):
    """Accept either a single matter list or a list of matter lists."""
    if matters_list is None:
        return []
    if isinstance(matters_list, list) and matters_list and not any(isinstance(i, list) for i in matters_list):
        return [matters_list]
    return matters_list

def create_matters_bs_spin(matters_list):
    """
    Create spin-polarized bandstructure matters.

    Accepted matter format:
        [bstype, label, directory, color_pair, linestyle_pair, weight, alpha, tolerance]

    Examples:
        ["monocolor", "GGA-PBE", "3.1_bandstructure/monolayer", ("blue", "purple")]
        ["monocolor", "GGA-PBE", "3.1_bandstructure/monolayer", "(blue,purple)", "(solid/dotted)"]
    """
    matters = []
    for current_matter in _bs_normalize_matters_list(matters_list):
        bstype, label, directory, *optional = current_matter

        color_pair = _bs_parse_pair(
            optional[0] if len(optional) > 0 else None,
            ("blue", "red"),
            duplicate_scalar=True
        )
        lstyle_pair = _bs_parse_pair(
            optional[1] if len(optional) > 1 else None,
            ("solid", "dotted"),
            duplicate_scalar=True
        )
        weight = get_or_default(optional[2] if len(optional) > 2 else None, 1.5)
        alpha = get_or_default(optional[3] if len(optional) > 3 else None, 1.0)
        current_tolerance = get_or_default(optional[4] if len(optional) > 4 else None, 0)

        fermi_energy = extract_fermi(directory)
        kpath, breaks = extract_kpath(directory, return_breaks=True)

        bstype_lower = bstype.lower()
        if bstype_lower in ["monocolor", "monocolor spin", "spin monocolor", "monocolor fm", "monocolor afm"]:
            bands_up = extract_eigenvalues_bands_spinUp(directory)
            bands_down = extract_eigenvalues_bands_spinDown(directory)
            kpath, bands_up, bands_down = _apply_breaks_insert_nan(kpath, breaks, bands_up, bands_down)
            matters.append([
                bstype, label, fermi_energy, kpath,
                bands_up, bands_down,
                color_pair, lstyle_pair, weight, alpha, current_tolerance, directory
            ])

        elif bstype_lower in ["bands", "bands spin", "spin bands", "bands fm", "bands afm"]:
            conduction_up = extract_eigenvalues_conductionBands_spinUp(directory, current_tolerance)
            valence_up = extract_eigenvalues_valenceBands_spinUp(directory, current_tolerance)
            conduction_down = extract_eigenvalues_conductionBands_spinDown(directory, current_tolerance)
            valence_down = extract_eigenvalues_valenceBands_spinDown(directory, current_tolerance)
            kpath, conduction_up, valence_up, conduction_down, valence_down = _apply_breaks_insert_nan(
                kpath, breaks, conduction_up, valence_up, conduction_down, valence_down
            )
            matters.append([
                bstype, label, fermi_energy, kpath,
                conduction_up, valence_up, conduction_down, valence_down,
                color_pair, lstyle_pair, weight, alpha, current_tolerance, directory
            ])

        else:
            raise ValueError(
                f"Unsupported spin bandstructure type: {bstype}. "
                "Use 'monocolor' or 'bands'."
            )

    return matters

def _plot_bandstructure_spin(title, matters_list=None, eigen_range=None, legend_loc=False, magnetic_order="FM"):
    help_info = """
    Usage: plot_bandstructure_FM / plot_bandstructure_AFM
        arg[0]: title
        arg[1]: matters list
        arg[2]: eigenvalue range, from -arg[2] to arg[2], or [emin, emax]
        arg[3]: legend location

    matter format:
        [bstype, label, directory, color_pair, linestyle_pair, weight, alpha, tolerance]

    examples:
        ["monocolor", "bands of GGA-PBE", "3.1_bandstructure/monolayer", ("blue", "purple")]
        ["monocolor", "bands of GGA-PBE", "3.1_bandstructure/monolayer", "(blue,purple)", "(solid/dotted)"]
    """
    if title in ["help", "Help"]:
        print(help_info)
        return

    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    fermi_color = color_sampling("Violet")
    annotate_color = color_sampling("Grey")

    matters = create_matters_bs_spin(matters_list)

    kpath_start = None
    kpath_end = None
    last_directory = None

    for matter in matters:
        bstype_lower = matter[0].lower()
        current_label = matter[1]
        fermi = matter[2]
        kpath = matter[3]
        last_directory = matter[-1]

        if bstype_lower in ["monocolor", "monocolor spin", "spin monocolor", "monocolor fm", "monocolor afm"]:
            bands_up = matter[4]
            bands_down = matter[5]
            color_pair = matter[6]
            lstyle_pair = matter[7]
            weight = matter[8]
            alpha = matter[9]

            spin_channels = [
                (bands_up, "spin up", _bs_resolve_color(color_pair[0]), lstyle_pair[0]),
                (bands_down, "spin down", _bs_resolve_color(color_pair[1]), lstyle_pair[1]),
            ]

            for bands, spin_label, color, lstyle in spin_channels:
                for band_index, band in enumerate(bands):
                    current_band = [eigenvalue - fermi for eigenvalue in band]
                    plot_label = f"{current_label} {spin_label}" if band_index == 0 else None
                    plt.plot(kpath, current_band, c=color, linestyle=lstyle, lw=weight,
                             alpha=alpha, label=plot_label, zorder=4)

        elif bstype_lower in ["bands", "bands spin", "spin bands", "bands fm", "bands afm"]:
            conduction_up = matter[4]
            valence_up = matter[5]
            conduction_down = matter[6]
            valence_down = matter[7]
            color_pair = matter[8]
            lstyle_pair = matter[9]
            weight = matter[10]
            alpha = matter[11]

            spin_channels = [
                (conduction_up + valence_up, "spin up", _bs_resolve_color(color_pair[0]), lstyle_pair[0]),
                (conduction_down + valence_down, "spin down", _bs_resolve_color(color_pair[1]), lstyle_pair[1]),
            ]

            for bands, spin_label, color, lstyle in spin_channels:
                for band_index, band in enumerate(bands):
                    current_band = [eigenvalue - fermi for eigenvalue in band]
                    plot_label = f"{current_label} {spin_label}" if band_index == 0 else None
                    plt.plot(kpath, current_band, c=color, linestyle=lstyle, lw=weight,
                             alpha=alpha, label=plot_label, zorder=4)

        kpath_start = kpath[0]
        kpath_end = kpath[-1]

    if kpath_start is None or kpath_end is None:
        raise ValueError("No bandstructure matter was provided.")

    plt.axhline(y=0, color=fermi_color[0], alpha=0.8, linestyle="--",
                label="Fermi energy", zorder=2)

    plt.title(f"{title}")
    plt.ylabel("Energy (eV)")

    if eigen_range is not None:
        demo_boundary = process_boundary(eigen_range)
        if demo_boundary[0] is None:
            plt.ylim(demo_boundary[1] * (-1), demo_boundary[1])
        else:
            plt.ylim(demo_boundary[0], demo_boundary[1])

    plt.xlim(kpath_start, kpath_end)

    high_symmetry_positions, high_symmetry_labels = kpoints_path_lists(last_directory)
    plt.xticks(high_symmetry_positions, high_symmetry_labels)

    for k_loc in high_symmetry_positions[1:-1]:
        plt.axvline(x=k_loc, color=annotate_color[1], linestyle="--", alpha=0.8, zorder=1)

    if legend_loc is True:
        plt.legend()
    elif legend_loc is not None and legend_loc is not False:
        plt.legend(loc=legend_loc)

    plt.tight_layout()

def plot_bandstructure_FM(title, matters_list=None, eigen_range=None, legend_loc=False):
    """Plot collinear spin-polarized bandstructure for FM calculations."""
    return _plot_bandstructure_spin(title, matters_list, eigen_range, legend_loc, magnetic_order="FM")

def plot_bandstructure_AFM(title, matters_list=None, eigen_range=None, legend_loc=False):
    """Plot collinear spin-polarized bandstructure for AFM calculations."""

    return _plot_bandstructure_spin(title, matters_list, eigen_range, legend_loc, magnetic_order="AFM")

def create_matters_bs_spin(matters_list):
    # Ensure input is a list of lists
    if isinstance(matters_list, list) and matters_list and not any(isinstance(i, list) for i in matters_list):
        source_data = matters_list[:]
        matters_list.clear()
        matters_list.append(source_data)

    matters = []
    for current_matter in matters_list:
        bstype, label, directory, *optional = current_matter

        # Set default values using get_or_default
        color = get_or_default(optional[0] if len(optional) > 0 else None, "default")
        lstyle = get_or_default(optional[1] if len(optional) > 1 else None, "solid")
        weight = get_or_default(optional[2] if len(optional) > 2 else None, 1.5)
        alpha = get_or_default(optional[3] if len(optional) > 3 else None, 1.0)
        current_tolerance = get_or_default(optional[4] if len(optional) > 4 else None, 0)

        # Spin channel calling
        label_lower = str(label).lower()
        if label_lower in ["spin up", "spin-up", "up", "spin 1", "spin1"]:
            spin_label = "spin up"
        elif label_lower in ["spin down", "spin-down", "down", "spin 2", "spin2"]:
            spin_label = "spin down"
        else:
            spin_label = "nonpolarized"

        # Band structure plotting style: monocolor
        if bstype.lower() in ["monocolor", "monocolor nonpolarized"]:
            fermi_energy = extract_fermi(directory)
            kpath, breaks = extract_kpath(directory, return_breaks=True)

            if spin_label == "spin up":
                bands = extract_eigenvalues_bands_spinUp(directory)
            elif spin_label == "spin down":
                bands = extract_eigenvalues_bands_spinDown(directory)
            else:
                bands = extract_eigenvalues_bands_nonpolarized(directory)

            kpath, bands = _apply_breaks_insert_nan(kpath, breaks, bands)
            matters.append([bstype, label, fermi_energy, kpath, bands, color, lstyle, weight, alpha, current_tolerance, directory])

        # Band structure plotting style: bands
        elif bstype.lower() in ["bands", "bands nonpolarized"]:
            fermi_energy = extract_fermi(directory)
            kpath, breaks = extract_kpath(directory, return_breaks=True)

            if spin_label == "spin up":
                conduction_bands = extract_eigenvalues_conductionBands_spinUp(directory, current_tolerance)
                valence_bands = extract_eigenvalues_valenceBands_spinUp(directory, current_tolerance)
            elif spin_label == "spin down":
                conduction_bands = extract_eigenvalues_conductionBands_spinDown(directory, current_tolerance)
                valence_bands = extract_eigenvalues_valenceBands_spinDown(directory, current_tolerance)
            else:
                conduction_bands = extract_eigenvalues_conductionBands_nonpolarized(directory, current_tolerance)
                valence_bands = extract_eigenvalues_valenceBands_nonpolarized(directory, current_tolerance)

            kpath, conduction_bands, valence_bands = _apply_breaks_insert_nan(kpath, breaks, conduction_bands, valence_bands)
            matters.append([bstype, label, fermi_energy, kpath, conduction_bands, valence_bands, color, lstyle, weight, alpha, current_tolerance, directory])

    return matters

def _mirror_legend_location(legend_loc):
    if legend_loc is True:
        legend_loc = "upper right"

    if legend_loc in [None, False]:
        return "upper right"

    legend_loc = str(legend_loc).lower()

    if "right" in legend_loc:
        return legend_loc.replace("right", "left")
    elif "left" in legend_loc:
        return legend_loc.replace("left", "right")
    else:
        return "upper left"

def _loc_to_axes_position(loc):
    loc = str(loc).lower()
    if loc == "upper left":
        return 0.02, 0.97, "left", "top"
    elif loc == "upper right":
        return 0.98, 0.97, "right", "top"
    elif loc == "lower left":
        return 0.02, 0.03, "left", "bottom"
    elif loc == "lower right":
        return 0.98, 0.03, "right", "bottom"
    elif loc == "center left":
        return 0.02, 0.50, "left", "center"
    elif loc == "center right":
        return 0.98, 0.50, "right", "center"
    else:
        return 0.02, 0.97, "left", "top"

def plot_bandstructure_spin(title, matters_list=None, state_label=None, eigen_range=None, legend_loc=False):
    # Help information
    help_info = """
    Usage: plot_bandstructure_spin
        arg[0]: title;
        arg[1]: matters list;
        arg[2]: state label, such as "FM", "AFM", or any label;
        arg[3]: the range of eigenvalues, from -arg[3] to arg[3];
        arg[4]: legend location;
    """
    if title in ["help", "Help"]:
        print(help_info)
        return

    # Figure settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Colors calling
    fermi_color = color_sampling("Violet")
    annotate_color = color_sampling("Grey")

    # Data calling and plotting
    matters = create_matters_bs_spin(matters_list)

    for matter in matters:
        current_label = matter[1]

        if matter[0].lower() in ["monocolor", "monocolor nonpolarized"]:
            fermi = matter[2]
            for bands_index in range(0, len(matter[4])):
                current_band = [eigenvalue - fermi for eigenvalue in matter[4][bands_index]]
                if bands_index == 0:
                    plt.plot(matter[3], current_band, c=color_sampling(matter[5])[1], linestyle=matter[6], lw=matter[7], alpha=matter[8], label=f"{current_label}", zorder=4)
                else:
                    plt.plot(matter[3], current_band, c=color_sampling(matter[5])[1], linestyle=matter[6], lw=matter[7], alpha=matter[8], zorder=4)

        elif matter[0].lower() in ["bands", "bands nonpolarized"]:
            fermi = matter[2]
            for bands_index in range(0, len(matter[4])):
                current_conduction_band = [eigenvalue - fermi for eigenvalue in matter[4][bands_index]]
                if bands_index == 0:
                    plt.plot(matter[3], current_conduction_band, c=color_sampling(matter[6])[2], linestyle=matter[7], lw=matter[8], alpha=matter[9], label=f"Conduction bands for {current_label}", zorder=4)
                else:
                    plt.plot(matter[3], current_conduction_band, c=color_sampling(matter[6])[2], linestyle=matter[7], lw=matter[8], alpha=matter[9], zorder=4)

            for bands_index in range(0, len(matter[5])):
                current_valence_band = [eigenvalue - fermi for eigenvalue in matter[5][bands_index]]
                if bands_index == 0:
                    plt.plot(matter[3], current_valence_band, c=color_sampling(matter[6])[0], linestyle=matter[7], lw=matter[8], alpha=matter[9], label=f"Valence bands for {current_label}", zorder=4)
                else:
                    plt.plot(matter[3], current_valence_band, c=color_sampling(matter[6])[0], linestyle=matter[7], lw=matter[8], alpha=matter[9], zorder=4)

        kpath_start = matter[3][0]
        kpath_end = matter[3][-1]
        fermi_last = matter[2]
        reference_directory = matter[-1]

    # Fermi energy as a horizon line
    plt.axhline(y=0, color=fermi_color[0], alpha=0.8, linestyle="--", label="Fermi energy", zorder=2)

    # Figure title and labels
    plt.title(f"{title}")
    plt.ylabel("Energy (eV)")

    # Eigenvalue range
    demo_boundary = process_boundary(eigen_range)
    if demo_boundary[0] is None:
        plt.ylim(demo_boundary[1]*(-1), demo_boundary[1])
    else:
        plt.ylim(demo_boundary[0], demo_boundary[1])

    # K-path range
    plt.xlim(kpath_start, kpath_end)

    # State label, mirrored against the legend location
    if state_label not in [None, False]:
        if legend_loc is True:
            current_legend_loc = fig_setting[4]
        elif legend_loc in [None, False]:
            current_legend_loc = False
        else:
            current_legend_loc = legend_loc

        state_loc = _mirror_legend_location(current_legend_loc)
        state_x, state_y, state_ha, state_va = _loc_to_axes_position(state_loc)

        # Fine tune the state label position in axes-relative coordinates
        state_x = state_x
        state_y = state_y

        plt.text(state_x, state_y, f"{state_label}", transform=plt.gca().transAxes,
                 fontsize=16, ha=state_ha, va=state_va,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=annotate_color[1], alpha=0.75),
                 zorder=6)

    # High symmetry path
    high_symmetry_positions, high_symmetry_labels = kpoints_path_lists(reference_directory)
    plt.xticks(high_symmetry_positions, high_symmetry_labels)

    for k_loc in high_symmetry_positions[1:-1]:
        plt.axvline(x=k_loc, color=annotate_color[1], linestyle="--", alpha=0.8, zorder=1)

    # Legend
    if legend_loc is True:
        plt.legend(loc=fig_setting[4])
    elif legend_loc is None or legend_loc is False:
        # Do not display the legend
        pass
    else:
        plt.legend(loc=legend_loc)

    plt.tight_layout()
