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

def extract_phonon_high_sym(directory, return_coords=False):
    """Parse a VASP QPOINTS/QPOINTS_OPT line-mode file and return high-symmetry path markers.
    Parameters
    directory : str
        Directory containing QPOINTS or QPOINTS_OPT.
    return_coords : bool
        If False (default), return a list of labels.
        If True, return a list of (label, [qx, qy, qz]) boundary points in plotting order.
        This keeps repeated labels when they represent distinct boundaries/branches.
    """
    qpoints_file_path = os.path.join(directory, "QPOINTS")
    qpoints_opt_path = os.path.join(directory, "QPOINTS_OPT")
    if os.path.exists(qpoints_opt_path):
        qpoints_file = qpoints_opt_path
    elif os.path.exists(qpoints_file_path):
        qpoints_file = qpoints_file_path
    else:
        raise FileNotFoundError("QPOINTS file not found in the directory.")

    with open(qpoints_file, "r", encoding="utf-8") as file:
        qlines = file.readlines()

    # Check line-mode (accept both 'line' and 'line-mode')
    if len(qlines) < 4 or qlines[2].strip()[:1].lower() != "l":
        raise ValueError(f"Expected a line-mode QPOINTS file (3rd line starts with 'L'), got: {qlines[2] if len(qlines) > 2 else '<missing>'}")

    # Collect all labeled endpoints (each segment is defined by two consecutive labeled lines).
    endpoints = []
    for line in qlines[4:]:
        tokens = line.strip().split()
        if len(tokens) < 4:
            continue
        label = tokens[-1]
        # Many people use Γ etc; isalpha() covers unicode letters too.
        if not label.isalpha():
            continue
        try:
            coords = [float(tokens[0]), float(tokens[1]), float(tokens[2])]
        except ValueError:
            continue
        endpoints.append((label, coords))

    if not endpoints:
        return [] if not return_coords else []

    # Pair endpoints into segments: (start, end), (start, end), ...
    segments = []
    for i in range(0, len(endpoints) - 1, 2):
        segments.append((endpoints[i], endpoints[i + 1]))

    # Build a boundary list that:
    #   - keeps repeated labels when they occur at different boundaries
    #   - inserts the start point again if a segment starts from a different point than the previous segment ended
    boundaries = [segments[0][0]]
    for si, (start, end) in enumerate(segments):
        boundaries.append(end)
        if si + 1 < len(segments):
            next_start = segments[si + 1][0]
            if np.linalg.norm(np.array(next_start[1]) - np.array(end[1])) > 1e-10:
                boundaries.append(next_start)

    if return_coords:
        return boundaries
    return [lbl for (lbl, _c) in boundaries]

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
    """
    Extract the phonon frequencies at q-points.
    In this case, we simply call extract_phonon_bands since the phonon frequencies are extracted from OUTCAR.
    """
    return extract_phonon_bands(directory)

def extract_phonon_bands(directory):
    """
    Extract phonon dispersion data from the OUTCAR file:
      - Parse the vibrational frequencies (in THz) for each q-point for each phonon branch.
      - Also compute the cumulative path (unweighted) based on the q-point coordinates.
    Returns:
      A dictionary containing:
        - "path": a list of cumulative distances corresponding to the q-points.
        - "bands": a 2D list where each sublist corresponds to the frequency sequence along the q-path for one phonon branch.
    """
    outcar_file = os.path.join(directory, "OUTCAR")
    # First pass: determine the number of phonon branches (num_bands)
    num_bands = 0
    with open(outcar_file, "r") as f: lines = f.readlines()
    for index, line in enumerate(lines):
        if "branch index" in line:
            order = index + 1
            while order < len(lines) and lines[order].strip():
                num_bands += 1
                order += 1
            break
    # Initialize storage for bands and path
    bands = [[] for _ in range(num_bands)]
    path = []
    prev_coords = None
    total_distance = 0.0
    index = 0
    while index < len(lines):
        line = lines[index]
        if "q-point No." in line:
            index += 1
            if index >= len(lines): break
            coord_line = lines[index].strip()
            parts = coord_line.split()
            try: coords = [float(parts[1]), float(parts[2]), float(parts[3])] # Assume the format: "q-point: x y z"
            except Exception:
                index += 1
                continue
            if prev_coords is None: total_distance = 0.0
            else:
                d = np.linalg.norm(np.array(coords) - np.array(prev_coords))
                total_distance += d
            path.append(total_distance)
            prev_coords = coords
            index += 1
        elif "branch index" in line:
            index += 1
            # For the current q-point, read the next num_bands lines for the phonon frequencies
            for band in range(num_bands):
                if index >= len(lines): break
                freq_line = lines[index].strip()
                if not freq_line:
                    index += 1
                    continue
                parts = freq_line.split()
                try:  freq = float(parts[1])
                except Exception: freq = None
                if freq is not None: bands[band].append(freq)
                index += 1
        else: index += 1
    return {"path": path, "bands": bands}

def create_matters_phonons(matters_list):
    # If matters_list is a single list (not nested), convert it to a nested list.
    if isinstance(matters_list, list) and matters_list and not any(isinstance(i, list) for i in matters_list):
        source_data = matters_list[:]
        matters_list.clear()
        matters_list.append(source_data)
    matters = []
    for current_matter in matters_list:
        # current_matter format: [ label, directory, *optional ]
        label, directory, *optional = current_matter
        color = get_or_default(optional[0] if len(optional) > 0 else None, "default")
        lstyle = get_or_default(optional[1] if len(optional) > 1 else None, "solid")
        weight = get_or_default(optional[2] if len(optional) > 2 else None, 1.5)
        alpha = get_or_default(optional[3] if len(optional) > 3 else None, 1.0)
        tolerance = get_or_default(optional[4] if len(optional) > 4 else None, 0)
        # qpath: computed using extract_qpath(directory)
        qpath = extract_qpath(directory)
        bands_data = extract_phonon_bands(directory)
        bands = bands_data["bands"]
        matters.append([label, 0, qpath, bands, color, lstyle, weight, alpha, tolerance])
    return matters

def plot_phonons(title, matters_list=None, eigen_range=None, legend_loc=False):
    """
    Plot the phonon dispersion curves in a style consistent with the bandstructure program.
    Parameters:
      - title: The title of the plot.
      - eigen_range: The frequency range (in THz) for the y-axis, e.g., 50 indicates [-50, 50].
      - matters_list: A list of phonon matters, each in the format [ label, directory, *optional ].
      - legend_loc: The location for the legend (e.g., "best"); if False, no legend is shown.
    """
    help_info = """
    Usage: plot_phonons
        arg[0]: title;
        arg[1]: the range of eigenvalues (frequency in THz), from -arg[1] to arg[1];
        arg[2]: matters list;
        arg[3]: legend location;
    """
    if title in ["help", "Help"]:
        print(help_info)
        return

    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    plt.rcParams.update(fig_setting[2])
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    fermi_color = color_sampling("Violet")  # Used for plotting the y=0 reference line
    annotate_color = color_sampling("Grey")

    # Create matters using the updated create_matters_phonons
    matters = create_matters_phonons(matters_list)
    # Use the weighted q-path from one of the matters (they should be identical)
    weighted_qpath = matters[0][2]

    for matter in matters:
        current_label = matter[0]
        qpath = matter[2]; bands = matter[3]; color = matter[4]; lstyle = matter[5]; weight = matter[6]; alpha = matter[7]
        # tolerance = matter[8]  # Currently not used in plotting
        for band_index, band in enumerate(bands):
            if band_index == 0:
                plt.plot(qpath, band, c=color_sampling(color)[1], linestyle=lstyle, lw=weight, alpha=alpha, label=f"{current_label}", zorder=4)
            else:
                plt.plot(qpath, band, c=color_sampling(color)[1], linestyle=lstyle, lw=weight, alpha=alpha, zorder=4)
    plt.axhline(y=0, color=fermi_color[0], alpha=0.8, linestyle="--", zorder=2)
    # plt.axhline(y=0, color=fermi_color[0], alpha=0.8, linestyle="--", label="0 (THz)", zorder=2)
    plt.title(title)
    plt.ylabel("Frequency (THz)")
    demo_boundary = process_boundary(eigen_range)
    if demo_boundary[0] is None:
        plt.ylim(-demo_boundary[1], demo_boundary[1])
    else:
        plt.ylim(demo_boundary[0], demo_boundary[1])
    if weighted_qpath:
        plt.xlim(weighted_qpath[0], weighted_qpath[-1])

    # Get the original directory from the original matters_list (format: [label, directory, *optional])
    orig_directory = matters_list[-1][1]

    # Extract high symmetry boundaries from QPOINTS (keep repeats for multi-segment / branched paths)
    # and set xticks using the weighted q-path.
    boundaries = extract_phonon_high_sym(orig_directory, return_coords=True)
    details = extract_phonon_high_sym_details(orig_directory)
    details_q = details["q_coords"]
    weighted_qpath_full = extract_qpath(orig_directory)

    high_sym_positions = []
    last_search_start = 0  # enforce monotonic matching along the OUTCAR q-point sequence

    for label, hs_coords in boundaries:
        if not details_q:
            break

        min_dist = float("inf")
        best_idx = None
        # Match forward only; otherwise repeated points (e.g., Γ ... Γ) collapse onto the first occurrence.
        for i, q in enumerate(details_q[last_search_start:], start=last_search_start):
            dist = np.linalg.norm(np.array(q) - np.array(hs_coords))
            if dist < min_dist:
                min_dist = dist
                best_idx = i
                if min_dist < 1e-12:
                    break

        if best_idx is None or best_idx >= len(weighted_qpath_full):
            continue

        pos = weighted_qpath_full[best_idx]

        # If two labeled endpoints map to the same x-position (segment boundary duplication),
        # merge the labels as "A|B" at a single tick.
        if high_sym_positions and abs(pos - high_sym_positions[-1][1]) < 1e-10:
            prev_label, prev_pos = high_sym_positions[-1]
            if prev_label != label:
                high_sym_positions[-1] = (f"{prev_label}|{label}", prev_pos)
        else:
            high_sym_positions.append((label, pos))

        last_search_start = best_idx + 1
    if high_sym_positions:
        ticks = [pos for label, pos in high_sym_positions]
        tick_labels = [label for label, pos in high_sym_positions]
        plt.xticks(ticks, tick_labels)
        for pos in ticks[1:-1]:
            plt.axvline(x=pos, color=annotate_color[1], linestyle="--", alpha=0.8, zorder=1)
    if legend_loc is None or legend_loc is False: pass
    else: plt.legend(loc=legend_loc)
    
    plt.tight_layout()
