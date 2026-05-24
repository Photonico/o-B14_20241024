#### Phonon dispersion
# pylint: disable = C0103, C0114, C0116, C0301, C0302, C0321, R0913, R0914, R0915, W0612, W0105

# Necessary packages invoking
import os
import re
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from vmatplot.output_settings import color_sampling, canvas_setting
from vmatplot.algorithms import transpose_matrix
from vmatplot.commons import extract_fermi, get_atoms_count, process_boundary, get_or_default

import matplotlib as mpl

mpl.rcParams["lines.solid_capstyle"] = "round"
mpl.rcParams["lines.dash_capstyle"]  = "round"
mpl.rcParams["lines.solid_joinstyle"] = "round"
mpl.rcParams["lines.dash_joinstyle"]  = "round"

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


def _clean_phonopy_label(label: str) -> str:
    """Best-effort cleanup of phonopy label strings (often LaTeX) into plain text."""
    if label is None:
        return ""
    s = str(label).strip()
    # Strip $...$ math wrappers
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()

    # Common LaTeX wrappers
    # e.g. \mathrm{K} -> K
    m = re.match(r"\\mathrm\{([^}]*)\}$", s)
    if m:
        s = m.group(1).strip()

    # Remove leading backslashes
    s = s.replace("\\", "")
    s = s.replace("\\", "")  # defensive

    # Map a few common Greek names to Unicode
    greek = {
        "Gamma": "Γ",
        "Delta": "Δ",
        "Sigma": "Σ",
        "Lambda": "Λ",
        "Xi": "Ξ",
        "Pi": "Π",
        "Omega": "Ω",
    }
    return greek.get(s, s)


def extract_phonopy_high_sym_from_band_yaml(directory: str):
    """Return (ticks, labels) for phonopy plots from band.yaml when possible.

    This mirrors the spirit of extract_phonon_high_sym (VASP/QPOINTS):
    - ticks: x positions of high-symmetry points along the path
    - labels: corresponding labels (e.g. Γ Z T Y Γ)

    Preference order:
      1) band.yaml: keys 'labels' + 'segment_nqpoint' / distance repeats
      2) band.conf / phonopy.conf: BAND_LABELS + segmentation inferred from qpath NaNs (fallback)
    """
    directory = str(directory)

    # --- locate band.yaml ---
    band_yaml = None
    for cand in ("band.yaml", "phonopy_band.yaml"):
        p = os.path.join(directory, cand)
        if os.path.exists(p):
            band_yaml = p
            break
    if band_yaml is None:
        return None, None

    # --- read YAML ---
    try:
        import yaml  # type: ignore
        with open(band_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return None, None

    phonon = (data or {}).get("phonon", [])
    if not phonon:
        return None, None

    dist = [float(p.get("distance", 0.0)) for p in phonon]
    n = len(dist)

    # --- ticks from segment_nqpoint if consistent; otherwise from distance repeats ---
    ticks = [dist[0]]
    seg = (data or {}).get("segment_nqpoint", None)
    if isinstance(seg, list) and seg and all(isinstance(x, (int, float, str)) for x in seg):
        try:
            seg_int = [int(x) for x in seg]
        except Exception:
            seg_int = []
        if seg_int and sum(seg_int) == n:
            idx = 0
            for s in seg_int:
                idx += s
                ticks.append(dist[idx - 1])
    if len(ticks) == 1:
        # Fallback: boundary points often appear twice -> distance[i] == distance[i-1]
        for i in range(1, n):
            if abs(dist[i] - dist[i - 1]) < 1e-12:
                ticks.append(dist[i])
        if abs(ticks[-1] - dist[-1]) > 1e-12:
            ticks.append(dist[-1])

    # --- labels ---
    # Prefer BAND_LABELS from band.conf (user-specified). If not available, fall back to band.yaml.
    labels = []
    try:
        conf = extract_phonopy_band_conf(directory)
        labels = conf.get("boundary_labels", []) if conf else []
    except Exception:
        labels = []

    # Normalize labels (handles LaTeX like $\Gamma$ or \\Gamma)
    if labels:
        labels = [_clean_phonopy_label(x) for x in labels]

    if not labels:
        pairs = (data or {}).get("labels", None)
        if isinstance(pairs, list) and pairs and all(isinstance(x, (list, tuple)) and len(x) >= 2 for x in pairs):
            labels = [_clean_phonopy_label(pairs[0][0])]
            for a, b, *rest in pairs:
                labels.append(_clean_phonopy_label(b))

    # Force labels length == ticks length (never trim ticks based on labels)
    if len(labels) < len(ticks):
        labels = labels + [""] * (len(ticks) - len(labels))
    elif len(labels) > len(ticks):
        labels = labels[: len(ticks)]

    return ticks, labels

def detect_phonon_backend(directory):
    """Detect which backend produced the phonon dispersion data stored under *directory*.

    Returns
    -------
    str
        "phonopy" if a phonopy-style parent directory is detected (band.yaml / phonopy_disp.yaml / band.conf),
        "vasp" if a direct VASP phonon OUTCAR+QPOINTS directory is detected,
        "unknown" otherwise.
    """
    if directory is None:
        return "unknown"
    # Phonopy parent directory
    if os.path.isfile(os.path.join(directory, "band.yaml")):
        return "phonopy"
    if os.path.isfile(os.path.join(directory, "phonopy_disp.yaml")) or os.path.isfile(os.path.join(directory, "phonopy.yaml")):
        return "phonopy"
    if os.path.isfile(os.path.join(directory, "band.conf")) or os.path.isfile(os.path.join(directory, "phonopy.conf")):
        # band.conf without band.yaml usually means "not yet generated", but it is still phonopy-style.
        return "phonopy"

    # Direct VASP phonon run directory
    outcar = os.path.join(directory, "OUTCAR")
    if os.path.isfile(outcar) and (os.path.isfile(os.path.join(directory, "QPOINTS")) or os.path.isfile(os.path.join(directory, "QPOINTS_OPT"))):
        return "vasp"

    return "unknown"


def create_matters_phonons(matters_list):
    """Create internal matter objects for phonon plotting.

    This function now supports BOTH:
      - Direct VASP phonon runs (OUTCAR + QPOINTS/QPOINTS_OPT in the same directory).
      - Phonopy workflows (parent directory contains band.yaml and band.conf / phonopy_disp.yaml, plus pd-*/ subdirs).

    The external matters_list format stays the same:
        [ label, directory, color?, lstyle?, weight?, alpha?, tolerance? ]
    """
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

        backend = detect_phonon_backend(directory)

        if backend == "phonopy":
            bands_data = extract_phonopy_bands(directory)
            qpath = bands_data["path"]
            bands = bands_data["bands"]
        elif backend == "vasp":
            qpath = extract_qpath(directory)
            bands_data = extract_phonon_bands(directory)
            bands = bands_data["bands"]
        else:
            raise FileNotFoundError(
                f"Cannot detect a phonon backend under '{directory}'. "
                "Expected either (OUTCAR + QPOINTS/QPOINTS_OPT) for direct VASP, "
                "or band.yaml / phonopy_disp.yaml for phonopy."
            )

        matters.append([label, 0, qpath, bands, color, lstyle, weight, alpha, tolerance, backend])

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

    # If VASP- and Phonopy-based dispersions are plotted together, their k-path distances can
    # differ by a constant scale (e.g. 2π convention). To make overlays comparable, rescale
    # the Phonopy k-path to the VASP k-path range.
    has_vasp = any(m[-1] == "vasp" for m in matters)
    has_phonopy = any(m[-1] == "phonopy" for m in matters)

    # Choose the reference dataset for axis ticks (prefer VASP when mixed).
    ref_idx = 0
    if has_vasp:
        for i, m in enumerate(matters):
            if m[-1] == "vasp":
                ref_idx = i
                break

    if has_vasp and has_phonopy:
        vasp_max = None
        for m in matters:
            if m[-1] != "vasp":
                continue
            q = np.asarray(m[2], dtype=float)
            if np.isfinite(q).any():
                qmax = float(np.nanmax(q))
                vasp_max = qmax if vasp_max is None else max(vasp_max, qmax)

        if vasp_max is not None and vasp_max > 0.0:
            for m in matters:
                if m[-1] != "phonopy":
                    continue
                q = np.asarray(m[2], dtype=float)
                if not np.isfinite(q).any():
                    continue
                qmax = float(np.nanmax(q))
                if qmax <= 0.0:
                    continue
                scale = vasp_max / qmax
                if abs(scale - 1.0) < 1e-12:
                    continue
                q_scaled = []
                for x in m[2]:
                    try:
                        xf = float(x)
                    except Exception:
                        q_scaled.append(x)
                        continue
                    if np.isfinite(xf):
                        q_scaled.append(xf * scale)
                    else:
                        q_scaled.append(np.nan)
                m[2] = q_scaled

    weighted_qpath = matters[ref_idx][2]

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
    # Determine x-range from all matters (robust when overlaying multiple materials).
    qmins, qmaxs = [], []
    for _m in matters:
        _qx = np.array(_m[2], dtype=float)
        if np.isfinite(_qx).any():
            qmins.append(float(np.nanmin(_qx)))
            qmaxs.append(float(np.nanmax(_qx)))
    if qmins:
        plt.xlim(min(qmins), max(qmaxs))

    # Get the original directory from the original matters_list (format: [label, directory, *optional])
    orig_directory = matters_list[ref_idx][1]
    backend_ref = matters[ref_idx][-1]

    if backend_ref == "phonopy":
        # Phonopy: prefer labels/segments recorded in band.yaml (most reliable).
        ticks, tick_labels = extract_phonopy_high_sym_from_band_yaml(orig_directory)

        # Fallback: infer segmentation from NaNs inserted into weighted_qpath + BAND_LABELS in band.conf.
        if not ticks:
            try:
                seg_nq = extract_phonopy_bands(orig_directory).get("segment_nqpoint", None)
            except Exception:
                seg_nq = None
            ticks, tick_labels = _phonopy_high_sym_positions(orig_directory, weighted_qpath, segment_nqpoint=seg_nq)

        if ticks:
            plt.xticks(ticks, tick_labels)
            for pos in ticks[1:-1]:
                plt.axvline(x=pos, color=annotate_color[1], linestyle="--", alpha=0.8, zorder=1)

    else:
        # Direct VASP: Extract high symmetry boundaries from QPOINTS (keep repeats for multi-segment / branched paths)
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
    
    # Collect minimum-frequency point for each matter (for stability diagnostics).
    minima = {}
    for _m, _m_orig in zip(matters, matters_list):
        _label = _m[0]
        _qx = np.asarray(_m[2], dtype=float)
        _bands = _m[3]
        _best_y = None
        _best_x = None
        for _band in _bands:
            _by = np.asarray(_band, dtype=float)
            _mask = np.isfinite(_qx) & np.isfinite(_by)
            if not _mask.any():
                continue
            _idxs = np.where(_mask)[0]
            _i_min = int(_idxs[np.argmin(_by[_idxs])])
            _y_min = float(_by[_i_min])
            _x_min = float(_qx[_i_min])
            if (_best_y is None) or (_y_min < _best_y):
                _best_y = _y_min
                _best_x = _x_min
        minima[_label] = {
            'x': _best_x,
            'freq_THz': _best_y,
            'backend': _m[-1] if len(_m) > 0 else None,
            'directory': _m_orig[1] if isinstance(_m_orig, (list, tuple)) and len(_m_orig) > 1 else None,
        }

    plt.tight_layout()
    return minima



#### Phonopy phonon dispersion (phonopy band.yaml)

# NOTE:
#   1) This section plots phonon dispersions generated by phonopy (band.yaml).
#   2) matters_list usage is identical to plot_phonons: each matter is
#        [ label, directory, color?, lstyle?, weight?, alpha?, tolerance? ]
#      where directory points to the *phonopy parent directory* (containing band.conf / band.yaml / phonopy_disp.yaml
#      and the displacement subdirectories such as pd-*/).
#   3) This module only plots. If band.yaml does not exist, generate it first in that parent directory:
#        phonopy --vasp -f pd-*/vasprun.xml
#        phonopy -p band.conf


def extract_phonopy_bands(directory, band_yaml="band.yaml"):
    """
    Extract phonon dispersion data from phonopy band.yaml.

    Key pitfall:
      In phonopy band.yaml, "distance" can be either
        (A) global cumulative distance along the entire path (monotonic), or
        (B) per-segment distance restarting from ~0 at each segment.

      If we "stitch" (B) correctly, we must add offsets between segments.
      But if we mistakenly stitch (A), later segments get pushed too far right and the plot looks "broken".

    What we do:
      1) Detect whether distance is globally monotonic (A). If yes: keep it, only insert NaN separators.
      2) Otherwise (B): stitch per segment by adding offsets, and insert NaN separators.
      3) NaN separators are inserted between segments to break polylines (same behavior as phonopy-bandplot).

    Returns:
      {
        "path": [...],              # cumulative distance with NaN separators
        "bands": [[...], ...],      # each band aligned with path (NaNs inserted)
        "segment_nqpoint": list|None
      }
    """
    band_path = os.path.join(directory, band_yaml)
    if not os.path.isfile(band_path):
        raise FileNotFoundError(f"{band_yaml} not found under '{directory}'. (Try: phonopy -p band.conf)")
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise ImportError("PyYAML is required to read band.yaml. Try: pip install pyyaml") from exc

    with open(band_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    phonon_list = data.get("phonon", None) if isinstance(data, dict) else None
    if not phonon_list:
        raise ValueError(f"Invalid {band_yaml}: missing 'phonon' list.")

    seg_nq = data.get("segment_nqpoint", None) if isinstance(data, dict) else None

    distances_raw = []
    bands_raw = None
    for iq, q in enumerate(phonon_list):
        if "distance" not in q:
            raise ValueError(f"Missing 'distance' at phonon[{iq}] in {band_yaml}.")
        distances_raw.append(float(q["distance"]))
        band_entries = q.get("band", None)
        if band_entries is None:
            raise ValueError(f"Missing 'band' at phonon[{iq}] in {band_yaml}.")
        if bands_raw is None:
            nb = len(band_entries)
            bands_raw = [[] for _ in range(nb)]
        for ib, b in enumerate(band_entries):
            freq = b.get("frequency", None)
            if freq is None:
                freq = b.get("freq", None)
            bands_raw[ib].append(np.nan if freq is None else float(freq))
    if bands_raw is None:
        raise ValueError(f"No bands found in {band_yaml}.")

    # Decide segment boundaries
    n_total = len(distances_raw)
    use_seg_nq = isinstance(seg_nq, list) and seg_nq and sum(int(n) for n in seg_nq) == n_total

    # Detect whether distance is globally monotonic (type A)
    tol = 1e-8
    monotonic = True
    for i in range(1, n_total):
        if distances_raw[i] < distances_raw[i-1] - tol:
            monotonic = False
            break

    qpath = []
    bands = [[] for _ in range(len(bands_raw))]

    def _append_point(x, freqs):
        qpath.append(x)
        for ib, v in enumerate(freqs):
            bands[ib].append(v)

    if monotonic:
        # Keep global distance; only insert NaN separators between segments if seg_nq is available.
        if use_seg_nq:
            idx0 = 0
            for iseg, n in enumerate(seg_nq):
                n = int(n)
                for j in range(idx0, idx0 + n):
                    _append_point(float(distances_raw[j]), [bands_raw[ib][j] for ib in range(len(bands_raw))])
                if iseg != len(seg_nq) - 1:
                    _append_point(np.nan, [np.nan] * len(bands_raw))
                idx0 += n
        else:
            # No reliable segment info; still handle reset-detected separators (shouldn't happen if monotonic)
            for j in range(n_total):
                _append_point(float(distances_raw[j]), [bands_raw[ib][j] for ib in range(len(bands_raw))])
    else:
        # Stitch per-segment distance (type B)
        if use_seg_nq:
            offset = 0.0
            idx0 = 0
            for iseg, n in enumerate(seg_nq):
                n = int(n)
                seg_dist = distances_raw[idx0: idx0 + n]
                seg0 = float(seg_dist[0])
                seg_len = float(seg_dist[-1] - seg0) if n > 0 else 0.0
                for j in range(n):
                    x = offset + float(seg_dist[j] - seg0)
                    _append_point(x, [bands_raw[ib][idx0 + j] for ib in range(len(bands_raw))])
                if iseg != len(seg_nq) - 1:
                    _append_point(np.nan, [np.nan] * len(bands_raw))
                offset += seg_len
                idx0 += n
        else:
            # Fallback: detect resets and stitch accordingly
            offset = 0.0
            seg_start = 0
            prev = distances_raw[0]
            for i in range(1, n_total):
                if distances_raw[i] < prev - tol:
                    seg_dist = distances_raw[seg_start:i]
                    seg0 = float(seg_dist[0])
                    seg_len = float(seg_dist[-1] - seg0)
                    for j in range(seg_start, i):
                        x = offset + float(distances_raw[j] - seg0)
                        _append_point(x, [bands_raw[ib][j] for ib in range(len(bands_raw))])
                    _append_point(np.nan, [np.nan] * len(bands_raw))
                    offset += seg_len
                    seg_start = i
                prev = distances_raw[i]
            # last segment
            seg_dist = distances_raw[seg_start:n_total]
            seg0 = float(seg_dist[0])
            for j in range(seg_start, n_total):
                x = offset + float(distances_raw[j] - seg0)
                _append_point(x, [bands_raw[ib][j] for ib in range(len(bands_raw))])

    return {"path": qpath, "bands": bands, "segment_nqpoint": seg_nq}




def _phonopy_parse_number(token):
    # support fraction like 1/2
    try:
        if "/" in token:
            from fractions import Fraction
            return float(Fraction(token))
        return float(token)
    except Exception:
        return None


def extract_phonopy_band_conf(directory, conf_name=None):
    """
    Parse band.conf (or specified conf) to obtain:
      - BAND_POINTS
      - BAND_LABELS (Γ Z T Y ...)
      - BAND q-points (used to infer number of segments)
    Returns: {"band_points": int|None, "nseg": int, "labels": [...]}.
    """
    candidates = []
    if conf_name is not None: candidates.append(os.path.join(directory, conf_name))
    candidates += [os.path.join(directory, "band.conf"), os.path.join(directory, "phonopy.conf"), os.path.join(directory, "phonopy_band.conf")]
    conf_path = None
    for p in candidates:
        if os.path.isfile(p):
            conf_path = p
            break
    if conf_path is None:
        raise FileNotFoundError(f"band.conf not found under '{directory}'.")

    kv = {}
    current_key = None
    with open(conf_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line: continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip().upper()
                val = val.strip()
                current_key = key
                kv.setdefault(key, [])
                if val: kv[key].append(val)
            else:
                if current_key is not None: kv[current_key].append(line)

    band_points = None
    if "BAND_POINTS" in kv:
        try: band_points = int(" ".join(kv["BAND_POINTS"]).split()[0])
        except Exception: band_points = None

    labels = []
    if "BAND_LABELS" in kv:
        labels = " ".join(kv["BAND_LABELS"]).split()

    if "BAND" not in kv:
        raise ValueError(f"'BAND' missing in {conf_path}.")
    band_tokens = " ".join(kv["BAND"]).split()
    if len(band_tokens) % 3 != 0:
        raise ValueError(f"Invalid BAND token count in {conf_path}: {len(band_tokens)} (must be multiple of 3).")
    band_pts = []
    for i in range(0, len(band_tokens), 3):
        x = _phonopy_parse_number(band_tokens[i])
        y = _phonopy_parse_number(band_tokens[i+1])
        z = _phonopy_parse_number(band_tokens[i+2])
        if x is None or y is None or z is None:
            raise ValueError(f"Invalid BAND entry near: {band_tokens[i:i+3]}")
        band_pts.append([x, y, z])
    if labels and len(labels) == len(band_pts):
        nseg = len(band_pts) - 1
        boundary_labels = labels
    else:
        if len(band_pts) % 2 != 0:
            raise ValueError(
                f"Invalid BAND point count in {conf_path}: {len(band_pts)}. "
                "It is neither a compact path matching BAND_LABELS nor an even-numbered pairwise path."
            )
        nseg = len(band_pts) // 2

        # boundary labels count should be nseg+1
        if labels:
            if len(labels) == nseg + 1:
                boundary_labels = labels
            else:
                boundary_labels = (labels + [""] * (nseg + 1))[: (nseg + 1)]
        else:
            boundary_labels = [f"P{i}" for i in range(nseg + 1)]
    return {"conf_path": conf_path, "band_points": band_points, "nseg": nseg, "boundary_labels": boundary_labels}

def _phonopy_clean_label(label):
    """Normalize labels coming from band.conf / phonopy.yaml / band.yaml.
    - Convert common LaTeX-like strings (e.g. '$\\Gamma$') to Unicode (e.g. 'Γ').
    - Strip quotes and surrounding '$'.
    """
    if label is None:
        return ""
    s = str(label).strip().strip("'").strip('"').strip()
    # Remove surrounding $...$
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    # Remove any remaining '$'
    s = s.replace("$", "").strip()

    # Convert common LaTeX greek macros to unicode
    greek_map = {
        r"\Gamma": "Γ",
        r"\Delta": "Δ",
        r"\Lambda": "Λ",
        r"\Sigma": "Σ",
        r"\Xi": "Ξ",
        r"\Pi": "Π",
        r"\Phi": "Φ",
        r"\Psi": "Ψ",
        r"\Omega": "Ω",
    }
    for k, v in greek_map.items():
        s = s.replace(k, v)

    # A few alternative notations
    # A few alternative notations
    s = s.replace("\\\\", "\\")  # collapse doubled backslashes if any
    s = s.replace("\\Gamma", "Γ").replace("\\Delta", "Δ").replace("\\Sigma", "Σ")
    # Final trim
    return s.strip()

def _phonopy_boundary_labels_from_band_conf(directory, nseg):
    try:
        conf = extract_phonopy_band_conf(directory)
    except Exception:
        return None
    labels = conf.get("boundary_labels") or []
    labels = [_phonopy_clean_label(x) for x in labels]
    if len(labels) == nseg + 1:
        return labels
    return None

def _phonopy_boundary_labels_from_phonopy_yaml(directory, nseg):
    # phonopy writes either phonopy.yaml or phonopy_disp.yaml (phonopy-yaml mode).
    for fname in ("phonopy.yaml", "phonopy_disp.yaml"):
        path = os.path.join(directory, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    m = re.match(r"^\s*band_labels\s*:\s*['\"]?([^'\"]+)['\"]?\s*$", line)
                    if m:
                        labels = [_phonopy_clean_label(x) for x in m.group(1).split()]
                        if len(labels) == nseg + 1:
                            return labels
        except Exception:
            pass
    return None

def _phonopy_boundary_labels_from_band_yaml(directory, nseg):
    # Parse band.yaml's "labels:" section, which is a list of pairs:
    #   labels:
    #   - [ '$\\Gamma$', 'Z' ]
    #   - [ 'Z', 'T' ]
    path = os.path.join(directory, "band.yaml")
    if not os.path.isfile(path):
        return None

    pairs = []
    in_labels = False
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not in_labels:
                    if re.match(r"^\s*labels\s*:\s*$", line):
                        in_labels = True
                    continue
                # End labels section when we hit a new top-level key
                if re.match(r"^[A-Za-z_].*:\s*$", line) and not line.startswith(" "):
                    break
                m = re.match(r"^\s*-\s*\[\s*(.+?)\s*,\s*(.+?)\s*\]\s*$", line)
                if not m:
                    continue
                a, b = m.group(1), m.group(2)
                pairs.append((_phonopy_clean_label(a), _phonopy_clean_label(b)))

        if not pairs:
            return None
        # boundary labels: first label of first pair, then the second label of each pair
        boundary = [pairs[0][0]] + [p[1] for p in pairs]
        if len(boundary) == nseg + 1:
            return boundary
    except Exception:
        return None
    return None

def _phonopy_get_boundary_labels(directory, nseg):
    """Return boundary labels (length nseg+1) with robust fallbacks."""
    for fn in (
        _phonopy_boundary_labels_from_band_conf,
        _phonopy_boundary_labels_from_phonopy_yaml,
        _phonopy_boundary_labels_from_band_yaml,
    ):
        labels = fn(directory, nseg)
        if labels:
            return labels
    return None



def _phonopy_high_sym_positions(directory, qpath, segment_nqpoint=None):
    """
    Build high-symmetry tick positions for phonopy bands.

    We do NOT use duplicated boundary points (which would create labels like 'Z|T').
    Instead, we infer boundary positions from segment ends on the stitched qpath:
      boundaries = [start_of_first_segment] + [end_of_each_segment]

    qpath may contain NaNs as separators (inserted by extract_phonopy_bands).
    Returns: (ticks, tick_labels)
    """
    conf = extract_phonopy_band_conf(directory)
    nseg = conf["nseg"]
    labels = _phonopy_get_boundary_labels(directory, nseg) or conf.get("boundary_labels")

    # Split into segments using NaNs
    segments = []
    current = []
    for x in qpath:
        if isinstance(x, float) and np.isnan(x):
            if current:
                segments.append(current)
                current = []
        else:
            current.append(float(x))
    if current:
        segments.append(current)

    if not segments:
        return [], []

    # boundary positions: start of first segment, then end of each segment (nseg segments expected)
    start0 = segments[0][0]
    ends = [seg[-1] for seg in segments[:nseg]]  # truncate if extra
    boundary_positions = [start0] + ends

    # Match labels length
    if labels:
        labels = [_phonopy_clean_label(x) for x in labels]
        boundary_labels = (labels + [""] * len(boundary_positions))[: len(boundary_positions)]
    else:
        boundary_labels = [f"P{i}" for i in range(len(boundary_positions))]

    return boundary_positions, boundary_labels


def create_matters_phonopy(matters_list):
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
        bands_data = extract_phonopy_bands(directory)
        qpath = bands_data["path"]
        bands = bands_data["bands"]
        matters.append([label, 0, qpath, bands, color, lstyle, weight, alpha, tolerance])
    return matters


def plot_phonopy(title, matters_list=None, eigen_range=None, legend_loc=False):
    """Backward-compatible alias.

    plot_phonons() now auto-detects whether each directory is a direct VASP phonon run
    or a phonopy parent directory. Therefore, plot_phonopy simply forwards to plot_phonons.
    """
    return plot_phonons(title, matters_list=matters_list, eigen_range=eigen_range, legend_loc=legend_loc)