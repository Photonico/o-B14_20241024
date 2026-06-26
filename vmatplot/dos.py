#### Declarations of process functions for DoS with vectorized programming
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

# Necessary packages invoking
import xml.etree.ElementTree as ET
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from vmatplot.commons import extract_fermi, get_or_default, check_spin
from vmatplot.output_settings import color_sampling, canvas_setting
from functools import lru_cache

import matplotlib as mpl

mpl.rcParams["lines.solid_capstyle"] = "round"
mpl.rcParams["lines.dash_capstyle"]  = "round"
mpl.rcParams["lines.solid_joinstyle"] = "round"
mpl.rcParams["lines.dash_joinstyle"]  = "round"

def cal_type(directory_path):
    kpoints_file_path = os.path.join(directory_path, "KPOINTS")
    kpoints_opt_path = os.path.join(directory_path, "KPOINTS_OPT")
    if os.path.exists(kpoints_opt_path):
        return "GGA-PBE"
    elif os.path.exists(kpoints_file_path):
        return "HSE06"

def extract_dos(directory_path, spin=1, negate=False, read_eigen=False):
    """
    Extract DOS data from VASP DOSCAR (instead of vasprun.xml).
    Parameters
    directory_path : str
        Directory containing DOSCAR.
    spin : int
        Spin channel index: 1 (spin up) or 2 (spin down). For non-spin-polarized DOSCAR, spin is ignored.
    negate : bool
        If True, multiply DOS and integrated DOS by -1 (useful for plotting spin-down as negative).
    read_eigen : bool
        Kept for API compatibility. DOSCAR does not contain eigenvalue/occupancy matrices in the same way;
        therefore eigen_matrix and occu_matrix are returned as None.
    Returns
    tuple
        (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,
         energy_dos_shift, total_dos_list, integrated_dos_list)
    """
    # Helper: parse number of ions from CONTCAR/POSCAR (fast and robust)
    def _read_ions_number_from_poscar_like(poscar_path):
        try:
            with open(poscar_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [next(f) for _ in range(8)]
        except Exception:
            return None
        # POSCAR format:
        # 0 comment
        # 1 scale
        # 2-4 lattice vectors
        # 5 element symbols OR element counts (VASP4 style)
        # 6 element counts if symbols present
        def _is_int_token(tok):
            try:
                int(tok)
                return True
            except Exception:
                # Some files might write counts as floats like "2.0"
                try:
                    return float(tok).is_integer()
                except Exception:
                    return False
        tokens6 = lines[5].split() if len(lines) > 5 else []
        tokens7 = lines[6].split() if len(lines) > 6 else []
        counts = None
        if tokens6 and all(_is_int_token(t) for t in tokens6):
            counts = [int(float(t)) for t in tokens6]
        elif tokens6 and tokens7 and (not all(_is_int_token(t) for t in tokens6)) and all(_is_int_token(t) for t in tokens7):
            counts = [int(float(t)) for t in tokens7]
        if counts:
            return int(sum(counts))
        return None

    # Helper: try to get k-points number from IBZKPT (optional)
    def _read_kpoints_number_from_ibzkpt(ibzkpt_path):
        try:
            with open(ibzkpt_path, "r", encoding="utf-8", errors="ignore") as f:
                _ = f.readline()  # comment
                line2 = f.readline()
            return int(line2.split()[0])
        except Exception:
            return None

    # DOSCAR path check
    doscar_path = os.path.join(directory_path, "DOSCAR")
    if not os.path.isfile(doscar_path):
        print(f"Error: The file DOSCAR does not exist in the directory {directory_path}.")
        return
    # ions_number (best-effort)
    ions_number = None
    contcar_path = os.path.join(directory_path, "CONTCAR")
    poscar_path  = os.path.join(directory_path, "POSCAR")
    if os.path.isfile(contcar_path):
        ions_number = _read_ions_number_from_poscar_like(contcar_path)
    if ions_number is None and os.path.isfile(poscar_path):
        ions_number = _read_ions_number_from_poscar_like(poscar_path)
    # kpoints_number (best-effort)
    kpoints_number = None
    ibzkpt_path = os.path.join(directory_path, "IBZKPT")
    if os.path.isfile(ibzkpt_path):
        kpoints_number = _read_kpoints_number_from_ibzkpt(ibzkpt_path)
    # Parse DOSCAR: locate the "DOS grid header" line, then read NEDOS DOS rows
    # Header line typically: EMAX  EMIN  NEDOS  EFERMI  (something)
    # DOS rows:
    #   non-spin:  E  DOS  IntDOS
    #   spin:      E  DOS(up) DOS(dn) IntDOS(up) IntDOS(dn)
    emax = emin = efermi = None
    nedos = None
    with open(doscar_path, "r", encoding="utf-8", errors="ignore") as f:
        # Find the DOS header line for the TOTAL DOS block
        # (Usually appears after 5 header lines, but we scan to be robust.)
        header_found = False
        for _ in range(2000):
            line = f.readline()
            if not line: break
            toks = line.split()
            if len(toks) < 4: continue
            try:
                _emax = float(toks[0])
                _emin = float(toks[1])
                _nedos = int(float(toks[2]))
                _efermi = float(toks[3])
                # Basic sanity checks to avoid false positives
                if _nedos > 10 and _emax > _emin and abs(_efermi) < 1.0e4:
                    emax, emin, nedos, efermi = _emax, _emin, _nedos, _efermi
                    header_found = True
                    break
            except Exception: continue
        if not header_found:
            print("Error: Failed to locate the DOS header line in DOSCAR.")
            return
        # Read the first DOS row to determine the number of columns
        first_row = f.readline()
        if not first_row:
            print("Error: DOSCAR ended unexpectedly while reading DOS rows.")
            return
        first_tokens = first_row.split()
        ncols = len(first_tokens)
        if ncols < 3:
            print("Error: Unexpected DOS row format in DOSCAR (too few columns).")
            return
        # Read remaining DOS rows (NEDOS total rows)
        dos_lines = [first_row]
        for _ in range(nedos - 1):
            row = f.readline()
            if not row:
                break
            dos_lines.append(row)
    # Convert DOS block to numpy array efficiently
    flat = np.fromstring("".join(dos_lines), sep=" ")
    if flat.size % ncols != 0:
        # Fallback: try splitting line-by-line if formatting is irregular
        data = []
        for row in dos_lines:
            parts = row.split()
            if len(parts) == ncols:
                data.append([float(x) for x in parts])
        dos = np.array(data, dtype=float)
    else: dos = flat.reshape(-1, ncols)
    # Select total/integrated DOS columns
    energy = dos[:, 0]
    if ncols >= 5:
        # Spin-polarized total DOS block: E, DOS(up), DOS(dn), Int(up), Int(dn)
        if int(spin) == 2:
            total_dos_list = dos[:, 2]
            integrated_dos_list = dos[:, 4]
        else:
            total_dos_list = dos[:, 1]
            integrated_dos_list = dos[:, 3]
    else:
        # Non-spin-polarized total DOS block: E, DOS, IntDOS
        total_dos_list = dos[:, 1]
        integrated_dos_list = dos[:, 2]
    # Shift energy by Fermi level
    energy_dos_shift = energy - efermi
    # Optional negation (commonly used for plotting spin-down as negative)
    if negate:
        total_dos_list = -1.0 * total_dos_list
        integrated_dos_list = -1.0 * integrated_dos_list
    # DOSCAR does not provide eigenvalue/occupancy matrices in this function's original layout
    eigen_matrix = None
    occu_matrix = None
    if read_eigen:
        # Kept intentionally silent-ish: do not break workflows, but clearly indicate limitation.
        print("Warning: read_eigen=True requested, but DOSCAR-based extractor does not provide eigen/occu matrices. Returning None for eigen_matrix and occu_matrix.")
    return (
        efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,
        energy_dos_shift, total_dos_list, integrated_dos_list
    )

def extract_dos_dev(directory_path, spin=1, negate=False, read_eigen=False):
    """
    Extract DOS data from VASP vasprun.xml.
    Parameters
    directory_path : str
        Directory containing vasprun.xml
    spin : int
        Spin channel index: 1 (spin up) or 2 (spin down)
    negate : bool
        If True, multiply DOS and integrated DOS by -1 (useful for plotting spin-down as negative)
    read_eigen : bool
        If True, also parse eigenvalues/occupancies (can be very slow for large systems)
    Returns
    -------
    tuple
        (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,
         energy_dos_shift, total_dos, integrated_dos)
        Notes: eigen_matrix and occu_matrix are None unless read_eigen=True.
    """
    # Build the full path to vasprun.xml
    file_path = os.path.join(directory_path, "vasprun.xml")
    if not os.path.isfile(file_path):
        print(f"Error: The file vasprun.xml does not exist in the directory {directory_path}.")
        return
    # Parse XML once
    tree = ET.parse(file_path)
    root = tree.getroot()
    # Read Fermi energy (your existing helper)
    efermi = extract_fermi(directory_path)
    # Number of ions: prefer atominfo (fast and robust)
    atom_el = root.find(".//atominfo/atoms")
    ions_number = int(atom_el.text.strip()) if atom_el is not None else None
    # Determine whether this run uses kpoints_opt (e.g., HSE-like workflows)
    kpoints_file_path = os.path.join(directory_path, "KPOINTS")
    kpoints_opt_path  = os.path.join(directory_path, "KPOINTS_OPT")
    use_opt = os.path.exists(kpoints_opt_path)
    # Get number of k-points (fast: just count <v> nodes, no float conversion needed)
    kpoints_number = None
    if use_opt:
        kp_varray = root.find(".//eigenvalues_kpoints_opt[@comment='kpoints_opt']/kpoints/varray[@name='kpointlist']")
        if kp_varray is None:
            kp_varray = root.find(".//kpoints/varray[@name='kpointlist']")
    else:
        kp_varray = root.find(".//kpoints/varray[@name='kpointlist']")
        if kp_varray is None:
            kp_varray = root.find(".//varray[@name='kpointlist']")
    if kp_varray is not None:
        kpoints_number = len(kp_varray.findall("v"))
    # DOS extraction
    if use_opt:
        path_dos = f"./calculation/dos[@comment='kpoints_opt']/total/array/set/set[@comment='spin {spin}']/r"
    else:
        path_dos = f".//total/array/set/set[@comment='spin {spin}']/r"
    r_nodes = root.findall(path_dos)
    if not r_nodes:
        print("Error: DOS nodes not found in vasprun.xml (check the XPath and VASP version).")
        return
    # Each <r> line is: energy  total_dos  integrated_dos
    flat = np.fromstring(" ".join(n.text for n in r_nodes), sep=" ")
    dos = flat.reshape(-1, 3)
    energy_dos_shift = dos[:, 0] - efermi
    total_dos_list = dos[:, 1]
    integrated_dos_list = dos[:, 2]
    if negate:
        total_dos_list *= -1.0
        integrated_dos_list *= -1.0

    # Optional: eigenvalues/occupancies
    eigen_matrix = None
    occu_matrix = None

    if read_eigen:
        if use_opt: spin_set = root.find(f"./calculation/projected_kpoints_opt/eigenvalues/array/set/set[@comment='spin {spin}']")
        else: spin_set = root.find(f".//eigenvalues/array/set/set[@comment='spin {spin}']")
        if spin_set is None: print("Warning: eigenvalues set not found; eigen_matrix and occu_matrix will be None.")
        else:
            k_sets = spin_set.findall("set")
            if kpoints_number is None:
                kpoints_number = len(k_sets)
            # Parse the first k-point to infer number of bands
            first_k = k_sets[0]
            r0 = first_k.findall("r") or list(first_k)
            nbands = len(r0)
            eigen_matrix = np.empty((nbands, len(k_sets)), dtype=float)
            occu_matrix  = np.empty((nbands, len(k_sets)), dtype=float)
            for ik, kset in enumerate(k_sets):
                rlist = kset.findall("r") or list(kset)
                block = np.fromstring(" ".join(r.text for r in rlist), sep=" ").reshape(-1, 2)
                eigen_matrix[:, ik] = block[:, 0]
                occu_matrix[:, ik]  = block[:, 1]
    return (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,
            energy_dos_shift, total_dos_list, integrated_dos_list)

def extract_dos_backup(directory_path):
    ## Construct the full path to the vasprun.xml file
    file_path = os.path.join(directory_path, "vasprun.xml")
    # Check if the vasprun.xml file exists in the given directory
    if not os.path.isfile(file_path):
        print(f"Error: The file vasprun.xml does not exist in the directory {directory_path}.")
        return

    ## Analysis vasprun.xml file
    tree = ET.parse(file_path)
    root = tree.getroot()
    kpoints_file_path = os.path.join(directory_path, "KPOINTS")
    kpoints_opt_path = os.path.join(directory_path, "KPOINTS_OPT")

    ## Extract Fermi energy
    # efermi_element = root.find(".//dos/i[@name='efermi']")
    # efermi = float(efermi_element.text.strip())
    efermi = extract_fermi(directory_path)

    ## Extract the number of ions
    first_positions = root.find(".//varray[@name='positions'][1]")
    positions_concatenated_text = " ".join([position.text for position in first_positions.findall("v")])
    positions_array = np.fromstring(positions_concatenated_text, sep=" ")
    positions_matrix = positions_array.reshape(-1, 3)
    ions_number = positions_matrix.shape[0]

    ## Extract the number of kpoints
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        kpointlist = root.find(".//eigenvalues_kpoints_opt[@comment='kpoints_opt']/kpoints/varray[@name='kpointlist']")
        kpointlist_concatenated_text = " ".join([kpointlist.text for kpointlist in kpointlist.findall("v")])
        kpointlist_array = np.fromstring(kpointlist_concatenated_text, sep=" ")
        kpointlist_matrix = kpointlist_array.reshape(-1, 3)
        kpoints_number = kpointlist_matrix.shape[0]
    # PBE algorithms
    elif os.path.exists(kpoints_file_path):
        kpointlist = root.find(".//varray[@name='kpointlist']")
        kpointlist_concatenated_text = " ".join([kpointlist.text for kpointlist in kpointlist.findall("v")])
        kpointlist_array = np.fromstring(kpointlist_concatenated_text, sep=" ")
        kpointlist_matrix = kpointlist_array.reshape(-1, 3)
        kpoints_number = kpointlist_matrix.shape[0]

    ## Extract eigen, occupancy number
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        for kpoints_index in range(1, kpoints_number+1):
            xpath_expr = f"./calculation/projected_kpoints_opt/eigenvalues/array/set/set[@comment='spin 1']/set[@comment='kpoint {kpoints_index}']"
            eigen_column = np.empty(0)
            occu_column  = np.empty(0)
            kpoint_set = root.find(xpath_expr)
            for eigen_occ_element in kpoint_set:
                values_eigen = list(map(float, eigen_occ_element.text.split()))
                eigen_var = values_eigen[0]
                eigen_column = np.append(eigen_column, eigen_var)
                occu_var = values_eigen[1]
                occu_column = np.append(occu_column, occu_var)
            if kpoints_index == 1 :
                eigen_matrix = eigen_column.reshape(-1, 1)
                occu_matrix = occu_column.reshape(-1, 1)
            else:
                eigen_matrix = np.hstack((eigen_matrix,eigen_column.reshape(-1, 1)))
                occu_matrix  = np.hstack((occu_matrix, occu_column.reshape(-1, 1)))
    # GGA-PBE algorithms
    elif os.path.exists(kpoints_file_path):
        for kpoints_index in range(1, kpoints_number+1):
            xpath_expr = f".//set[@comment='kpoint {kpoints_index}']"
            eigen_column = np.empty(0)
            occu_column  = np.empty(0)
            kpoint_set = root.find(xpath_expr)
            for eigen_occ_element in kpoint_set:
                values_eigen = list(map(float, eigen_occ_element.text.split()))
                eigen_var = values_eigen[0]
                eigen_column = np.append(eigen_column, eigen_var)
                occu_var = values_eigen[1]
                occu_column = np.append(occu_column, occu_var)
            if kpoints_index == 1 :
                eigen_matrix = eigen_column.reshape(-1, 1)
                occu_matrix = occu_column.reshape(-1, 1)
            else:
                eigen_matrix = np.hstack((eigen_matrix,eigen_column.reshape(-1, 1)))
                occu_matrix  = np.hstack((occu_matrix, occu_column.reshape(-1, 1)))

    ## Extract energy, Total DoS, and Integrated DoS
    # lists initialization
    energy_dos_list     = np.array([])
    total_dos_list      = np.array([])
    integrated_dos_list = np.array([])

    if os.path.exists(kpoints_opt_path):
        path_dos = "./calculation/dos[@comment='kpoints_opt']/total/array/set/set[@comment='spin 1']/r"
    elif os.path.exists(kpoints_file_path):
        path_dos = ".//total/array/set/set[@comment='spin 1']/r"

    for element_dos in root.findall(path_dos):
        values_dos = list(map(float, element_dos.text.split()))
        energy_dos_list = np.append(energy_dos_list, values_dos[0])
        total_dos_list = np.append(total_dos_list, values_dos[1])
        integrated_dos_list = np.append(integrated_dos_list, values_dos[2])
    shift = efermi
    energy_dos_shift = energy_dos_list - shift

    return (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,             # 0 ~ 4
            energy_dos_shift, total_dos_list, integrated_dos_list)                      # 5 ~ 7


@lru_cache(maxsize=None)
def extract_dos_fast_cached(directory_path, spin=1, negate=False):
    return extract_dos(directory_path, spin=spin, negate=negate)

def extract_dos_spin_up(directory_path):
    return extract_dos(directory_path)

def extract_dos_spin_down(directory_path, negate_label=False):
    ## Construct the full path to the vasprun.xml file
    file_path = os.path.join(directory_path, "vasprun.xml")
    # Check if the vasprun.xml file exists in the given directory
    if not os.path.isfile(file_path):
        print(f"Error: The file vasprun.xml does not exist in the directory {directory_path}.")
        return

    ## Analysis vasprun.xml file
    tree = ET.parse(file_path)
    root = tree.getroot()
    kpoints_file_path = os.path.join(directory_path, "KPOINTS")
    kpoints_opt_path = os.path.join(directory_path, "KPOINTS_OPT")

    ## Extract Fermi energy
    # efermi_element = root.find(".//dos/i[@name='efermi']")
    # efermi = float(efermi_element.text.strip())
    efermi = extract_fermi(directory_path)

    ## Extract the number of ions
    first_positions = root.find(".//varray[@name='positions'][1]")
    positions_concatenated_text = " ".join([position.text for position in first_positions.findall("v")])
    positions_array = np.fromstring(positions_concatenated_text, sep=" ")
    positions_matrix = positions_array.reshape(-1, 3)
    ions_number = positions_matrix.shape[0]

    ## Extract the number of kpoints
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        kpointlist = root.find(".//eigenvalues_kpoints_opt[@comment='kpoints_opt']/kpoints/varray[@name='kpointlist']")
        kpointlist_concatenated_text = " ".join([kpointlist.text for kpointlist in kpointlist.findall("v")])
        kpointlist_array = np.fromstring(kpointlist_concatenated_text, sep=" ")
        kpointlist_matrix = kpointlist_array.reshape(-1, 3)
        kpoints_number = kpointlist_matrix.shape[0]
    # PBE algorithms
    elif os.path.exists(kpoints_file_path):
        kpointlist = root.find(".//varray[@name='kpointlist']")
        kpointlist_concatenated_text = " ".join([kpointlist.text for kpointlist in kpointlist.findall("v")])
        kpointlist_array = np.fromstring(kpointlist_concatenated_text, sep=" ")
        kpointlist_matrix = kpointlist_array.reshape(-1, 3)
        kpoints_number = kpointlist_matrix.shape[0]

    ## Extract eigen, occupancy number
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        for kpoints_index in range(1, kpoints_number+1):
            xpath_expr = f"./calculation/projected_kpoints_opt/eigenvalues/array/set/set[@comment='spin 2']/set[@comment='kpoint {kpoints_index}']"
            eigen_column = np.empty(0)
            occu_column  = np.empty(0)
            kpoint_set = root.find(xpath_expr)
            for eigen_occ_element in kpoint_set:
                values_eigen = list(map(float, eigen_occ_element.text.split()))
                eigen_var = values_eigen[0]
                eigen_column = np.append(eigen_column, eigen_var)
                occu_var = values_eigen[1]
                occu_column = np.append(occu_column, occu_var)
            if kpoints_index == 1 :
                eigen_matrix = eigen_column.reshape(-1, 1)
                occu_matrix = occu_column.reshape(-1, 1)
            else:
                eigen_matrix = np.hstack((eigen_matrix,eigen_column.reshape(-1, 1)))
                occu_matrix  = np.hstack((occu_matrix, occu_column.reshape(-1, 1)))
    # GGA-PBE algorithms
    elif os.path.exists(kpoints_file_path):
        for kpoints_index in range(1, kpoints_number+1):
            xpath_expr = f".//set[@comment='kpoint {kpoints_index}']"
            eigen_column = np.empty(0)
            occu_column  = np.empty(0)
            kpoint_set = root.find(xpath_expr)
            for eigen_occ_element in kpoint_set:
                values_eigen = list(map(float, eigen_occ_element.text.split()))
                eigen_var = values_eigen[0]
                eigen_column = np.append(eigen_column, eigen_var)
                occu_var = values_eigen[1]
                occu_column = np.append(occu_column, occu_var)
            if kpoints_index == 1 :
                eigen_matrix = eigen_column.reshape(-1, 1)
                occu_matrix = occu_column.reshape(-1, 1)
            else:
                eigen_matrix = np.hstack((eigen_matrix,eigen_column.reshape(-1, 1)))
                occu_matrix  = np.hstack((occu_matrix, occu_column.reshape(-1, 1)))

    ## Extract energy, Total DoS, and Integrated DoS
    # lists initialization
    energy_dos_list     = np.array([])
    total_dos_list      = np.array([])
    integrated_dos_list = np.array([])

    if os.path.exists(kpoints_opt_path):
        path_dos = "./calculation/dos[@comment='kpoints_opt']/total/array/set/set[@comment='spin 2']/r"
    elif os.path.exists(kpoints_file_path):
        path_dos = ".//total/array/set/set[@comment='spin 2']/r"

    for element_dos in root.findall(path_dos):
        values_dos = list(map(float, element_dos.text.split()))
        energy_dos_list = np.append(energy_dos_list, values_dos[0])
        total_dos_list = np.append(total_dos_list, values_dos[1])
        integrated_dos_list = np.append(integrated_dos_list, values_dos[2])
    shift = efermi
    energy_dos_shift = energy_dos_list - shift

    if negate_label is True:
        return (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,             # 0 ~ 4
                energy_dos_shift, total_dos_list*(-1), integrated_dos_list*(-1))            # 5 ~ 7
    else:
        return (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,             # 0 ~ 4
                energy_dos_shift, total_dos_list, integrated_dos_list)                      # 5 ~ 7

# DoS Plotting
def create_matters_dos(matters_list):
    """
    Create a list of structured lists for DoS (Density of States) plotting.
    Parameters:
    - matters_list: A list of lists, where each inner list can contain:
      [label, directory, line_color, line_style, line_weight, line_alpha].
    Returns:
    - A list of lists, where each list contains:
      - label: Matter label;
      - dos_data: Extracted DoS data;
      - spin_direction: unpolarized or spin up or spin down when the spin polarization is active;
      - line_color: Color family for plotting;
      - line_style: Line style for plotting;
      - line_weight: Line width for plotting;
      - line_alpha: Line transparency (alpha value) for plotting.
    """
    # Default values for optional parameters
    default_values = {
        "line_color": "default",
        "line_style": "solid",
        "line_weight": 1.5,
        "line_alpha": 1.0,
    }
    # Ensure input is a list of lists
    if isinstance(matters_list, list) and matters_list and not any(isinstance(i, list) for i in matters_list):
        source_data = matters_list[:]
        matters_list.clear()
        matters_list.append(source_data)
    matters = []
    for matter_dir in matters_list:
        # Unpack the list with optional parameters
        label, directory, spin_direction, *optional_params = matter_dir
        line_color = get_or_default(optional_params[0] if len(optional_params) > 0 else None, default_values["line_color"])
        line_style = get_or_default(optional_params[1] if len(optional_params) > 1 else None, default_values["line_style"])
        line_weight = get_or_default(optional_params[2] if len(optional_params) > 2 else None, default_values["line_weight"])
        line_alpha = get_or_default(optional_params[3] if len(optional_params) > 3 else None, default_values["line_alpha"])

        # Extract DoS data
        spin_label = check_spin(directory)
        if spin_label is False:
            dos_data = extract_dos(directory)
            if spin_direction not in ["unpolarized", "non-polarized", "spin off", "spin-off"]:
                print("if the spin polarization is turn-on, please input 'spin up' or 'spin down', if not, please input 'unpolarized'.")
        elif spin_label is True:
            if spin_direction.lower() in ["up", "spin up", "spin-up"]:
                dos_data = extract_dos_spin_up(directory)
            elif spin_direction.lower() in ["down", "spin down", "spin-down"]:
                dos_data = extract_dos_spin_down(directory, False)
            elif spin_direction.lower() in ["negative spin down", "negative spin-down"]:
                dos_data = extract_dos_spin_down(directory, True)
            else: print("if the spin polarization is turn-on, please input 'spin up', 'spin down', or 'negative spin down', if not, please input 'unpolarized'.")

        # Append structured matter list
        spin_direction_label = None
        if spin_label is False:
            spin_direction_label = "unpolarized"
        elif spin_label is True:
            if spin_direction.lower() in ["up", "spin up", "spin-up"]:
                spin_direction_label = "spin-up"
            elif spin_direction.lower() in ["down", "spin down", "spin-down"]:
                spin_direction_label = "spin-down"
            elif spin_direction.lower() in ["negative spin down", "negative spin-down"]:
                spin_direction_label = "spin-down"

        matters.append([label, dos_data, spin_direction_label, line_color, line_style, line_weight, line_alpha])
    return matters

# Universal DoS Plotting
def plot_dos(title, matters_list = None, x_range = None, y_lim = None, dos_quantity = None):
    # Help information
    help_info = "Usage: plot_dos \n" + \
                "Use extract_dos to extract the DoS data into a two-dimensional list firstly.\n"

    if title in ["help", "Help"]:
        print(help_info)
        return

    # Figure Settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi = fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    # Color calling
    fermi_color = color_sampling("Violet")
    matters = create_matters_dos(matters_list)

    def _dos_plain_label(current_label, spin_direction_label):
        if spin_direction_label is None:
            return f"{current_label}" if current_label not in [None, ""] else ""
        if current_label in [None, ""]:
            return f"{spin_direction_label}"
        return f"{current_label} ({spin_direction_label})"

    if all(term is not None for term in [x_range, y_lim]):
        # Data plotting
        if dos_quantity in ["All", "all"]:
            for _, matter in enumerate(matters):
                # Labels
                current_label = matter[0]
                current_label = _dos_plain_label(current_label, matter[2])
                plt.plot(matter[1][5], matter[1][6], c=color_sampling(matter[3])[1], linestyle=matter[4], lw=matter[5], alpha=matter[6], label=f"{current_label}", zorder=3)
                plt.plot(matter[1][5], matter[1][7], c=color_sampling(matter[3])[2], linestyle=matter[4], lw=matter[5], alpha=matter[6], label=f"{current_label}", zorder=2)
                efermi = matter[1][0]
        if dos_quantity in ["Total", "total"]:
            for _, matter in enumerate(matters):
                # Labels
                current_label = matter[0]
                current_label = _dos_plain_label(current_label, matter[2])
                plt.plot(matter[1][5], matter[1][6], c=color_sampling(matter[3])[1], linestyle=matter[4], lw=matter[5], alpha=matter[6], label=f"{current_label}", zorder=2)
                efermi = matter[1][0]
        if dos_quantity in ["Integrated", "integrated"]:
            for _, matter in enumerate(matters):
                # Labels
                current_label = matter[0]
                current_label = _dos_plain_label(current_label, matter[2])
                plt.plot(matter[1][5], matter[1][7], c=color_sampling(matter[3])[2], linestyle=matter[4], lw=matter[5], alpha=matter[6], label=f"{current_label}", zorder=2)
                efermi = matter[1][0]
        # Plot Fermi energy as a vertical line
        shift = efermi
        plt.axvline(x = efermi-shift, linestyle="--", c=fermi_color[0], alpha=0.80, label="Fermi energy", zorder = 1)
        fermi_energy_text = f"Fermi energy\n{efermi:.3f} (eV)"
        if len(matters) == 1:
            plt.text(efermi-shift-x_range*0.02, y_lim*0.98, fermi_energy_text, fontsize =1.0*12, c=fermi_color[0], rotation=0, va = "top", ha="right")
        else: pass
        # Title
        # plt.title(f"Electronic density of state for {title} ({supplement})")
        plt.title(f"{title}")
        plt.ylabel(r"Density of States"); plt.xlabel(r"Energy (eV)")
        # axes limit
        plt.xlim(x_range*(-1), x_range)
        if isinstance(y_lim, (int, float)):
            plt.ylim(None, y_lim)
        elif isinstance(y_lim, (list, tuple, np.ndarray)) and len(y_lim) == 1:
            plt.ylim(None, y_lim[0])
        elif isinstance(y_lim, (list, tuple, np.ndarray)) and len(y_lim) > 1:
            plt.ylim(y_lim[0],y_lim[-1])
        y_bot = plt.ylim()[0]
        if y_bot < 0:
            plt.axhline(y=0, linestyle="--", c=color_sampling("Grey")[1], zorder = 1)

        plt.legend(loc="best")
        # plt.legend(loc="upper right")
        plt.tight_layout()


# Spin-polarized total DoS Plotting from one calculation
# The following spin-DOS functions do not require DOSCAR.
# They read the VASP DOS table from vasprun.xml when available, and otherwise
# reconstruct a Gaussian-broadened DOS from OUTCAR eigenvalues.

def _extract_fermi_from_outcar(directory_path):
    """Extract the last reported Fermi energy from OUTCAR. Internal helper."""
    outcar_path = os.path.join(directory_path, "OUTCAR")
    if not os.path.isfile(outcar_path):
        return None

    efermi = None
    with open(outcar_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if "Fermi energy:" in line:
                try:
                    efermi = float(line.split("Fermi energy:", 1)[1].split()[0])
                except Exception:
                    pass
            elif "E-fermi" in line:
                try:
                    efermi = float(line.split("E-fermi", 1)[1].replace(":", " ").split()[0])
                except Exception:
                    pass
    return efermi


def _extract_fermi_from_vasprun_root(root):
    """Extract Fermi energy from a parsed vasprun.xml root. Internal helper."""
    fermi_paths = [
        ".//calculation/dos[@comment='kpoints_opt']/i[@name='efermi']",
        ".//dos[@comment='kpoints_opt']/i[@name='efermi']",
        ".//calculation/dos/i[@name='efermi']",
        ".//dos/i[@name='efermi']",
    ]
    for path in fermi_paths:
        element = root.find(path)
        if element is not None and element.text is not None:
            try:
                return float(element.text.strip())
            except Exception:
                pass
    return None


def extract_dos_vasprun(directory_path, spin=1, negate=False, read_eigen=False):
    """
    Extract total DOS from vasprun.xml without using DOSCAR.

    Parameters
    ----------
    directory_path : str
        Directory containing vasprun.xml.
    spin : int
        Spin channel index: 1 (spin up) or 2 (spin down).
    negate : bool
        If True, multiply DOS and integrated DOS by -1.
    read_eigen : bool
        Kept for API compatibility. This extractor returns None for eigen_matrix
        and occu_matrix.

    Returns
    -------
    tuple or None
        (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,
         energy_dos_shift, total_dos_list, integrated_dos_list)
    """
    file_path = os.path.join(directory_path, "vasprun.xml")
    if not os.path.isfile(file_path):
        return None

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception:
        # Incomplete vasprun.xml files are common after interrupted or copied runs.
        return None

    efermi = _extract_fermi_from_vasprun_root(root)
    if efermi is None:
        efermi = _extract_fermi_from_outcar(directory_path)
    if efermi is None:
        try:
            efermi = extract_fermi(directory_path)
        except Exception:
            efermi = None
    if efermi is None:
        return None

    atom_el = root.find(".//atominfo/atoms")
    ions_number = int(atom_el.text.strip()) if atom_el is not None and atom_el.text is not None else None

    kpoints_number = None
    kp_paths = [
        ".//eigenvalues_kpoints_opt[@comment='kpoints_opt']/kpoints/varray[@name='kpointlist']",
        ".//kpoints/varray[@name='kpointlist']",
        ".//varray[@name='kpointlist']",
    ]
    for kp_path in kp_paths:
        kp_varray = root.find(kp_path)
        if kp_varray is not None:
            kpoints_number = len(kp_varray.findall("v"))
            break

    path_candidates = [
        f"./calculation/dos[@comment='kpoints_opt']/total/array/set/set[@comment='spin {spin}']/r",
        f".//dos[@comment='kpoints_opt']/total/array/set/set[@comment='spin {spin}']/r",
        f".//calculation/dos/total/array/set/set[@comment='spin {spin}']/r",
        f".//dos/total/array/set/set[@comment='spin {spin}']/r",
        f".//total/array/set/set[@comment='spin {spin}']/r",
    ]
    r_nodes = []
    for path_dos in path_candidates:
        r_nodes = root.findall(path_dos)
        if r_nodes:
            break

    # Non-spin-polarized vasprun.xml fallback. Only spin 1 can use this path.
    if not r_nodes and int(spin) == 1:
        non_spin_candidates = [
            "./calculation/dos/total/array/set/r",
            ".//dos/total/array/set/r",
        ]
        for path_dos in non_spin_candidates:
            r_nodes = root.findall(path_dos)
            if r_nodes:
                break

    if not r_nodes:
        return None

    try:
        flat = np.fromstring(" ".join(n.text for n in r_nodes if n.text is not None), sep=" ")
        dos = flat.reshape(-1, 3)
    except Exception:
        return None

    energy_dos_shift = dos[:, 0] - efermi
    total_dos_list = dos[:, 1]
    integrated_dos_list = dos[:, 2]

    if negate:
        total_dos_list = -1.0 * total_dos_list
        integrated_dos_list = -1.0 * integrated_dos_list

    eigen_matrix = None
    occu_matrix = None
    if read_eigen:
        print("Warning: read_eigen=True requested, but extract_dos_vasprun returns None for eigen_matrix and occu_matrix.")

    return (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,
            energy_dos_shift, total_dos_list, integrated_dos_list)


def _read_outcar_parameters(outcar_lines):
    """Read basic DOS/electronic parameters from OUTCAR. Internal helper."""
    params = {
        "efermi": None,
        "nions": None,
        "nedos": None,
        "emin": None,
        "emax": None,
        "sigma": None,
    }

    for line in outcar_lines:
        if "NIONS" in line:
            try:
                params["nions"] = int(line.split("NIONS", 1)[1].replace("=", " ").split()[0])
            except Exception:
                pass
        if "NEDOS" in line:
            try:
                params["nedos"] = int(float(line.split("NEDOS", 1)[1].replace("=", " ").split()[0]))
            except Exception:
                pass
        if "SIGMA" in line and "=" in line:
            try:
                params["sigma"] = float(line.split("SIGMA", 1)[1].replace("=", " ").split()[0])
            except Exception:
                pass
        if "EMIN" in line and "EMAX" in line and "energy-range for DOS" in line:
            try:
                parts = line.replace(";", " ").replace("=", " ").split()
                params["emin"] = float(parts[parts.index("EMIN") + 1])
                params["emax"] = float(parts[parts.index("EMAX") + 1])
            except Exception:
                pass
        if "Fermi energy:" in line:
            try:
                params["efermi"] = float(line.split("Fermi energy:", 1)[1].split()[0])
            except Exception:
                pass
        elif "E-fermi" in line:
            try:
                params["efermi"] = float(line.split("E-fermi", 1)[1].replace(":", " ").split()[0])
            except Exception:
                pass

    return params


def _read_outcar_kpoint_weights(outcar_lines, kpoints_number=None):
    """Read k-point weights from OUTCAR and normalize them. Internal helper."""
    weights = []

    # Prefer the integer weights printed by the first IBZKPT block. They carry
    # more significant digits than the rounded normalized weights in the later table.
    for index, line in enumerate(outcar_lines):
        if "Found" in line and "irreducible k-points" in line:
            try:
                found_kpoints = int(line.split()[1])
            except Exception:
                found_kpoints = kpoints_number
            for cursor in range(index + 1, min(index + 20, len(outcar_lines))):
                if "Coordinates" in outcar_lines[cursor] and "Weight" in outcar_lines[cursor]:
                    candidate = []
                    row = cursor + 1
                    while row < len(outcar_lines) and len(candidate) < found_kpoints:
                        parts = outcar_lines[row].split()
                        if len(parts) >= 4:
                            try:
                                float(parts[0]); float(parts[1]); float(parts[2])
                                candidate.append(float(parts[3]))
                            except Exception:
                                pass
                        elif candidate:
                            break
                        row += 1
                    if candidate:
                        weights = candidate
                        break
            if weights:
                break

    # Fallback: read the normalized weights table.
    if not weights:
        for index, line in enumerate(outcar_lines):
            if "k-points in reciprocal lattice and weights" in line:
                candidate = []
                row = index + 1
                while row < len(outcar_lines):
                    parts = outcar_lines[row].split()
                    if len(parts) >= 4:
                        try:
                            float(parts[0]); float(parts[1]); float(parts[2])
                            candidate.append(float(parts[3]))
                        except Exception:
                            if candidate:
                                break
                    elif candidate:
                        break
                    row += 1
                if candidate:
                    weights = candidate
                    break

    if not weights:
        if kpoints_number is None:
            return None
        weights_array = np.ones(int(kpoints_number), dtype=float)
    else:
        weights_array = np.array(weights, dtype=float)

    if kpoints_number is not None and len(weights_array) != int(kpoints_number):
        weights_array = np.ones(int(kpoints_number), dtype=float)

    weights_sum = np.sum(weights_array)
    if weights_sum == 0:
        weights_array = np.ones_like(weights_array, dtype=float)
        weights_sum = np.sum(weights_array)
    return weights_array / weights_sum


def _read_outcar_eigen_matrices(outcar_lines):
    """Read spin-resolved eigenvalue/occupation matrices from OUTCAR. Internal helper."""
    eigen_data = {1: {}, 2: {}}
    occu_data = {1: {}, 2: {}}
    current_spin = None
    current_kpoint = None

    for line in outcar_lines:
        stripped = line.strip()
        if stripped.startswith("spin component"):
            parts = stripped.split()
            try:
                current_spin = int(parts[-1])
            except Exception:
                current_spin = None
            current_kpoint = None
            continue

        if current_spin in eigen_data and stripped.startswith("k-point") and ":" in stripped:
            try:
                current_kpoint = int(stripped.split()[1])
                # Keep the last printed eigenvalue block if OUTCAR contains more than one.
                eigen_data[current_spin][current_kpoint] = []
                occu_data[current_spin][current_kpoint] = []
            except Exception:
                current_kpoint = None
            continue

        if current_spin in eigen_data and current_kpoint is not None:
            parts = stripped.split()
            if len(parts) >= 3:
                try:
                    int(parts[0])
                    eigen = float(parts[1])
                    occu = float(parts[2])
                    eigen_data[current_spin][current_kpoint].append(eigen)
                    occu_data[current_spin][current_kpoint].append(occu)
                except Exception:
                    pass

    matrices = {}
    occupancies = {}
    for spin, data in eigen_data.items():
        if not data:
            matrices[spin] = None
            occupancies[spin] = None
            continue
        kpoint_indices = sorted(data.keys())
        if not kpoint_indices:
            matrices[spin] = None
            occupancies[spin] = None
            continue
        nbands = min(len(data[k]) for k in kpoint_indices if len(data[k]) > 0)
        if nbands == 0:
            matrices[spin] = None
            occupancies[spin] = None
            continue
        eigen_matrix = np.array([data[k][:nbands] for k in kpoint_indices], dtype=float).T
        occu_matrix = np.array([occu_data[spin][k][:nbands] for k in kpoint_indices], dtype=float).T
        matrices[spin] = eigen_matrix
        occupancies[spin] = occu_matrix

    return matrices, occupancies


def extract_dos_outcar(directory_path, spin=1, negate=False, read_eigen=False, gaussian_width=None):
    """
    Reconstruct a spin-resolved DOS from OUTCAR eigenvalues without using DOSCAR.

    This path is a fallback for runs where vasprun.xml is incomplete or does not
    contain the total DOS table. The reconstructed curve uses Gaussian broadening
    with SIGMA from OUTCAR, unless gaussian_width is supplied.

    Parameters
    ----------
    directory_path : str
        Directory containing OUTCAR.
    spin : int
        Spin channel index: 1 (spin up) or 2 (spin down).
    negate : bool
        If True, multiply DOS and integrated DOS by -1.
    read_eigen : bool
        If True, return eigen_matrix and occu_matrix. Otherwise they are still
        computed internally, but returned as None for compatibility with the
        lightweight DOS interface.
    gaussian_width : float or None
        Gaussian broadening width in eV. Default: SIGMA from OUTCAR, or 0.05 eV.

    Returns
    -------
    tuple or None
        (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,
         energy_dos_shift, total_dos_list, integrated_dos_list)
    """
    outcar_path = os.path.join(directory_path, "OUTCAR")
    if not os.path.isfile(outcar_path):
        return None

    with open(outcar_path, "r", encoding="utf-8", errors="ignore") as file:
        outcar_lines = file.readlines()

    params = _read_outcar_parameters(outcar_lines)
    eigen_matrices, occu_matrices = _read_outcar_eigen_matrices(outcar_lines)

    spin = int(spin)
    eigen_matrix_full = eigen_matrices.get(spin)
    occu_matrix_full = occu_matrices.get(spin)
    if eigen_matrix_full is None:
        return None

    efermi = params["efermi"]
    if efermi is None:
        return None

    ions_number = params["nions"]
    kpoints_number = eigen_matrix_full.shape[1]
    weights = _read_outcar_kpoint_weights(outcar_lines, kpoints_number)
    if weights is None:
        weights = np.ones(kpoints_number, dtype=float) / kpoints_number

    sigma = gaussian_width if gaussian_width is not None else params["sigma"]
    if sigma is None or sigma <= 0:
        sigma = 0.05

    nedos = params["nedos"] if params["nedos"] is not None and params["nedos"] > 10 else 4000

    eigen_min = float(np.min(eigen_matrix_full))
    eigen_max = float(np.max(eigen_matrix_full))

    # OUTCAR sometimes reports DOS EMIN/EMAX as user parameters and may contain
    # reversed signs. For reconstructed DOS, the eigenvalue span is the safer range.
    energy_min = eigen_min - 6.0 * sigma
    energy_max = eigen_max + 6.0 * sigma
    if energy_max <= energy_min:
        energy_min = eigen_min - 1.0
        energy_max = eigen_max + 1.0

    energy = np.linspace(energy_min, energy_max, int(nedos))
    total_dos_list = np.zeros_like(energy)
    prefactor = 1.0 / (sigma * np.sqrt(2.0 * np.pi))

    for k_index in range(kpoints_number):
        eigen_column = eigen_matrix_full[:, k_index]
        diff = (energy[:, None] - eigen_column[None, :]) / sigma
        total_dos_list += weights[k_index] * prefactor * np.exp(-0.5 * diff * diff).sum(axis=1)

    if len(energy) > 1:
        dE = energy[1] - energy[0]
    else:
        dE = 1.0
    integrated_dos_list = np.cumsum(total_dos_list) * dE
    energy_dos_shift = energy - efermi

    if negate:
        total_dos_list = -1.0 * total_dos_list
        integrated_dos_list = -1.0 * integrated_dos_list

    eigen_matrix = eigen_matrix_full if read_eigen else None
    occu_matrix = occu_matrix_full if read_eigen else None

    return (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,
            energy_dos_shift, total_dos_list, integrated_dos_list)


def extract_dos_xml_or_outcar(directory_path, spin=1, negate=False, read_eigen=False):
    """
    Extract spin-resolved DOS without using DOSCAR.

    Priority:
    1. Read the VASP total DOS table from vasprun.xml.
    2. If that is unavailable, reconstruct a Gaussian-broadened DOS from OUTCAR.
    """
    dos_data = extract_dos_vasprun(directory_path, spin=spin, negate=negate, read_eigen=read_eigen)
    if dos_data is not None:
        return dos_data

    dos_data = extract_dos_outcar(directory_path, spin=spin, negate=negate, read_eigen=read_eigen)
    if dos_data is not None:
        return dos_data

    print(f"Error: failed to extract DOS from vasprun.xml or OUTCAR in {directory_path}.")
    return None


@lru_cache(maxsize=None)
def extract_dos_xml_or_outcar_cached(directory_path, spin=1, negate=False):
    return extract_dos_xml_or_outcar(directory_path, spin=spin, negate=negate)



def _normalize_spin_dos_mode(spin_mode):
    """Normalize spin-DOS plotting mode. Internal helper."""
    if spin_mode is None:
        return None

    spin_mode = str(spin_mode).lower().strip()

    if spin_mode in ["up", "spin up", "spin-up"]:
        return "up"
    if spin_mode in ["down", "spin down", "spin-down", "negative spin down", "negative spin-down"]:
        return "down"
    if spin_mode in ["total", "spin total", "spin-total"]:
        return "total"

    return None


def create_matters_dos_spin(matters_list):
    """
    Create a list of structured lists for strict spin-polarized DoS plotting.

    Parameters:
    - matters_list: A list of lists, where each inner list should contain:
      [label, directory, spin_mode, line_color, line_style, line_weight, line_alpha].

      The spin_mode field must be written explicitly and is case-insensitive:
      - "spin up", "spin-up", or "up": plot only the spin-up channel;
      - "spin down", "spin-down", or "down": plot only the spin-down channel as negative DoS;
      - "total": plot the summed total DOS, i.e. spin-up + spin-down.

      One matter corresponds to one plotted curve. This function does not expand
      one entry into multiple curves.

    Returns:
    - A list of lists, where each list contains:
      - label: Matter label;
      - spin_mode: up, down, or total;
      - dos_up: Extracted spin-up DoS data, or None when unused;
      - dos_down: Extracted spin-down DoS data, or None when unused;
      - line_color: Color family for plotting;
      - line_style: Line style for plotting;
      - line_weight: Line width for plotting;
      - line_alpha: Line transparency (alpha value) for plotting.
    """
    default_values = {
        "line_color": "Grey",
        "line_style": "solid",
        "line_weight": 1.5,
        "line_alpha": 1.0,
    }

    if matters_list is None:
        print("Error: please provide matters_list for spin-polarized DoS plotting.")
        return []

    if isinstance(matters_list, list) and matters_list and not any(isinstance(i, list) for i in matters_list):
        matters_list = [matters_list[:]]

    matters = []
    for matter_dir in matters_list:
        if len(matter_dir) < 3:
            print("Error: each matter in plot_dos_spin should be [label, directory, spin_mode, ...].")
            continue

        label, directory, spin_mode_raw, *optional_params = matter_dir
        spin_mode = _normalize_spin_dos_mode(spin_mode_raw)
        if spin_mode is None:
            print("Error: spin_mode should be explicitly written as 'spin up', 'spin down', or 'total'.")
            continue

        line_color = get_or_default(optional_params[0] if len(optional_params) > 0 else None, default_values["line_color"])
        line_style = get_or_default(optional_params[1] if len(optional_params) > 1 else None, default_values["line_style"])
        line_weight = get_or_default(optional_params[2] if len(optional_params) > 2 else None, default_values["line_weight"])
        line_alpha = get_or_default(optional_params[3] if len(optional_params) > 3 else None, default_values["line_alpha"])

        dos_up = None
        dos_down = None
        if spin_mode in ["up", "total"]:
            dos_up = extract_dos_xml_or_outcar_cached(directory, spin=1, negate=False)
        if spin_mode in ["down", "total"]:
            dos_down = extract_dos_xml_or_outcar_cached(directory, spin=2, negate=False)

        if spin_mode == "up" and dos_up is None:
            print(f"Error: failed to extract spin-up DoS data for {label}.")
            continue
        if spin_mode == "down" and dos_down is None:
            print(f"Error: failed to extract spin-down DoS data for {label}.")
            continue
        if spin_mode == "total" and (dos_up is None or dos_down is None):
            print(f"Error: failed to extract total spin DoS data for {label}.")
            continue

        matters.append([label, spin_mode, dos_up, dos_down, line_color, line_style, line_weight, line_alpha])
    return matters


def _set_dos_range(axis_name, value):
    """Set x/y range for DoS plotting. Internal helper."""
    if value is None:
        return
    if isinstance(value, (int, float)):
        if axis_name == "x":
            plt.xlim(value*(-1), value)
        elif axis_name == "y":
            plt.ylim(abs(value)*(-1), abs(value))
    elif isinstance(value, (list, tuple, np.ndarray)) and len(value) == 1:
        if axis_name == "x":
            plt.xlim(value[0]*(-1), value[0])
        elif axis_name == "y":
            plt.ylim(abs(value[0])*(-1), abs(value[0]))
    elif isinstance(value, (list, tuple, np.ndarray)) and len(value) > 1:
        if axis_name == "x":
            plt.xlim(value[0], value[-1])
        elif axis_name == "y":
            plt.ylim(value[0], value[-1])


def _spin_dos_label(label_prefix, current_label, spin_mode_label):
    """Create a compact legend label for spin-DOS plotting. Internal helper."""
    spin_mode_label = str(spin_mode_label).lower().strip()
    display_labels = {
        "up": "spin-up",
        "spin up": "spin-up",
        "spin-up": "spin-up",
        "down": "spin-down",
        "spin down": "spin-down",
        "spin-down": "spin-down",
        "total": "total DOS",
    }
    spin_mode_label = display_labels.get(spin_mode_label, spin_mode_label)

    label_prefix = "" if label_prefix is None else str(label_prefix)
    current_label_empty = current_label in [None, ""]
    quantity_label = label_prefix.strip().lower()

    # If the matter label is empty, do not create parentheses.
    if current_label_empty:
        if quantity_label == "":
            return f"{spin_mode_label}"
        if quantity_label == "total":
            if spin_mode_label == "total DOS":
                return "total DOS"
            return f"{spin_mode_label} DOS"
        if quantity_label == "integrated":
            if spin_mode_label == "total DOS":
                return "integrated total DOS"
            return f"integrated {spin_mode_label} DOS"
        return f"{label_prefix}{spin_mode_label}"

    # If the matter label exists, keep the spin channel as a parenthetical qualifier.
    if label_prefix == "":
        return f"{current_label} ({spin_mode_label})"
    return f"{label_prefix}{current_label} ({spin_mode_label})"


# Universal spin-polarized DoS Plotting

def plot_dos_spin(title, matters_list=None, x_range=None, y_lim=None, dos_quantity=None):
    """
    Plot spin-polarized DoS from one or more VASP calculations.

    Parameters:
    - title: Figure title;
    - matters_list: A list of lists. Each inner list should contain:
      [label, directory, spin_mode, line_color, line_style, line_weight, line_alpha].

      The spin_mode field must be written explicitly and is case-insensitive:
      - "spin up", "spin-up", or "up": plot only the spin-up channel;
      - "spin down", "spin-down", or "down": plot only the spin-down channel as negative DoS;
      - "total": plot the summed total DOS, i.e. spin-up + spin-down.

      One matter corresponds to one plotted curve. This function does not expand
      one entry into multiple curves.

    - x_range: Energy range. Use a number for symmetric range, or [left, right] for asymmetric range;
    - y_lim: DoS range. Use a number for symmetric range, or [bottom, top] for asymmetric range;
    - dos_quantity: "Total", "Integrated", or "All". The default is "Total".

    Example:
    matter = [["Sample", "DoS/FM", "spin up", "Blue"],
              ["Sample", "DoS/FM", "spin down", "Orange"],
              ["Sample", "DoS/FM", "total", "Purple"]]
    plot_dos_spin("Spin-polarized DoS", matter, [-8, 4], [-8, 8], "Total")
    """
    help_info = """Usage: plot_dos_spin
Use the same argument order as plot_dos: title, matters_list, x_range, y_lim, dos_quantity.
Each matter should be [label, directory, spin_mode, line_color, line_style, line_weight, line_alpha].
spin_mode is case-insensitive. Use 'spin up', 'spin-up', or 'up'; 'spin down', 'spin-down', or 'down'; or 'total'.
One matter corresponds to one curve. 'total' means total DOS, i.e. spin-up plus spin-down.
This function reads vasprun.xml first, then OUTCAR; DOSCAR is not required.
"""

    if title in ["help", "Help"]:
        print(help_info)
        return

    if dos_quantity is None:
        dos_quantity = "Total"

    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]; plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    fermi_color = color_sampling("Grey")
    matters = create_matters_dos_spin(matters_list)
    if not matters:
        return

    if dos_quantity in ["Total", "total"]:
        dos_indices = [(6, "")]
    elif dos_quantity in ["Integrated", "integrated"]:
        dos_indices = [(7, "Integrated ")]
    elif dos_quantity in ["All", "all"]:
        dos_indices = [(6, "Total "), (7, "Integrated ")]
    else:
        print("Error: dos_quantity should be 'Total', 'Integrated', or 'All'.")
        return

    for _, matter in enumerate(matters):
        current_label = matter[0]
        spin_mode = matter[1]
        dos_up = matter[2]
        dos_down = matter[3]
        line_color = matter[4]
        line_style = matter[5]
        line_weight = matter[6]
        line_alpha = matter[7]
        dos_color = color_sampling(line_color)
        curve_color = dos_color[1]

        for index, label_prefix in dos_indices:
            if spin_mode == "up" and dos_up is not None:
                up_label = _spin_dos_label(label_prefix, current_label, "up")
                plt.plot(dos_up[5], dos_up[index], c=curve_color, linestyle=line_style, lw=line_weight, alpha=line_alpha, label=up_label, zorder=3)
            elif spin_mode == "down" and dos_down is not None:
                down_label = _spin_dos_label(label_prefix, current_label, "down")
                plt.plot(dos_down[5], -1.0 * dos_down[index], c=curve_color, linestyle=line_style, lw=line_weight, alpha=line_alpha, label=down_label, zorder=2)
            elif spin_mode == "total" and dos_up is not None and dos_down is not None:
                total_label = _spin_dos_label(label_prefix, current_label, "total")
                total_dos = dos_up[index] + dos_down[index]
                plt.plot(dos_up[5], total_dos, c=curve_color, linestyle=line_style, lw=line_weight, alpha=line_alpha, label=total_label, zorder=3)

    plt.axvline(x=0, linestyle="--", c=fermi_color[1], alpha=0.80, zorder=1)
    plt.axhline(y=0, linestyle=":", c=fermi_color[1], alpha=0.80, zorder=1)

    plt.title(f"{title}")
    plt.ylabel(r"Density of States (states/eV)"); plt.xlabel(r"Energy (eV)")

    _set_dos_range("x", x_range)
    _set_dos_range("y", y_lim)

    plt.legend(loc="best")
    plt.tight_layout()
