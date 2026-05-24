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
        matters.append([label, dos_data, line_color, line_style, line_weight, line_alpha])
    return matters

# Universal DoS Plotting
def plot_dos(title, matters_list = None, x_range = None, y_lim = None, dos_type = None):
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
    if all(term is not None for term in [x_range, y_lim]):
        # Data plotting
        if dos_type in ["All", "all"]:
            for _, matter in enumerate(matters):
                # Labels
                current_label = matter[0]
                plt.plot(matter[1][5], matter[1][6], c=color_sampling(matter[2])[1], linestyle=matter[3], lw=matter[4], alpha=matter[5], label=f"{current_label}", zorder=3)
                plt.plot(matter[1][5], matter[1][7], c=color_sampling(matter[2])[2], linestyle=matter[3], lw=matter[4], alpha=matter[5], label=f"{current_label}", zorder=2)
                efermi = matter[1][0]
        if dos_type in ["Total", "total"]:
            for _, matter in enumerate(matters):
                # Labels
                current_label = matter[0]
                plt.plot(matter[1][5], matter[1][6], c=color_sampling(matter[2])[1], linestyle=matter[3], lw=matter[4], alpha=matter[5], label=f"{current_label}", zorder=2)
                efermi = matter[1][0]
        if dos_type in ["Integrated", "integrated"]:
            for _, matter in enumerate(matters):
                # Labels
                current_label = matter[0]
                plt.plot(matter[1][5], matter[1][7], c=color_sampling(matter[2])[2], linestyle=matter[3], lw=matter[4], alpha=matter[5], label=f"{current_label}", zorder=2)
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
