#### Declarations of process functions for PDoS with vectorized programming
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915, W0612

# Necessary packages invoking
import xml.etree.ElementTree as ET
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from vmatplot.commons import extract_fermi, get_or_default, get_elements
from vmatplot.output_settings import color_sampling, canvas_setting

import matplotlib as mpl

mpl.rcParams["lines.solid_capstyle"] = "round"
mpl.rcParams["lines.dash_capstyle"]  = "round"
mpl.rcParams["lines.solid_joinstyle"] = "round"
mpl.rcParams["lines.dash_joinstyle"]  = "round"

def cal_type_pdos(directory_path):
    kpoints_file_path = os.path.join(directory_path, "KPOINTS")
    kpoints_opt_path = os.path.join(directory_path, "KPOINTS_OPT")
    if os.path.exists(kpoints_opt_path):
        return "GGA-PBE"
    elif os.path.exists(kpoints_file_path):
        return "HSE06"

def count_pdos_atoms_vasp(directory_path):
    """
    Get the total number of atoms (ions) from the vasprun.xml file in the specified folder.
    This version reads from <atominfo>/<atoms> tag, which is more reliable.
    """
    file_path = os.path.join(directory_path, "vasprun.xml")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file vasprun.xml does not exist in the directory: {directory_path}")

    tree = ET.parse(file_path)
    root = tree.getroot()
    # Find the <atoms> tag inside <atominfo>
    atoms_tag = root.find(".//atominfo/atoms")
    if atoms_tag is None or atoms_tag.text is None:
        raise ValueError("Failed to locate the total atom count in <atominfo>.")
    try:
        atom_count = int(atoms_tag.text)
    except ValueError:
        raise ValueError("The value inside <atoms> is not an integer.")
    return atom_count

# Extract Kpoints number
def extract_kpoints_number(directory_path):
    ## Construct the full path to the vasprun.xml file
    file_path = os.path.join(directory_path, "vasprun.xml")
    tree = ET.parse(file_path)
    root = tree.getroot()
    kpoints_file_path = os.path.join(directory_path, "KPOINTS")
    kpoints_opt_path = os.path.join(directory_path, "KPOINTS_OPT")
    ## Extract the number of kpoints
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        kpointlist = root.find(".//eigenvalues_kpoints_opt[@comment='kpoints_opt']/kpoints/varray[@name='kpointlist']")
        kpointlist_concatenated_text = " ".join([kpointlist.text for kpointlist in kpointlist.findall("v")])
        kpointlist_array = np.fromstring(kpointlist_concatenated_text, sep=" ")
        kpointlist_matrix = kpointlist_array.reshape(-1, 3)
        kpoints_number = kpointlist_matrix.shape[0]
    # GGA-PBE algorithms
    elif os.path.exists(kpoints_file_path):
        kpointlist = root.find(".//varray[@name='kpointlist']")
        kpointlist_concatenated_text = " ".join([kpointlist.text for kpointlist in kpointlist.findall("v")])
        kpointlist_array = np.fromstring(kpointlist_concatenated_text, sep=" ")
        kpointlist_matrix = kpointlist_array.reshape(-1, 3)
        kpoints_number = kpointlist_matrix.shape[0]
    return kpoints_number

## Extract eigen, occupancy number
def extract_eigen_occupancy(directory_path):
    ## Construct the full path to the vasprun.xml file
    file_path = os.path.join(directory_path, "vasprun.xml")
    tree = ET.parse(file_path)
    root = tree.getroot()
    kpoints_file_path = os.path.join(directory_path, "KPOINTS")
    kpoints_opt_path = os.path.join(directory_path, "KPOINTS_OPT")
    kpoints_number = extract_kpoints_number(directory_path)
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        for kpoints_index in range(1, kpoints_number+1):
            xpath_expr = f"./calculation/projected_kpoints_opt/eigenvalues/array/set/set[@comment='spin 1']/set[@comment='kpoint {kpoints_index}']"
            eigen_column = np.empty(0)
            occu_column  = np.empty(0)
            kpoint_set = root.find(xpath_expr)
            for eigen_occ_element in kpoint_set:
                eigen_values = list(map(float, eigen_occ_element.text.split()))
                eigen_column = np.append(eigen_column, eigen_values[0])
                occu_column = np.append(occu_column, eigen_values[1])
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
                eigen_values = list(map(float, eigen_occ_element.text.split()))
                eigen_column = np.append(eigen_column, eigen_values[0])
                occu_column = np.append(occu_column, eigen_values[1])
            if kpoints_index == 1 :
                eigen_matrix = eigen_column.reshape(-1, 1)
                occu_matrix = occu_column.reshape(-1, 1)
            else:
                eigen_matrix = np.hstack((eigen_matrix,eigen_column.reshape(-1, 1)))
                occu_matrix  = np.hstack((occu_matrix, occu_column.reshape(-1, 1)))
    return (eigen_matrix, occu_matrix)

# Extract DoS
def read_total_dos_from_doscar(directory_path):
    doscar_path = os.path.join(directory_path, "DOSCAR")
    if not os.path.isfile(doscar_path):
        raise FileNotFoundError(f"Error: DOSCAR not found in {directory_path}")
    with open(doscar_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(5):
            if not f.readline():
                raise ValueError("Error: DOSCAR is too short.")
        header = None
        for _ in range(8000):
            line = f.readline()
            if not line: break
            toks = line.split()
            if len(toks) < 4: continue
            try:
                emax = float(toks[0])
                emin = float(toks[1])
                nedos = int(float(toks[2]))
                efermi = float(toks[3])
                if nedos > 10 and emax > emin and abs(efermi) < 1.0e5:
                    header = (nedos, efermi)
                    break
            except Exception: continue
        if header is None:
            raise ValueError("Error: Failed to locate TOTAL DOS header line in DOSCAR.")
        nedos, efermi = header
        first = f.readline()
        if not first:
            raise ValueError("Error: DOSCAR ended while reading TOTAL DOS.")
        ncols = len(first.split())
        lines = [first]
        for _ in range(nedos - 1):
            line = f.readline()
            if not line:
                break
            lines.append(line)
    flat = np.fromstring("".join(lines), sep=" ")
    if flat.size % ncols != 0:
        data = []
        for row in lines:
            parts = row.split()
            if len(parts) == ncols:
                data.append([float(x) for x in parts])
        arr = np.array(data, dtype=float)
    else:  arr = flat.reshape(-1, ncols)
    energy_shift = arr[:, 0] - efermi
    dos_total = arr[:, 1] if ncols == 3 else (arr[:, 1] + arr[:, 2])
    return efermi, energy_shift, dos_total

# Extract energy list
def extract_energy_list(directory_path):
    ## Construct the full path to the vasprun.xml file
    file_path = os.path.join(directory_path, "vasprun.xml")
    tree = ET.parse(file_path)
    root = tree.getroot()
    kpoints_file_path = os.path.join(directory_path, "KPOINTS")
    kpoints_opt_path = os.path.join(directory_path, "KPOINTS_OPT")
    ## Initialization
    energy_dos_list     = np.array([])
    efermi = extract_fermi(directory_path)
    if os.path.exists(kpoints_opt_path):
        path_dos = "./calculation/dos[@comment='kpoints_opt']/total/array/set/set[@comment='spin 1']/r"
    elif os.path.exists(kpoints_file_path):
        path_dos = ".//total/array/set/set[@comment='spin 1']/r"
    for element_dos in root.findall(path_dos):
        values_dos = list(map(float, element_dos.text.split()))
        energy_dos_list = np.append(energy_dos_list, values_dos[0])
    shift = efermi
    return energy_dos_list

def extract_energy_shift(directory_path):
    energy_dos_list = extract_energy_list(directory_path)
    shift = extract_fermi(directory_path)
    energy_dos_shift = energy_dos_list - shift
    return energy_dos_shift

# Total PDoS: univseral elements and layers
def extract_pdos(directory_path, spin=1, negate=False, read_eigen=False):
    """
    Extract total DOS and orbital-resolved PDOS from VASP DOSCAR.
    Requirements
    - DOSCAR must contain per-ion, per-orbital projections (typically LORBIT = 11).
    - eigen_matrix and occu_matrix are not available from DOSCAR in this layout, so they are returned as None.
    Parameters
    directory_path : str
        Directory containing DOSCAR (and preferably CONTCAR/POSCAR).
    spin : int
        1 for spin-up, 2 for spin-down (only used if DOSCAR is spin-polarized).
    negate : bool
        If True, multiply DOS/PDOS and integrated DOS by -1 (useful for plotting spin-down negative).
    read_eigen : bool
        Kept for API compatibility; not supported for DOSCAR parsing.
    Returns
    tuple
        (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,             # 0 ~ 4
         energy_dos_shift, total_dos_list, integrated_pdos_list,                     # 5 ~ 7
         energy_pdos_shift, s_pdos_sum, p_y_pdos_sum, p_z_pdos_sum, p_x_pdos_sum,    # 8 ~ 12
         d_xy_pdos_sum, d_yz_pdos_sum, d_z2_pdos_sum, d_zx_pdos_sum,                 # 13 ~ 16
         x2_y2_pdos_sum)                                                            # 17
    """
    doscar_path = os.path.join(directory_path, "DOSCAR")
    if not os.path.isfile(doscar_path):
        print(f"Error: The file DOSCAR does not exist in the directory {directory_path}.")
        return
    spin = int(spin) if int(spin) in (1, 2) else 1
    ions_number = None
    for poscar_name in ("CONTCAR", "POSCAR"):
        poscar_path = os.path.join(directory_path, poscar_name)
        if not os.path.isfile(poscar_path): continue
        try:
            with open(poscar_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [next(f) for _ in range(8)]
            tokens6 = lines[5].split() if len(lines) > 5 else []
            tokens7 = lines[6].split() if len(lines) > 6 else []
            def _is_intlike(tok):
                try:
                    int(tok)
                    return True
                except Exception:
                    try:
                        return float(tok).is_integer()
                    except Exception:
                        return False
            if tokens6 and all(_is_intlike(t) for t in tokens6):
                ions_number = int(sum(int(float(t)) for t in tokens6))
                break
            if tokens6 and tokens7 and (not all(_is_intlike(t) for t in tokens6)) and all(_is_intlike(t) for t in tokens7):
                ions_number = int(sum(int(float(t)) for t in tokens7))
                break
        except Exception: continue
    if ions_number is None:
        print("Error: Failed to determine ions_number (need CONTCAR/POSCAR to parse PDOS blocks reliably).")
        return
    kpoints_number = None
    ibzkpt_path = os.path.join(directory_path, "IBZKPT")
    if os.path.isfile(ibzkpt_path):
        try:
            with open(ibzkpt_path, "r", encoding="utf-8", errors="ignore") as f:
                _ = f.readline()
                line2 = f.readline()
            kpoints_number = int(line2.split()[0])
        except Exception:
            kpoints_number = None
    def _seek_dos_header(fh, max_lines=8000):
        for _ in range(max_lines):
            line = fh.readline()
            if not line:
                return None
            toks = line.split()
            if len(toks) < 4: continue
            try:
                emax = float(toks[0])
                emin = float(toks[1])
                nedos = int(float(toks[2]))
                efermi = float(toks[3])
                if nedos > 10 and emax > emin and abs(efermi) < 1.0e5:
                    return (emax, emin, nedos, efermi)
            except Exception:
                continue
        return None
    def _read_block(fh, nlines):
        first = fh.readline()
        if not first:
            return None, None
        ncols = len(first.split())
        if ncols < 3:
            return None, None
        lines = [first]
        for _ in range(nlines - 1):
            line = fh.readline()
            if not line:
                break
            lines.append(line)
        flat = np.fromstring("".join(lines), sep=" ")
        if flat.size % ncols != 0:
            data = []
            for row in lines:
                parts = row.split()
                if len(parts) == ncols:
                    data.append([float(x) for x in parts])
            arr = np.array(data, dtype=float)
        else:
            arr = flat.reshape(-1, ncols)
        return arr, ncols
    with open(doscar_path, "r", encoding="utf-8", errors="ignore") as fh:
        hdr = _seek_dos_header(fh)
        if hdr is None:
            print("Error: Failed to locate TOTAL DOS header line in DOSCAR.")
            return
        _, _, nedos, efermi = hdr
        total_arr, total_ncols = _read_block(fh, nedos)
        if total_arr is None:
            print("Error: Failed to read TOTAL DOS block from DOSCAR.")
            return
        energy_total = total_arr[:, 0]
        if total_ncols >= 5:
            if spin == 2:
                total_dos_list = total_arr[:, 2]
                integrated_pdos_list = total_arr[:, 4]
            else:
                total_dos_list = total_arr[:, 1]
                integrated_pdos_list = total_arr[:, 3]
        else:
            total_dos_list = total_arr[:, 1]
            integrated_pdos_list = total_arr[:, 2]
        energy_dos_shift = energy_total - efermi
        s_sum = py_sum = pz_sum = px_sum = None
        dxy_sum = dyz_sum = dz2_sum = dzx_sum = x2y2_sum = None
        energy_pdos = None
        pdos_ncols_ref = None
        for ion_i in range(ions_number):
            ihdr = _seek_dos_header(fh)
            if ihdr is None:
                print(f"Error: Failed to locate PDOS header for ion {ion_i+1} in DOSCAR.")
                return
            ion_arr, ion_ncols = _read_block(fh, nedos)
            if ion_arr is None:
                print(f"Error: Failed to read PDOS block for ion {ion_i+1} from DOSCAR.")
                return
            if pdos_ncols_ref is None:
                pdos_ncols_ref = ion_ncols
                energy_pdos = ion_arr[:, 0]
                ngrid = ion_arr.shape[0]
                s_sum    = np.zeros(ngrid, dtype=float)
                py_sum   = np.zeros(ngrid, dtype=float)
                pz_sum   = np.zeros(ngrid, dtype=float)
                px_sum   = np.zeros(ngrid, dtype=float)
                dxy_sum  = np.zeros(ngrid, dtype=float)
                dyz_sum  = np.zeros(ngrid, dtype=float)
                dz2_sum  = np.zeros(ngrid, dtype=float)
                dzx_sum  = np.zeros(ngrid, dtype=float)
                x2y2_sum = np.zeros(ngrid, dtype=float)
            else:
                if ion_ncols != pdos_ncols_ref or ion_arr.shape[0] != s_sum.shape[0]:
                    print("Error: Inconsistent PDOS block format/grid among ions in DOSCAR.")
                    return
            if ion_ncols in (10, 11):
                s_sum    += ion_arr[:, 1]
                py_sum   += ion_arr[:, 2]
                pz_sum   += ion_arr[:, 3]
                px_sum   += ion_arr[:, 4]
                dxy_sum  += ion_arr[:, 5]
                dyz_sum  += ion_arr[:, 6]
                dz2_sum  += ion_arr[:, 7]
                dzx_sum  += ion_arr[:, 8]
                x2y2_sum += ion_arr[:, 9]
            elif ion_ncols in (19, 21):
                if spin == 2:
                    s_sum    += ion_arr[:, 2]
                    py_sum   += ion_arr[:, 4]
                    pz_sum   += ion_arr[:, 6]
                    px_sum   += ion_arr[:, 8]
                    dxy_sum  += ion_arr[:, 10]
                    dyz_sum  += ion_arr[:, 12]
                    dz2_sum  += ion_arr[:, 14]
                    dzx_sum  += ion_arr[:, 16]
                    x2y2_sum += ion_arr[:, 18]
                else:
                    s_sum    += ion_arr[:, 1]
                    py_sum   += ion_arr[:, 3]
                    pz_sum   += ion_arr[:, 5]
                    px_sum   += ion_arr[:, 7]
                    dxy_sum  += ion_arr[:, 9]
                    dyz_sum  += ion_arr[:, 11]
                    dz2_sum  += ion_arr[:, 13]
                    dzx_sum  += ion_arr[:, 15]
                    x2y2_sum += ion_arr[:, 17]
            else:
                print(
                    f"Error: PDOS block has {ion_ncols} columns. Expected LORBIT=11 formats: "
                    "10/11 (non-spin) or 19/21 (spin-polarized). Please set LORBIT=11 and rerun."
                )
                return
    energy_pdos_shift = energy_pdos - efermi
    if negate:
        total_dos_list = -1.0 * total_dos_list
        integrated_pdos_list = -1.0 * integrated_pdos_list
        s_sum = -1.0 * s_sum
        py_sum = -1.0 * py_sum
        pz_sum = -1.0 * pz_sum
        px_sum = -1.0 * px_sum
        dxy_sum = -1.0 * dxy_sum
        dyz_sum = -1.0 * dyz_sum
        dz2_sum = -1.0 * dz2_sum
        dzx_sum = -1.0 * dzx_sum
        x2y2_sum = -1.0 * x2y2_sum
    eigen_matrix = None
    occu_matrix = None
    if read_eigen:
        print("Warning: read_eigen=True requested, but DOSCAR-based PDOS extractor cannot provide eigen/occu matrices. Returning None.")
    return (
        efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,
        energy_dos_shift, total_dos_list, integrated_pdos_list,
        energy_pdos_shift, s_sum, py_sum, pz_sum, px_sum,
        dxy_sum, dyz_sum, dz2_sum, dzx_sum,
        x2y2_sum
    )

def extract_pdos_backup(directory_path):
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

    ## Extract energy, total DoS, and integrated DoS
    # lists initialization
    energy_dos_list     = np.array([])
    total_dos_list      = np.array([])
    integrated_pdos_list = np.array([])

    if os.path.exists(kpoints_opt_path):
        path_dos_spin_1 = "./calculation/dos[@comment='kpoints_opt']/total/array/set/set[@comment='spin 1']/r"
        path_dos_spin_2 = "./calculation/dos[@comment='kpoints_opt']/total/array/set/set[@comment='spin 2']/r"
    elif os.path.exists(kpoints_file_path):
        path_dos_spin_1 = ".//total/array/set/set[@comment='spin 1']/r"
        path_dos_spin_2 = ".//total/array/set/set[@comment='spin 2']/r"

    spin2_exists = root.find(path_dos_spin_2) is not None

    for element_dos in root.findall(path_dos_spin_1):
        dos_values = list(map(float, element_dos.text.split()))
        energy_var = dos_values[0]
        energy_dos_list = np.append(energy_dos_list, energy_var)
        total_dos_var = dos_values[1]
        total_dos_list = np.append(total_dos_list, total_dos_var)
        integrated_dos_var = dos_values[2]
        integrated_pdos_list = np.append(integrated_pdos_list, integrated_dos_var)
    shift = efermi
    energy_dos_shift = energy_dos_list - shift

    ## Extract energy, s-PDoS, p_y-PDoS, p_z-PDoS, p_x-PDoS, d_xy-PDoS, d_yz-PDoS, d_z2-PDoS, d_zx-PDoS, x2-y2-PDoS
    # Matrices initialization
    for ions_index in range(1, ions_number + 1):
        path_ions = f".//set[@comment='ion {ions_index}']/set[@comment='spin 1']/r"
        # Columns initialization
        energy_pdos_column  = np.empty(0)
        s_pdos_column       = np.empty(0)
        p_y_pdos_column     = np.empty(0)
        p_z_pdos_column     = np.empty(0)
        p_x_pdos_column     = np.empty(0)
        d_xy_pdos_column    = np.empty(0)
        d_yz_pdos_column    = np.empty(0)
        d_z2_pdos_column    = np.empty(0)
        d_zx_pdos_column    = np.empty(0)
        x2_y2_pdos_column   = np.empty(0)
        for pdos_element in root.findall(path_ions):
            pdos_values = list(map(float, pdos_element.text.split()))
            # Columns of energy
            energy_pdos_column = np.append(energy_pdos_column, pdos_values[0])
            # Columns of s-PDoS
            s_pdos_column = np.append(s_pdos_column, pdos_values[1])
            # Columns of p_y-PDoS
            p_y_pdos_column = np.append(p_y_pdos_column, pdos_values[2])
            # Columns of p_z-PDoS
            p_z_pdos_column = np.append(p_z_pdos_column, pdos_values[3])
            # Columns of p_x-PDoS
            p_x_pdos_column = np.append(p_x_pdos_column, pdos_values[4])
            # Columns of d_xy-PDoS
            d_xy_pdos_column = np.append(d_xy_pdos_column, pdos_values[5])
            # Columns of d_yz-PDoS
            d_yz_pdos_column = np.append(d_yz_pdos_column, pdos_values[6])
            # Columns of d_z2-PDoS
            d_z2_pdos_column = np.append(d_z2_pdos_column, pdos_values[7])
            # Columns of d_zx-PDoS
            d_zx_pdos_column = np.append(d_zx_pdos_column, pdos_values[8])
            # Columns of x2-y2-PDoS
            x2_y2_pdos_column = np.append(x2_y2_pdos_column, pdos_values[9])
        if ions_index == 1:
            energy_pdos_matrix = energy_pdos_column.reshape(-1, 1)
            s_pdos_matrix = s_pdos_column.reshape(-1, 1)
            p_y_pdos_matrix = p_y_pdos_column.reshape(-1, 1)
            p_z_pdos_matrix = p_z_pdos_column.reshape(-1, 1)
            p_x_pdos_matrix = p_x_pdos_column.reshape(-1, 1)
            d_xy_pdos_matrix = d_xy_pdos_column.reshape(-1, 1)
            d_yz_pdos_matrix = d_yz_pdos_column.reshape(-1, 1)
            d_z2_pdos_matrix = d_z2_pdos_column.reshape(-1, 1)
            d_zx_pdos_matrix = d_zx_pdos_column.reshape(-1, 1)
            x2_y2_pdos_matrix = x2_y2_pdos_column.reshape(-1, 1)
        else:
            energy_pdos_matrix = np.hstack((energy_pdos_matrix, energy_pdos_column.reshape(-1, 1)))
            s_pdos_matrix = np.hstack((s_pdos_matrix, s_pdos_column.reshape(-1, 1)))
            p_y_pdos_matrix = np.hstack((p_y_pdos_matrix, p_y_pdos_column.reshape(-1, 1)))
            p_z_pdos_matrix = np.hstack((p_z_pdos_matrix, p_z_pdos_column.reshape(-1, 1)))
            p_x_pdos_matrix = np.hstack((p_x_pdos_matrix, p_x_pdos_column.reshape(-1, 1)))
            d_xy_pdos_matrix = np.hstack((d_xy_pdos_matrix, d_xy_pdos_column.reshape(-1, 1)))
            d_yz_pdos_matrix = np.hstack((d_yz_pdos_matrix, d_yz_pdos_column.reshape(-1, 1)))
            d_z2_pdos_matrix = np.hstack((d_z2_pdos_matrix, d_z2_pdos_column.reshape(-1, 1)))
            d_zx_pdos_matrix = np.hstack((d_zx_pdos_matrix, d_zx_pdos_column.reshape(-1, 1)))
            x2_y2_pdos_matrix = np.hstack((x2_y2_pdos_matrix, x2_y2_pdos_column.reshape(-1, 1)))
    energy_pdos_sum = energy_pdos_matrix[:,0]
    s_pdos_sum = np.sum(s_pdos_matrix, axis=1)
    p_y_pdos_sum = np.sum(p_y_pdos_matrix, axis=1)
    p_z_pdos_sum = np.sum(p_z_pdos_matrix, axis=1)
    p_x_pdos_sum = np.sum(p_x_pdos_matrix, axis=1)
    d_xy_pdos_sum = np.sum(d_xy_pdos_matrix, axis=1)
    d_yz_pdos_sum = np.sum(d_yz_pdos_matrix, axis=1)
    d_z2_pdos_sum = np.sum(d_z2_pdos_matrix, axis=1)
    d_zx_pdos_sum = np.sum(d_zx_pdos_matrix, axis=1)
    x2_y2_pdos_sum = np.sum(x2_y2_pdos_matrix, axis=1)
    energy_pdos_shift = energy_pdos_sum - shift
    return (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,             # 0 ~ 4
            energy_dos_shift, total_dos_list, integrated_pdos_list,                      # 5 ~ 7
            energy_pdos_shift, s_pdos_sum, p_y_pdos_sum, p_z_pdos_sum, p_x_pdos_sum,    # 8 ~ 12
            d_xy_pdos_sum, d_yz_pdos_sum, d_z2_pdos_sum, d_zx_pdos_sum,                 # 13 ~ 16
            x2_y2_pdos_sum)

# Extract PDoS for elements
def extract_element_pdos(directory_path, element):
    """
    Extract element-resolved orbital PDOS from DOSCAR using element indexing from CONTCAR/POSCAR.

    Requirements
    1. DOSCAR must contain per-ion, per-orbital projections (typically LORBIT = 11).
    2. CONTCAR or POSCAR must be VASP 5/6 style (element symbols + element counts).

    Returns (same layout as your original function)
    (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,             # 0 ~ 4
     energy_dos_shift,                                                           # 5
     total_pdos_list, integrated_pdos_list,                                       # 6 ~ 7
     energy_pdos_shift, s_pdos_sum, p_y_pdos_sum, p_z_pdos_sum, p_x_pdos_sum,    # 8 ~ 12
     d_xy_pdos_sum, d_yz_pdos_sum, d_z2_pdos_sum, d_zx_pdos_sum,                 # 13 ~ 16
     x2_y2_pdos_sum)                                                             # 17
    """
    doscar_path = os.path.join(directory_path, "DOSCAR")
    if not os.path.isfile(doscar_path):
        print(f"Error: The file DOSCAR does not exist in the directory {directory_path}.")
        return

    poscar_path = os.path.join(directory_path, "CONTCAR")
    if not os.path.isfile(poscar_path):
        poscar_path = os.path.join(directory_path, "POSCAR")
    if not os.path.isfile(poscar_path):
        print("Error: CONTCAR/POSCAR not found. Element indexing requires POSCAR-like file.")
        return

    try:
        with open(poscar_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [next(f) for _ in range(8)]
    except Exception:
        print("Error: Failed to read CONTCAR/POSCAR.")
        return

    tokens6 = lines[5].split() if len(lines) > 5 else []
    tokens7 = lines[6].split() if len(lines) > 6 else []

    def _is_intlike(tok):
        try:
            int(tok)
            return True
        except Exception:
            try:
                return float(tok).is_integer()
            except Exception:
                return False

    if not tokens6 or not tokens7:
        print("Error: Invalid POSCAR/CONTCAR format (missing element symbols or counts).")
        return

    if all(_is_intlike(t) for t in tokens6):
        print("Error: VASP4-style POSCAR detected (no element symbols). Cannot map `element` to indices.")
        return

    if not all(_is_intlike(t) for t in tokens7):
        print("Error: Failed to parse element counts from POSCAR/CONTCAR.")
        return

    element_symbols = tokens6
    element_counts = [int(float(t)) for t in tokens7]
    if len(element_symbols) != len(element_counts):
        print("Error: Mismatch between element symbols and element counts in POSCAR/CONTCAR.")
        return

    ions_number = int(sum(element_counts))

    element_map = {}
    running = 1
    for sym, cnt in zip(element_symbols, element_counts):
        start_i = running
        end_i = running + cnt - 1
        element_map[sym] = (start_i, end_i)
        running = end_i + 1

    if element not in element_map:
        print(f"Error: Element '{element}' not found in POSCAR/CONTCAR symbols: {element_symbols}")
        return

    index_start, index_end = element_map[element]

    kpoints_number = None
    ibzkpt_path = os.path.join(directory_path, "IBZKPT")
    if os.path.isfile(ibzkpt_path):
        try:
            with open(ibzkpt_path, "r", encoding="utf-8", errors="ignore") as f:
                _ = f.readline()
                line2 = f.readline()
            kpoints_number = int(line2.split()[0])
        except Exception:
            kpoints_number = None

    eigen_matrix = None
    occu_matrix = None

    with open(doscar_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(5):
            if not f.readline():
                print("Error: DOSCAR is too short.")
                return

        header_line = f.readline()
        if not header_line:
            print("Error: DOSCAR ended unexpectedly.")
            return

        header_tokens = header_line.split()
        if len(header_tokens) < 4:
            found = False
            for _ in range(8000):
                line = f.readline()
                if not line:
                    break
                toks = line.split()
                if len(toks) < 4:
                    continue
                try:
                    _emax = float(toks[0])
                    _emin = float(toks[1])
                    _nedos = int(float(toks[2]))
                    _efermi = float(toks[3])
                    if _nedos > 10 and _emax > _emin and abs(_efermi) < 1.0e5:
                        header_tokens = toks
                        found = True
                        break
                except Exception:
                    continue
            if not found:
                print("Error: Failed to locate TOTAL DOS header line in DOSCAR.")
                return

        try:
            nedos = int(float(header_tokens[2]))
            efermi = float(header_tokens[3])
        except Exception:
            print("Error: Failed to parse TOTAL DOS header line in DOSCAR.")
            return

        first_row = f.readline()
        if not first_row:
            print("Error: DOSCAR ended unexpectedly while reading TOTAL DOS block.")
            return

        ncols_total = len(first_row.split())
        if ncols_total < 3:
            print("Error: Unexpected TOTAL DOS row format in DOSCAR.")
            return

        total_lines = [first_row]
        for _ in range(nedos - 1):
            line = f.readline()
            if not line:
                break
            total_lines.append(line)

        flat = np.fromstring("".join(total_lines), sep=" ")
        if flat.size % ncols_total != 0:
            data = []
            for row in total_lines:
                parts = row.split()
                if len(parts) == ncols_total:
                    data.append([float(x) for x in parts])
            total_arr = np.array(data, dtype=float)
        else:
            total_arr = flat.reshape(-1, ncols_total)

        energy_total = total_arr[:, 0]
        energy_dos_shift = energy_total - efermi

        s_pdos_sum = np.zeros_like(energy_dos_shift, dtype=float)
        p_y_pdos_sum = np.zeros_like(energy_dos_shift, dtype=float)
        p_z_pdos_sum = np.zeros_like(energy_dos_shift, dtype=float)
        p_x_pdos_sum = np.zeros_like(energy_dos_shift, dtype=float)
        d_xy_pdos_sum = np.zeros_like(energy_dos_shift, dtype=float)
        d_yz_pdos_sum = np.zeros_like(energy_dos_shift, dtype=float)
        d_z2_pdos_sum = np.zeros_like(energy_dos_shift, dtype=float)
        d_zx_pdos_sum = np.zeros_like(energy_dos_shift, dtype=float)
        x2_y2_pdos_sum = np.zeros_like(energy_dos_shift, dtype=float)

        energy_pdos_sum = None
        is_spin_polarized_total = (ncols_total >= 5)

        for ion_index in range(1, ions_number + 1):
            hdr = None
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                toks = line.split()
                if len(toks) < 4:
                    continue
                try:
                    _emax = float(toks[0])
                    _emin = float(toks[1])
                    _nedos = int(float(toks[2]))
                    _efermi = float(toks[3])
                    if _nedos == nedos and _emax > _emin:
                        hdr = toks
                        break
                except Exception:
                    continue

            if hdr is None:
                print(f"Error: Failed to locate PDOS header for ion {ion_index} in DOSCAR.")
                return

            if ion_index < index_start or ion_index > index_end:
                for _ in range(nedos):
                    if not f.readline():
                        print(f"Error: DOSCAR ended unexpectedly while skipping ion {ion_index} block.")
                        return
                continue

            first_pdos = f.readline()
            if not first_pdos:
                print(f"Error: DOSCAR ended unexpectedly while reading ion {ion_index} PDOS block.")
                return

            ncols_ion = len(first_pdos.split())
            if ncols_ion < 10:
                print(f"Error: Unexpected PDOS format for ion {ion_index} (too few columns: {ncols_ion}).")
                return

            pdos_lines = [first_pdos]
            for _ in range(nedos - 1):
                line = f.readline()
                if not line:
                    break
                pdos_lines.append(line)

            flatp = np.fromstring("".join(pdos_lines), sep=" ")
            if flatp.size % ncols_ion != 0:
                pdata = []
                for row in pdos_lines:
                    parts = row.split()
                    if len(parts) == ncols_ion:
                        pdata.append([float(x) for x in parts])
                ion_arr = np.array(pdata, dtype=float)
            else:
                ion_arr = flatp.reshape(-1, ncols_ion)

            if energy_pdos_sum is None:
                energy_pdos_sum = ion_arr[:, 0]

            if ncols_ion in (10, 11):
                s_pdos_sum    += ion_arr[:, 1]
                p_y_pdos_sum  += ion_arr[:, 2]
                p_z_pdos_sum  += ion_arr[:, 3]
                p_x_pdos_sum  += ion_arr[:, 4]
                d_xy_pdos_sum += ion_arr[:, 5]
                d_yz_pdos_sum += ion_arr[:, 6]
                d_z2_pdos_sum += ion_arr[:, 7]
                d_zx_pdos_sum += ion_arr[:, 8]
                x2_y2_pdos_sum += ion_arr[:, 9]
            # elif ncols_ion in (19, 21):
            #     s_pdos_sum    += ion_arr[:, 1]
            #     p_y_pdos_sum  += ion_arr[:, 3]
            #     p_z_pdos_sum  += ion_arr[:, 5]
            #     p_x_pdos_sum  += ion_arr[:, 7]
            #     d_xy_pdos_sum += ion_arr[:, 9]
            #     d_yz_pdos_sum += ion_arr[:, 11]
            #     d_z2_pdos_sum += ion_arr[:, 13]
            #     d_zx_pdos_sum += ion_arr[:, 15]
            #     x2_y2_pdos_sum += ion_arr[:, 17]
            elif ncols_ion in (19, 21):
                s_pdos_sum    += ion_arr[:, 1]  + ion_arr[:, 2]
                p_y_pdos_sum  += ion_arr[:, 3]  + ion_arr[:, 4]
                p_z_pdos_sum  += ion_arr[:, 5]  + ion_arr[:, 6]
                p_x_pdos_sum  += ion_arr[:, 7]  + ion_arr[:, 8]
                d_xy_pdos_sum += ion_arr[:, 9]  + ion_arr[:, 10]
                d_yz_pdos_sum += ion_arr[:, 11] + ion_arr[:, 12]
                d_z2_pdos_sum += ion_arr[:, 13] + ion_arr[:, 14]
                d_zx_pdos_sum += ion_arr[:, 15] + ion_arr[:, 16]
                x2_y2_pdos_sum += ion_arr[:, 17] + ion_arr[:, 18]
            else:
                if is_spin_polarized_total:
                    print(
                        f"Error: PDOS block for ion {ion_index} has {ncols_ion} columns. "
                        "Expected LORBIT=11 spin-polarized format: 19 or 21 columns."
                    )
                else:
                    print(
                        f"Error: PDOS block for ion {ion_index} has {ncols_ion} columns. "
                        "Expected LORBIT=11 non-spin format: 10 or 11 columns."
                    )
                return

    if energy_pdos_sum is None:
        print(f"Error: No PDOS data accumulated for element '{element}'. Check ion index mapping.")
        return

    total_pdos_list = (
        s_pdos_sum + p_y_pdos_sum + p_z_pdos_sum + p_x_pdos_sum +
        d_xy_pdos_sum + d_yz_pdos_sum + d_z2_pdos_sum + d_zx_pdos_sum +
        x2_y2_pdos_sum
    )
    integrated_pdos_list = np.trapz(total_pdos_list, x=energy_dos_shift)
    energy_pdos_shift = energy_pdos_sum - efermi

    return (
        efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,
        energy_dos_shift,
        total_pdos_list, integrated_pdos_list,
        energy_pdos_shift, s_pdos_sum, p_y_pdos_sum, p_z_pdos_sum, p_x_pdos_sum,
        d_xy_pdos_sum, d_yz_pdos_sum, d_z2_pdos_sum, d_zx_pdos_sum,
        x2_y2_pdos_sum
    )

def extract_element_pdos_backup(directory_path, element):
    ## Construct the full path to the vasprun.xml file
    file_path = os.path.join(directory_path, "vasprun.xml")
    # Check if the vasprun.xml file exists in the given directory
    if not os.path.isfile(file_path):
        print(f"Error: The file vasprun.xml does not exist in the directory {directory_path}.")
        return

    ## Analysis vasprun.xml file
    tree = ET.parse(file_path)
    root = tree.getroot()
    # kpoints_file_path = os.path.join(directory_path, "KPOINTS")
    # kpoints_opt_path = os.path.join(directory_path, "KPOINTS_OPT")

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

    ## Analysis elements
    index_start = get_elements(directory_path)[element][0]
    index_end = get_elements(directory_path)[element][1]

    ## Extract the number of kpoints
    kpoints_number =extract_kpoints_number(directory_path)

    ## Extract eigen, occupancy number
    eigen_matrix = extract_eigen_occupancy(directory_path)[0]
    occu_matrix  = extract_eigen_occupancy(directory_path)[1]

    ## Extract energy list
    # lists initialization
    total_pdos_list     = np.array([])
    integrated_pdos_list = np.array([])

    shift = extract_fermi(directory_path)
    energy_dos_shift = extract_energy_shift(directory_path)

    ## Extract energy, s-PDoS, p_y-PDoS, p_z-PDoS, p_x-PDoS, d_xy-PDoS, d_yz-PDoS, d_z2-PDoS, d_zx-PDoS, x2-y2-PDoS
    # Matrices initialization
    for ions_index in range(index_start, index_end + 1):
        path_ions = f".//set[@comment='ion {ions_index}']/set[@comment='spin 1']/r"
        # Columns initialization
        energy_pdos_column      = np.array([])
        s_pdos_column           = np.array([])
        p_y_pdos_column         = np.array([])
        p_z_pdos_column         = np.array([])
        p_x_pdos_column         = np.array([])
        d_xy_pdos_column        = np.array([])
        d_yz_pdos_column        = np.array([])
        d_z2_pdos_column        = np.array([])
        d_zx_pdos_column        = np.array([])
        x2_y2_pdos_column       = np.array([])
        for pdos_element in root.findall(path_ions):
            pdos_values = list(map(float, pdos_element.text.split()))
            # Columns of energy
            energy_pdos_column = np.append(energy_pdos_column, pdos_values[0])
            # Columns of s-PDoS
            s_pdos_column = np.append(s_pdos_column, pdos_values[1])
            # Columns of p_y-PDoS
            p_y_pdos_column = np.append(p_y_pdos_column, pdos_values[2])
            # Columns of p_z-PDoS
            p_z_pdos_column = np.append(p_z_pdos_column, pdos_values[3])
            # Columns of p_x-PDoS
            p_x_pdos_column = np.append(p_x_pdos_column, pdos_values[4])
            # Columns of d_xy-PDoS
            d_xy_pdos_column = np.append(d_xy_pdos_column, pdos_values[5])
            # Columns of d_yz-PDoS
            d_yz_pdos_column = np.append(d_yz_pdos_column, pdos_values[6])
            # Columns of d_z2-PDoS
            d_z2_pdos_column = np.append(d_z2_pdos_column, pdos_values[7])
            # Columns of d_zx-PDoS
            d_zx_pdos_column = np.append(d_zx_pdos_column, pdos_values[8])
            # Columns of x2-y2-PDoS
            x2_y2_pdos_column = np.append(x2_y2_pdos_column, pdos_values[9])
        if ions_index == index_start:
            energy_pdos_matrix = energy_pdos_column.reshape(-1, 1)
            s_pdos_matrix = s_pdos_column.reshape(-1, 1)
            p_y_pdos_matrix = p_y_pdos_column.reshape(-1, 1)
            p_z_pdos_matrix = p_z_pdos_column.reshape(-1, 1)
            p_x_pdos_matrix = p_x_pdos_column.reshape(-1, 1)
            d_xy_pdos_matrix = d_xy_pdos_column.reshape(-1, 1)
            d_yz_pdos_matrix = d_yz_pdos_column.reshape(-1, 1)
            d_z2_pdos_matrix = d_z2_pdos_column.reshape(-1, 1)
            d_zx_pdos_matrix = d_zx_pdos_column.reshape(-1, 1)
            x2_y2_pdos_matrix = x2_y2_pdos_column.reshape(-1, 1)
        else:
            energy_pdos_matrix = np.hstack((energy_pdos_matrix, energy_pdos_column.reshape(-1, 1)))
            s_pdos_matrix = np.hstack((s_pdos_matrix, s_pdos_column.reshape(-1, 1)))
            p_y_pdos_matrix = np.hstack((p_y_pdos_matrix, p_y_pdos_column.reshape(-1, 1)))
            p_z_pdos_matrix = np.hstack((p_z_pdos_matrix, p_z_pdos_column.reshape(-1, 1)))
            p_x_pdos_matrix = np.hstack((p_x_pdos_matrix, p_x_pdos_column.reshape(-1, 1)))
            d_xy_pdos_matrix = np.hstack((d_xy_pdos_matrix, d_xy_pdos_column.reshape(-1, 1)))
            d_yz_pdos_matrix = np.hstack((d_yz_pdos_matrix, d_yz_pdos_column.reshape(-1, 1)))
            d_z2_pdos_matrix = np.hstack((d_z2_pdos_matrix, d_z2_pdos_column.reshape(-1, 1)))
            d_zx_pdos_matrix = np.hstack((d_zx_pdos_matrix, d_zx_pdos_column.reshape(-1, 1)))
            x2_y2_pdos_matrix = np.hstack((x2_y2_pdos_matrix, x2_y2_pdos_column.reshape(-1, 1)))
    energy_pdos_sum = energy_pdos_matrix[:,0]
    s_pdos_sum = np.sum(s_pdos_matrix, axis=1)
    p_y_pdos_sum = np.sum(p_y_pdos_matrix, axis=1)
    p_z_pdos_sum = np.sum(p_z_pdos_matrix, axis=1)
    p_x_pdos_sum = np.sum(p_x_pdos_matrix, axis=1)
    d_xy_pdos_sum = np.sum(d_xy_pdos_matrix, axis=1)
    d_yz_pdos_sum = np.sum(d_yz_pdos_matrix, axis=1)
    d_z2_pdos_sum = np.sum(d_z2_pdos_matrix, axis=1)
    d_zx_pdos_sum = np.sum(d_zx_pdos_matrix, axis=1)
    x2_y2_pdos_sum = np.sum(x2_y2_pdos_matrix, axis=1)
    total_pdos_list = s_pdos_sum + p_y_pdos_sum + p_z_pdos_sum + p_x_pdos_sum + d_xy_pdos_sum + d_yz_pdos_sum + d_z2_pdos_sum + d_zx_pdos_sum + x2_y2_pdos_sum
    integrated_pdos_list = np.trapz(total_pdos_list, x = energy_dos_shift)
    energy_pdos_shift = energy_pdos_sum - shift
    return (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,             # 0 ~ 4
            energy_dos_shift,                                                           # 5
            total_pdos_list, integrated_pdos_list,                                       # 6 ~ 7
            energy_pdos_shift, s_pdos_sum, p_y_pdos_sum, p_z_pdos_sum, p_x_pdos_sum,    # 8 ~ 12
            d_xy_pdos_sum, d_yz_pdos_sum, d_z2_pdos_sum, d_zx_pdos_sum,                 # 13 ~ 16
            x2_y2_pdos_sum)

# PDoS for customized range
def extract_segment_pdos(directory_path, start, end=None):
    """
    Extract segment-resolved orbital PDOS from VASP DOSCAR for ions in [start, end].

    Requirements
    1. DOSCAR must include per-ion, per-orbital projections (typically LORBIT = 11).
    2. CONTCAR or POSCAR must exist to determine the total number of ions.

    Notes
    - This function follows the original behavior of reading the "spin 1" channel only.
      If DOSCAR is spin-polarized, it uses the spin-up columns.
    - eigen_matrix and occu_matrix are returned as None for DOSCAR-based extraction.

    Returns (same layout as your original function)
    (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,                 # 0 ~ 4
     energy_dos_shift, total_pdos_list, integrated_pdos_list,                        # 5 ~ 7
     energy_pdos_shift, s_pdos_sum, p_y_pdos_sum, p_z_pdos_sum, p_x_pdos_sum,        # 8 ~ 12
     d_xy_pdos_sum, d_yz_pdos_sum, d_z2_pdos_sum, d_zx_pdos_sum,                     # 13 ~ 16
     x2_y2_pdos_sum)                                                                 # 17
    """
    doscar_path = os.path.join(directory_path, "DOSCAR")
    if not os.path.isfile(doscar_path):
        print(f"Error: The file DOSCAR does not exist in the directory {directory_path}.")
        return

    if end is None:
        index_start = int(start)
        index_end = int(start)
    else:
        index_start = int(start)
        index_end = int(end)

    if index_start < 1 or index_end < 1 or index_end < index_start:
        print("Error: Invalid ion index range. Use 1-based indices and ensure end >= start.")
        return

    poscar_path = os.path.join(directory_path, "CONTCAR")
    if not os.path.isfile(poscar_path):
        poscar_path = os.path.join(directory_path, "POSCAR")
    if not os.path.isfile(poscar_path):
        print("Error: CONTCAR/POSCAR not found. Cannot determine the total number of ions.")
        return

    ions_number = None
    try:
        with open(poscar_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [next(f) for _ in range(8)]
        tokens6 = lines[5].split() if len(lines) > 5 else []
        tokens7 = lines[6].split() if len(lines) > 6 else []

        def _is_intlike(tok):
            try:
                int(tok)
                return True
            except Exception:
                try:
                    return float(tok).is_integer()
                except Exception:
                    return False

        if tokens6 and all(_is_intlike(t) for t in tokens6):
            ions_number = int(sum(int(float(t)) for t in tokens6))
        elif tokens7 and all(_is_intlike(t) for t in tokens7):
            ions_number = int(sum(int(float(t)) for t in tokens7))
    except Exception:
        ions_number = None

    if ions_number is None:
        print("Error: Failed to parse ions_number from CONTCAR/POSCAR.")
        return

    if index_end > ions_number:
        print(f"Error: end index {index_end} exceeds ions_number {ions_number}.")
        return

    kpoints_number = None
    ibzkpt_path = os.path.join(directory_path, "IBZKPT")
    if os.path.isfile(ibzkpt_path):
        try:
            with open(ibzkpt_path, "r", encoding="utf-8", errors="ignore") as f:
                _ = f.readline()
                line2 = f.readline()
            kpoints_number = int(line2.split()[0])
        except Exception:
            kpoints_number = None

    eigen_matrix = None
    occu_matrix = None

    with open(doscar_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(5):
            if not f.readline():
                print("Error: DOSCAR is too short.")
                return

        header_line = f.readline()
        if not header_line:
            print("Error: DOSCAR ended unexpectedly.")
            return

        header_tokens = header_line.split()
        if len(header_tokens) < 4:
            found = False
            for _ in range(8000):
                line = f.readline()
                if not line:
                    break
                toks = line.split()
                if len(toks) < 4:
                    continue
                try:
                    emax = float(toks[0])
                    emin = float(toks[1])
                    nedos = int(float(toks[2]))
                    efermi = float(toks[3])
                    if nedos > 10 and emax > emin and abs(efermi) < 1.0e5:
                        header_tokens = toks
                        found = True
                        break
                except Exception:
                    continue
            if not found:
                print("Error: Failed to locate TOTAL DOS header line in DOSCAR.")
                return

        try:
            nedos = int(float(header_tokens[2]))
            efermi = float(header_tokens[3])
        except Exception:
            print("Error: Failed to parse TOTAL DOS header line in DOSCAR.")
            return

        first_row = f.readline()
        if not first_row:
            print("Error: DOSCAR ended unexpectedly while reading TOTAL DOS block.")
            return

        ncols_total = len(first_row.split())
        if ncols_total < 3:
            print("Error: Unexpected TOTAL DOS row format in DOSCAR.")
            return

        total_lines = [first_row]
        for _ in range(nedos - 1):
            line = f.readline()
            if not line:
                break
            total_lines.append(line)

        flat = np.fromstring("".join(total_lines), sep=" ")
        if flat.size % ncols_total != 0:
            data = []
            for row in total_lines:
                parts = row.split()
                if len(parts) == ncols_total:
                    data.append([float(x) for x in parts])
            total_arr = np.array(data, dtype=float)
        else:
            total_arr = flat.reshape(-1, ncols_total)

        energy_total = total_arr[:, 0]
        energy_dos_shift = energy_total - efermi

        energy_pdos_sum = None
        s_pdos_sum = p_y_pdos_sum = p_z_pdos_sum = p_x_pdos_sum = None
        d_xy_pdos_sum = d_yz_pdos_sum = d_z2_pdos_sum = d_zx_pdos_sum = x2_y2_pdos_sum = None

        for ion_index in range(1, ions_number + 1):
            ion_header = None
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                toks = line.split()
                if len(toks) < 4:
                    continue
                try:
                    _emax = float(toks[0])
                    _emin = float(toks[1])
                    _nedos = int(float(toks[2]))
                    _efermi = float(toks[3])
                    if _nedos == nedos and _emax > _emin:
                        ion_header = toks
                        break
                except Exception:
                    continue

            if ion_header is None:
                print(f"Error: Failed to locate PDOS header for ion {ion_index} in DOSCAR.")
                return

            if ion_index < index_start or ion_index > index_end:
                for _ in range(nedos):
                    if not f.readline():
                        print(f"Error: DOSCAR ended unexpectedly while skipping ion {ion_index} block.")
                        return
                continue

            first_pdos = f.readline()
            if not first_pdos:
                print(f"Error: DOSCAR ended unexpectedly while reading ion {ion_index} PDOS block.")
                return

            ncols_ion = len(first_pdos.split())
            if ncols_ion < 10:
                print(f"Error: Unexpected PDOS row format for ion {ion_index} (too few columns: {ncols_ion}).")
                return

            pdos_lines = [first_pdos]
            for _ in range(nedos - 1):
                line = f.readline()
                if not line:
                    break
                pdos_lines.append(line)

            flatp = np.fromstring("".join(pdos_lines), sep=" ")
            if flatp.size % ncols_ion != 0:
                pdata = []
                for row in pdos_lines:
                    parts = row.split()
                    if len(parts) == ncols_ion:
                        pdata.append([float(x) for x in parts])
                ion_arr = np.array(pdata, dtype=float)
            else:
                ion_arr = flatp.reshape(-1, ncols_ion)

            if energy_pdos_sum is None:
                energy_pdos_sum = ion_arr[:, 0]
                ngrid = ion_arr.shape[0]
                s_pdos_sum = np.zeros(ngrid, dtype=float)
                p_y_pdos_sum = np.zeros(ngrid, dtype=float)
                p_z_pdos_sum = np.zeros(ngrid, dtype=float)
                p_x_pdos_sum = np.zeros(ngrid, dtype=float)
                d_xy_pdos_sum = np.zeros(ngrid, dtype=float)
                d_yz_pdos_sum = np.zeros(ngrid, dtype=float)
                d_z2_pdos_sum = np.zeros(ngrid, dtype=float)
                d_zx_pdos_sum = np.zeros(ngrid, dtype=float)
                x2_y2_pdos_sum = np.zeros(ngrid, dtype=float)

            if ncols_ion in (10, 11):
                s_pdos_sum += ion_arr[:, 1]
                p_y_pdos_sum += ion_arr[:, 2]
                p_z_pdos_sum += ion_arr[:, 3]
                p_x_pdos_sum += ion_arr[:, 4]
                d_xy_pdos_sum += ion_arr[:, 5]
                d_yz_pdos_sum += ion_arr[:, 6]
                d_z2_pdos_sum += ion_arr[:, 7]
                d_zx_pdos_sum += ion_arr[:, 8]
                x2_y2_pdos_sum += ion_arr[:, 9]
            # elif ncols_ion in (19, 21):
            #     s_pdos_sum += ion_arr[:, 1]
            #     p_y_pdos_sum += ion_arr[:, 3]
            #     p_z_pdos_sum += ion_arr[:, 5]
            #     p_x_pdos_sum += ion_arr[:, 7]
            #     d_xy_pdos_sum += ion_arr[:, 9]
            #     d_yz_pdos_sum += ion_arr[:, 11]
            #     d_z2_pdos_sum += ion_arr[:, 13]
            #     d_zx_pdos_sum += ion_arr[:, 15]
            #     x2_y2_pdos_sum += ion_arr[:, 17]
            elif ncols_ion in (19, 21):
                s_pdos_sum    += ion_arr[:, 1]  + ion_arr[:, 2]
                p_y_pdos_sum  += ion_arr[:, 3]  + ion_arr[:, 4]
                p_z_pdos_sum  += ion_arr[:, 5]  + ion_arr[:, 6]
                p_x_pdos_sum  += ion_arr[:, 7]  + ion_arr[:, 8]
                d_xy_pdos_sum += ion_arr[:, 9]  + ion_arr[:, 10]
                d_yz_pdos_sum += ion_arr[:, 11] + ion_arr[:, 12]
                d_z2_pdos_sum += ion_arr[:, 13] + ion_arr[:, 14]
                d_zx_pdos_sum += ion_arr[:, 15] + ion_arr[:, 16]
                x2_y2_pdos_sum += ion_arr[:, 17] + ion_arr[:, 18]
            else:
                print(
                    f"Error: PDOS block for ion {ion_index} has {ncols_ion} columns. "
                    "Expected LORBIT=11 formats: 10/11 (non-spin) or 19/21 (spin-polarized)."
                )
                return

    if energy_pdos_sum is None:
        print("Error: No PDOS data accumulated. Check start/end indices and DOSCAR content.")
        return

    energy_pdos_shift = energy_pdos_sum - efermi
    total_pdos_list = (
        s_pdos_sum + p_y_pdos_sum + p_z_pdos_sum + p_x_pdos_sum +
        d_xy_pdos_sum + d_yz_pdos_sum + d_z2_pdos_sum + d_zx_pdos_sum +
        x2_y2_pdos_sum
    )
    integrated_pdos_list = np.trapz(total_pdos_list, x=energy_dos_shift)

    return (
        efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,
        energy_dos_shift, total_pdos_list, integrated_pdos_list,
        energy_pdos_shift, s_pdos_sum, p_y_pdos_sum, p_z_pdos_sum, p_x_pdos_sum,
        d_xy_pdos_sum, d_yz_pdos_sum, d_z2_pdos_sum, d_zx_pdos_sum,
        x2_y2_pdos_sum
    )

def extract_segment_pdos_backup(directory_path, start, end = None):
    ## Construct the full path to the vasprun.xml file
    file_path = os.path.join(directory_path, "vasprun.xml")
    # Check if the vasprun.xml file exists in the given directory
    if not os.path.isfile(file_path):
        print(f"Error: The file vasprun.xml does not exist in the directory {directory_path}.")
        return

    ## Analysis vasprun.xml file
    tree = ET.parse(file_path)
    root = tree.getroot()

    ## Analysis elements
    if end is None:
        index_start = start
        index_end = start
    else:
        index_start = start
        index_end = end    

    ## Extract Fermi energy
    efermi = extract_fermi(directory_path)

    ## Extract the number of ions
    first_positions = root.find(".//varray[@name='positions'][1]")
    positions_concatenated_text = " ".join([position.text for position in first_positions.findall("v")])
    positions_array = np.fromstring(positions_concatenated_text, sep=" ")
    positions_matrix = positions_array.reshape(-1, 3)
    ions_number = positions_matrix.shape[0]

    ## Extract the number of kpoints
    kpoints_number =extract_kpoints_number(directory_path)

    ## Extract eigen, occupancy number
    ## Extract eigen, occupancy number
    eigen_matrix = extract_eigen_occupancy(directory_path)[0]
    occu_matrix  = extract_eigen_occupancy(directory_path)[1]
    # eigen_sum = np.sum(eigen_matrix, axis=1)
    # occu_sum  = np.sum(occu_matrix, axis=1)

    ## Extract energy list
    # lists initialization
    total_pdos_list     = np.array([])
    integrated_pdos_list = np.array([])

    shift = extract_fermi(directory_path)
    energy_dos_shift = extract_energy_shift(directory_path)

    ## Extract energy, s-PDoS, p_y-PDoS, p_z-PDoS, p_x-PDoS, d_xy-PDoS, d_yz-PDoS, d_z2-PDoS, d_zx-PDoS, x2-y2-PDoS
    # Matrices initialization
    for ions_index in range(index_start, index_end + 1):
        path_ions = f".//set[@comment='ion {ions_index}']/set[@comment='spin 1']/r"
        # Columns initialization
        energy_pdos_column  = np.empty(0)
        s_pdos_column       = np.empty(0)
        p_y_pdos_column     = np.empty(0)
        p_z_pdos_column     = np.empty(0)
        p_x_pdos_column     = np.empty(0)
        d_xy_pdos_column    = np.empty(0)
        d_yz_pdos_column    = np.empty(0)
        d_z2_pdos_column    = np.empty(0)
        d_zx_pdos_column    = np.empty(0)
        x2_y2_pdos_column   = np.empty(0)
        for pdos_element in root.findall(path_ions):
            pdos_values = list(map(float, pdos_element.text.split()))
            # Columns of energy
            energy_pdos_column = np.append(energy_pdos_column, pdos_values[0])
            # Columns of s-PDoS
            s_pdos_column = np.append(s_pdos_column, pdos_values[1])
            # Columns of p_y-PDoS
            p_y_pdos_column = np.append(p_y_pdos_column, pdos_values[2])
            # Columns of p_z-PDoS
            p_z_pdos_column = np.append(p_z_pdos_column, pdos_values[3])
            # Columns of p_x-PDoS
            p_x_pdos_column = np.append(p_x_pdos_column, pdos_values[4])
            # Columns of d_xy-PDoS
            d_xy_pdos_column = np.append(d_xy_pdos_column, pdos_values[5])
            # Columns of d_yz-PDoS
            d_yz_pdos_column = np.append(d_yz_pdos_column, pdos_values[6])
            # Columns of d_z2-PDoS
            d_z2_pdos_column = np.append(d_z2_pdos_column, pdos_values[7])
            # Columns of d_zx-PDoS
            d_zx_pdos_column = np.append(d_zx_pdos_column, pdos_values[8])
            # Columns of x2-y2-PDoS
            x2_y2_pdos_column = np.append(x2_y2_pdos_column, pdos_values[9])
        if ions_index == index_start:
            energy_pdos_matrix = energy_pdos_column.reshape(-1, 1)
            s_pdos_matrix = s_pdos_column.reshape(-1, 1)
            p_y_pdos_matrix = p_y_pdos_column.reshape(-1, 1)
            p_z_pdos_matrix = p_z_pdos_column.reshape(-1, 1)
            p_x_pdos_matrix = p_x_pdos_column.reshape(-1, 1)
            d_xy_pdos_matrix = d_xy_pdos_column.reshape(-1, 1)
            d_yz_pdos_matrix = d_yz_pdos_column.reshape(-1, 1)
            d_z2_pdos_matrix = d_z2_pdos_column.reshape(-1, 1)
            d_zx_pdos_matrix = d_zx_pdos_column.reshape(-1, 1)
            x2_y2_pdos_matrix = x2_y2_pdos_column.reshape(-1, 1)
        else:
            energy_pdos_matrix = np.hstack((energy_pdos_matrix, energy_pdos_column.reshape(-1, 1)))
            s_pdos_matrix = np.hstack((s_pdos_matrix, s_pdos_column.reshape(-1, 1)))
            p_y_pdos_matrix = np.hstack((p_y_pdos_matrix, p_y_pdos_column.reshape(-1, 1)))
            p_z_pdos_matrix = np.hstack((p_z_pdos_matrix, p_z_pdos_column.reshape(-1, 1)))
            p_x_pdos_matrix = np.hstack((p_x_pdos_matrix, p_x_pdos_column.reshape(-1, 1)))
            d_xy_pdos_matrix = np.hstack((d_xy_pdos_matrix, d_xy_pdos_column.reshape(-1, 1)))
            d_yz_pdos_matrix = np.hstack((d_yz_pdos_matrix, d_yz_pdos_column.reshape(-1, 1)))
            d_z2_pdos_matrix = np.hstack((d_z2_pdos_matrix, d_z2_pdos_column.reshape(-1, 1)))
            d_zx_pdos_matrix = np.hstack((d_zx_pdos_matrix, d_zx_pdos_column.reshape(-1, 1)))
            x2_y2_pdos_matrix = np.hstack((x2_y2_pdos_matrix, x2_y2_pdos_column.reshape(-1, 1)))
    energy_pdos_sum = energy_pdos_matrix[:,0]
    s_pdos_sum = np.sum(s_pdos_matrix, axis=1)
    p_y_pdos_sum = np.sum(p_y_pdos_matrix, axis=1)
    p_z_pdos_sum = np.sum(p_z_pdos_matrix, axis=1)
    p_x_pdos_sum = np.sum(p_x_pdos_matrix, axis=1)
    d_xy_pdos_sum = np.sum(d_xy_pdos_matrix, axis=1)
    d_yz_pdos_sum = np.sum(d_yz_pdos_matrix, axis=1)
    d_z2_pdos_sum = np.sum(d_z2_pdos_matrix, axis=1)
    d_zx_pdos_sum = np.sum(d_zx_pdos_matrix, axis=1)
    x2_y2_pdos_sum = np.sum(x2_y2_pdos_matrix, axis=1)
    energy_pdos_shift = energy_pdos_sum - shift
    total_pdos_list = s_pdos_sum + p_y_pdos_sum + p_z_pdos_sum + p_x_pdos_sum + d_xy_pdos_sum + d_yz_pdos_sum + d_z2_pdos_sum + d_zx_pdos_sum + x2_y2_pdos_sum
    integrated_pdos_list = np.trapz(total_pdos_list, x=energy_dos_shift)

    return (efermi, ions_number, kpoints_number, eigen_matrix, occu_matrix,                 # 0 ~ 4
            energy_dos_shift, total_pdos_list, integrated_pdos_list,                        # 5 ~ 7
            energy_pdos_shift, s_pdos_sum, p_y_pdos_sum, p_z_pdos_sum, p_x_pdos_sum,        # 8 ~ 12
            d_xy_pdos_sum, d_yz_pdos_sum, d_z2_pdos_sum, d_zx_pdos_sum,                     # 13 ~ 16
            x2_y2_pdos_sum)

def extract_index_pdos(directory_path, index=None):
    """
    Extract orbital-resolved PDOS from VASP DOSCAR for specified ion indices.

    Requirements
    1. DOSCAR must include per-ion, per-orbital projections (typically LORBIT = 11).
    2. CONTCAR or POSCAR must exist to determine the total number of ions.

    Notes
    - This function follows the original behavior of using the spin-up channel if DOSCAR is spin-polarized.
    - eigen_matrix and occu_matrix are returned as None for DOSCAR-based extraction.

    Parameters
    ----------
    directory_path : str
        Directory containing DOSCAR.
    index : int, tuple, list, or nested combination
        Ion selection rules (1-based indices):
        - None / "All" / "Total" => all ions
        - int => a single ion
        - tuple(a, b) => inclusive range [a, b]
        - list/tuple => can contain ints, ranges, or nested lists/tuples

    Returns
    -------
    tuple
        (efermi, total_ions, kpoints_number, eigen_matrix, occu_matrix,              # 0 ~ 4
         energy_pdos_shift, total_pdos_list, integrated_pdos_list,                   # 5 ~ 7
         energy_pdos_shift, s_pdos_sum, p_y_pdos_sum, p_z_pdos_sum, p_x_pdos_sum,    # 8 ~ 12
         d_xy_pdos_sum, d_yz_pdos_sum, d_z2_pdos_sum, d_zx_pdos_sum,                 # 13 ~ 16
         x2_y2_pdos_sum)                                                             # 17
    """
    doscar_path = os.path.join(directory_path, "DOSCAR")
    if not os.path.isfile(doscar_path):
        raise FileNotFoundError(f"Error: The file DOSCAR does not exist in the directory {directory_path}.")

    poscar_path = os.path.join(directory_path, "CONTCAR")
    if not os.path.isfile(poscar_path):
        poscar_path = os.path.join(directory_path, "POSCAR")
    if not os.path.isfile(poscar_path):
        raise FileNotFoundError("Error: CONTCAR/POSCAR not found. Cannot determine the total number of ions.")

    try:
        with open(poscar_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [next(f) for _ in range(8)]
    except Exception as exc:
        raise ValueError("Error: Failed to read CONTCAR/POSCAR.") from exc

    tokens6 = lines[5].split() if len(lines) > 5 else []
    tokens7 = lines[6].split() if len(lines) > 6 else []

    def _parse_counts(tokens):
        if not tokens:
            return None
        counts = []
        for t in tokens:
            try:
                counts.append(int(t))
                continue
            except Exception:
                pass
            try:
                v = float(t)
                if not v.is_integer():
                    return None
                counts.append(int(v))
            except Exception:
                return None
        return counts

    counts = _parse_counts(tokens6)
    if counts is None:
        counts = _parse_counts(tokens7)
    if counts is None:
        raise ValueError("Error: Failed to parse ion counts from CONTCAR/POSCAR.")

    total_ions = int(sum(counts))

    if index in (None, [], "All", "all", "Total", "total"):
        index_list = list(range(1, total_ions + 1))
    else:
        to_expand = [index]
        expanded = []
        while to_expand:
            item = to_expand.pop()
            if item in (None, [], "All", "all", "Total", "total"):
                expanded.extend(range(1, total_ions + 1))
            elif isinstance(item, int):
                expanded.append(item)
            elif isinstance(item, tuple) and len(item) == 2 and all(isinstance(x, int) for x in item):
                a, b = item
                if a <= b:
                    expanded.extend(range(a, b + 1))
                else:
                    expanded.extend(range(b, a + 1))
            elif isinstance(item, (list, tuple)):
                for sub in reversed(item):
                    to_expand.append(sub)
            else:
                raise ValueError(f"Invalid index type: {type(item)}")
        index_list = sorted(set(expanded))

    index_list = [i for i in index_list if 1 <= i <= total_ions]
    if not index_list:
        raise ValueError(f"No valid indices found. Valid range is 1 to {total_ions}.")

    index_set = set(index_list)

    kpoints_number = None
    ibzkpt_path = os.path.join(directory_path, "IBZKPT")
    if os.path.isfile(ibzkpt_path):
        try:
            with open(ibzkpt_path, "r", encoding="utf-8", errors="ignore") as f:
                _ = f.readline()
                line2 = f.readline()
            kpoints_number = int(line2.split()[0])
        except Exception:
            kpoints_number = None

    eigen_matrix = None
    occu_matrix = None

    with open(doscar_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(5):
            if not f.readline():
                raise ValueError("Error: DOSCAR is too short.")

        header_tokens = None
        for _ in range(8000):
            line = f.readline()
            if not line:
                break
            toks = line.split()
            if len(toks) < 4:
                continue
            try:
                emax = float(toks[0])
                emin = float(toks[1])
                nedos = int(float(toks[2]))
                efermi = float(toks[3])
                if nedos > 10 and emax > emin and abs(efermi) < 1.0e5:
                    header_tokens = toks
                    break
            except Exception:
                continue

        if header_tokens is None:
            raise ValueError("Error: Failed to locate TOTAL DOS header line in DOSCAR.")

        nedos = int(float(header_tokens[2]))
        efermi = float(header_tokens[3])

        first_row = f.readline()
        if not first_row:
            raise ValueError("Error: DOSCAR ended unexpectedly while reading TOTAL DOS block.")
        ncols_total = len(first_row.split())
        if ncols_total < 3:
            raise ValueError("Error: Unexpected TOTAL DOS row format in DOSCAR.")

        for _ in range(nedos - 1):
            if not f.readline():
                raise ValueError("Error: DOSCAR ended unexpectedly while reading TOTAL DOS block.")

        energy_pdos_sum = None
        s_pdos_sum = None
        p_y_pdos_sum = None
        p_z_pdos_sum = None
        p_x_pdos_sum = None
        d_xy_pdos_sum = None
        d_yz_pdos_sum = None
        d_z2_pdos_sum = None
        d_zx_pdos_sum = None
        x2_y2_pdos_sum = None

        for ion_index in range(1, total_ions + 1):
            ion_hdr_found = False
            for _ in range(300):
                line = f.readline()
                if not line:
                    break
                toks = line.split()
                if len(toks) < 4:
                    continue
                try:
                    _ = float(toks[0])
                    _ = float(toks[1])
                    _ned = int(float(toks[2]))
                    _ = float(toks[3])
                    if _ned == nedos:
                        ion_hdr_found = True
                        break
                except Exception:
                    continue

            if not ion_hdr_found:
                raise ValueError(f"Error: Failed to locate PDOS header for ion {ion_index} in DOSCAR.")

            if ion_index not in index_set:
                for _ in range(nedos):
                    if not f.readline():
                        raise ValueError(f"Error: DOSCAR ended unexpectedly while skipping ion {ion_index} PDOS block.")
                continue

            first_pdos = f.readline()
            if not first_pdos:
                raise ValueError(f"Error: DOSCAR ended unexpectedly while reading ion {ion_index} PDOS block.")

            ncols_ion = len(first_pdos.split())
            if ncols_ion < 10:
                raise ValueError(f"Error: Unexpected PDOS row format for ion {ion_index} (too few columns: {ncols_ion}).")

            pdos_lines = [first_pdos]
            for _ in range(nedos - 1):
                line = f.readline()
                if not line:
                    raise ValueError(f"Error: DOSCAR ended unexpectedly while reading ion {ion_index} PDOS block.")
                pdos_lines.append(line)

            flatp = np.fromstring("".join(pdos_lines), sep=" ")
            if flatp.size % ncols_ion != 0:
                pdata = []
                for row in pdos_lines:
                    parts = row.split()
                    if len(parts) == ncols_ion:
                        pdata.append([float(x) for x in parts])
                ion_arr = np.array(pdata, dtype=float)
            else:
                ion_arr = flatp.reshape(-1, ncols_ion)

            if energy_pdos_sum is None:
                energy_pdos_sum = ion_arr[:, 0]
                ngrid = ion_arr.shape[0]
                s_pdos_sum = np.zeros(ngrid, dtype=float)
                p_y_pdos_sum = np.zeros(ngrid, dtype=float)
                p_z_pdos_sum = np.zeros(ngrid, dtype=float)
                p_x_pdos_sum = np.zeros(ngrid, dtype=float)
                d_xy_pdos_sum = np.zeros(ngrid, dtype=float)
                d_yz_pdos_sum = np.zeros(ngrid, dtype=float)
                d_z2_pdos_sum = np.zeros(ngrid, dtype=float)
                d_zx_pdos_sum = np.zeros(ngrid, dtype=float)
                x2_y2_pdos_sum = np.zeros(ngrid, dtype=float)

            if ncols_ion in (10, 11):
                s_pdos_sum += ion_arr[:, 1]
                p_y_pdos_sum += ion_arr[:, 2]
                p_z_pdos_sum += ion_arr[:, 3]
                p_x_pdos_sum += ion_arr[:, 4]
                d_xy_pdos_sum += ion_arr[:, 5]
                d_yz_pdos_sum += ion_arr[:, 6]
                d_z2_pdos_sum += ion_arr[:, 7]
                d_zx_pdos_sum += ion_arr[:, 8]
                x2_y2_pdos_sum += ion_arr[:, 9]
            # elif ncols_ion in (19, 21):
            #     s_pdos_sum += ion_arr[:, 1]
            #     p_y_pdos_sum += ion_arr[:, 3]
            #     p_z_pdos_sum += ion_arr[:, 5]
            #     p_x_pdos_sum += ion_arr[:, 7]
            #     d_xy_pdos_sum += ion_arr[:, 9]
            #     d_yz_pdos_sum += ion_arr[:, 11]
            #     d_z2_pdos_sum += ion_arr[:, 13]
            #     d_zx_pdos_sum += ion_arr[:, 15]
            #     x2_y2_pdos_sum += ion_arr[:, 17]
            elif ncols_ion in (19, 21):
                s_pdos_sum    += ion_arr[:, 1]  + ion_arr[:, 2]
                p_y_pdos_sum  += ion_arr[:, 3]  + ion_arr[:, 4]
                p_z_pdos_sum  += ion_arr[:, 5]  + ion_arr[:, 6]
                p_x_pdos_sum  += ion_arr[:, 7]  + ion_arr[:, 8]
                d_xy_pdos_sum += ion_arr[:, 9]  + ion_arr[:, 10]
                d_yz_pdos_sum += ion_arr[:, 11] + ion_arr[:, 12]
                d_z2_pdos_sum += ion_arr[:, 13] + ion_arr[:, 14]
                d_zx_pdos_sum += ion_arr[:, 15] + ion_arr[:, 16]
                x2_y2_pdos_sum += ion_arr[:, 17] + ion_arr[:, 18]
            else:
                raise ValueError(
                    f"Error: PDOS block for ion {ion_index} has {ncols_ion} columns. "
                    "Expected LORBIT=11 formats: 10/11 (non-spin) or 19/21 (spin-polarized)."
                )

    total_pdos_list = (
        s_pdos_sum + p_y_pdos_sum + p_z_pdos_sum + p_x_pdos_sum +
        d_xy_pdos_sum + d_yz_pdos_sum + d_z2_pdos_sum + d_zx_pdos_sum +
        x2_y2_pdos_sum
    )
    energy_pdos_shift = energy_pdos_sum - efermi
    integrated_pdos_list = np.trapz(total_pdos_list, x=energy_pdos_shift)

    return (
        efermi, total_ions, kpoints_number, eigen_matrix, occu_matrix,
        energy_pdos_shift, total_pdos_list, integrated_pdos_list,
        energy_pdos_shift, s_pdos_sum, p_y_pdos_sum, p_z_pdos_sum, p_x_pdos_sum,
        d_xy_pdos_sum, d_yz_pdos_sum, d_z2_pdos_sum, d_zx_pdos_sum,
        x2_y2_pdos_sum
    )

def extract_index_pdos_backup(directory_path, index=None):
    """
    Extract PDoS data for specified ions from vasprun.xml, supporting GGA-PBE and HSE06 algorithms.

    Parameters:
        directory_path (str): Path to the directory containing vasprun.xml.
        index (int, list of int, or tuple, optional): Specific ion index, list of indices, or ranges (tuples).
                                                     If None, extract for all ions.

    Returns:
        tuple: Extracted data including Fermi energy, number of ions, kpoints, eigenvalues, and PDoS components.
    """
    import os
    import xml.etree.ElementTree as ET
    import numpy as np

    # Construct the full path to the vasprun.xml file
    file_path = os.path.join(directory_path, "vasprun.xml")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Error: The file vasprun.xml does not exist in the directory {directory_path}.")

    # Parse the vasprun.xml file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # # Get total number of atoms
    # def count_pdos_atoms(directory_path):
    #     """Count total atoms in vasprun.xml"""
    #     positions_section = root.find(".//varray[@name='positions'][1]")
    #     if positions_section is None:
    #         raise ValueError("Failed to locate the positions section in vasprun.xml.")
    #     return len(positions_section.findall("v"))

    total_ions = count_pdos_atoms_vasp(directory_path)

    # Handle index logic: Normalize index to a list of integers
    # if index in (None, [], "All", "all", "Total", "total"):  # Special cases: all ions
    #     index = list(range(1, total_ions + 1))
    # elif isinstance(index, int):  # Single integer
    #     index = [index]
    # elif isinstance(index, tuple):  # Single tuple (range)
    #     index = list(range(index[0], index[1] + 1))
    # elif isinstance(index, list):  # List with mixed values
    #     expanded_index = []
    #     for item in index:
    #         if isinstance(item, tuple):
    #             expanded_index.extend(range(item[0], item[1] + 1))
    #         else:
    #             expanded_index.append(item)
    #     index = sorted(set(expanded_index))     # Remove duplicates and sort
    # else:
    #     raise ValueError(f"Invalid index type: {type(index)}")

    # Handle index logic: Normalize index to a list of integers
    if index in (None, [], "All", "all", "Total", "total"):  # Special cases: all ions
        index = list(range(1, total_ions + 1))
    else:
        to_expand = [index]
        expanded_index = []
        while to_expand:
            item = to_expand.pop()
            if item in (None, [], "All", "all", "Total", "total"):
                expanded_index.extend(range(1, total_ions + 1))
            elif isinstance(item, int):
                expanded_index.append(item)
            elif isinstance(item, tuple) and len(item) == 2 and all(isinstance(x, int) for x in item):
                expanded_index.extend(range(item[0], item[1] + 1))
            elif isinstance(item, (list, tuple)):
                to_expand.extend(reversed(item))
            else:
                raise ValueError(f"Invalid index type: {type(item)}")
        index = sorted(set(expanded_index))

    # Filter out indices that are out of range
    index = [i for i in index if 1 <= i <= total_ions]

    # test
    print(index)

    # If no valid indices remain after filtering, raise an error
    if not index:
        raise ValueError(f"No valid indices found. Valid range is 1 to {total_ions}.")

    # Extract Fermi energy
    def extract_fermi(directory_path):
        efermi_element = root.find(".//dos/i[@name='efermi']")
        if efermi_element is None:
            raise ValueError("Failed to extract Fermi energy from vasprun.xml.")
        return float(efermi_element.text.strip())

    efermi = extract_fermi(directory_path)

    # Identify HSE06 or PBE
    kpoints_file_path = os.path.join(directory_path, "KPOINTS")
    kpoints_opt_path = os.path.join(directory_path, "KPOINTS_OPT")
    use_hse06 = os.path.exists(kpoints_opt_path)

    # Extract the number of kpoints
    if use_hse06:
        kpointlist = root.find(".//eigenvalues_kpoints_opt[@comment='kpoints_opt']/kpoints/varray[@name='kpointlist']")
    else:
        kpointlist = root.find(".//varray[@name='kpointlist']")
    kpoints_number = len(kpointlist.findall("v"))

    # Extract eigenvalues and occupancy numbers
    eigen_matrix = []
    occu_matrix = []

    for kpoints_index in range(1, kpoints_number + 1):
        if use_hse06:
            xpath_expr = f"./calculation/projected_kpoints_opt/eigenvalues/array/set/set[@comment='spin 1']/set[@comment='kpoint {kpoints_index}']"
        else:
            xpath_expr = f".//set[@comment='kpoint {kpoints_index}']"

        kpoint_set = root.find(xpath_expr)
        eigen_column = []
        occu_column = []

        for eigen_occ_element in kpoint_set:
            values = list(map(float, eigen_occ_element.text.split()))
            eigen_column.append(values[0])
            occu_column.append(values[1])

        eigen_matrix.append(eigen_column)
        occu_matrix.append(occu_column)

    eigen_matrix = np.array(eigen_matrix).T
    occu_matrix = np.array(occu_matrix).T

    # Extract PDoS components
    energy_pdos_sum = None
    s_pdos_sum = None
    p_y_pdos_sum = None
    p_z_pdos_sum = None
    p_x_pdos_sum = None
    d_xy_pdos_sum = None
    d_yz_pdos_sum = None
    d_z2_pdos_sum = None
    d_zx_pdos_sum = None
    x2_y2_pdos_sum = None

    for ions_index in index:
        path_ions = f".//set[@comment='ion {ions_index}']/set[@comment='spin 1']/r"
        pdos_data = np.array([
            list(map(float, pdos_element.text.split()))
            for pdos_element in root.findall(path_ions)
        ])

        if energy_pdos_sum is None:
            energy_pdos_sum = pdos_data[:, 0]
            s_pdos_sum = pdos_data[:, 1]
            p_y_pdos_sum = pdos_data[:, 2]
            p_z_pdos_sum = pdos_data[:, 3]
            p_x_pdos_sum = pdos_data[:, 4]
            d_xy_pdos_sum = pdos_data[:, 5]
            d_yz_pdos_sum = pdos_data[:, 6]
            d_z2_pdos_sum = pdos_data[:, 7]
            d_zx_pdos_sum = pdos_data[:, 8]
            x2_y2_pdos_sum = pdos_data[:, 9]
        else:
            s_pdos_sum += pdos_data[:, 1]
            p_y_pdos_sum += pdos_data[:, 2]
            p_z_pdos_sum += pdos_data[:, 3]
            p_x_pdos_sum += pdos_data[:, 4]
            d_xy_pdos_sum += pdos_data[:, 5]
            d_yz_pdos_sum += pdos_data[:, 6]
            d_z2_pdos_sum += pdos_data[:, 7]
            d_zx_pdos_sum += pdos_data[:, 8]
            x2_y2_pdos_sum += pdos_data[:, 9]

    total_pdos_list = (
        s_pdos_sum + p_y_pdos_sum + p_z_pdos_sum + p_x_pdos_sum +
        d_xy_pdos_sum + d_yz_pdos_sum + d_z2_pdos_sum + d_zx_pdos_sum + x2_y2_pdos_sum
    )
    energy_pdos_shift = energy_pdos_sum - efermi
    integrated_pdos_list = np.trapz(total_pdos_list, x=energy_pdos_shift)

    return (
        efermi, total_ions, kpoints_number, eigen_matrix, occu_matrix,              # 0 ~ 4
        energy_pdos_shift, total_pdos_list, integrated_pdos_list,                   # 5 ~ 7
        energy_pdos_shift, s_pdos_sum, p_y_pdos_sum, p_z_pdos_sum, p_x_pdos_sum,    # 8 ~ 12
        d_xy_pdos_sum, d_yz_pdos_sum, d_z2_pdos_sum, d_zx_pdos_sum,                 # 13 ~ 16
        x2_y2_pdos_sum                                                              # 17
    )

def extract_index_pdos_old(directory_path, index=None):
    """
    Extract PDoS data for specified ions from vasprun.xml, supporting GGA-PBE and HSE06 algorithms.

    Parameters:
        directory_path (str): Path to the directory containing vasprun.xml.
        index (int, list of int, or tuple, optional): Specific ion index, list of indices, or ranges (tuples).
                                                     If None, extract for all ions.

    Returns:
        tuple: Extracted data including Fermi energy, number of ions, kpoints, eigenvalues, and PDoS components.
    """
    # Construct the full path to the vasprun.xml file
    file_path = os.path.join(directory_path, "vasprun.xml")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Error: The file vasprun.xml does not exist in the directory {directory_path}.")

    # Parse the vasprun.xml file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Get total number of atoms
    total_ions = count_pdos_atoms_vasp(directory_path)

    # Handle index logic: Normalize index to a list of integers
    if index in (None, [], "All", "all", "Total", "total"):  # Special cases: all ions
        index = list(range(1, total_ions + 1))
    elif isinstance(index, int):  # Single integer
        index = [index]
    elif isinstance(index, tuple):  # Single tuple
        index = list(range(index[0], index[1] + 1))
    elif isinstance(index, list):  # List with mixed values
        expanded_index = []
        for item in index:
            if isinstance(item, tuple):
                expanded_index.extend(range(item[0], item[1] + 1))
            else:
                expanded_index.append(item)
        index = sorted(set(expanded_index))  # Remove duplicates and sort
    else:
        raise ValueError(f"Invalid index type: {type(index)}")

    # Filter out indices that are out of range
    index = [i for i in index if 1 <= i <= total_ions]

    # If no valid indices remain after filtering, raise an error
    if not index:
        raise ValueError(f"No valid indices found. Valid range is 1 to {total_ions}.")

    # Extract Fermi energy
    efermi = extract_fermi(directory_path)

    # Identify HSE06 or PBE
    kpoints_file_path = os.path.join(directory_path, "KPOINTS")
    kpoints_opt_path = os.path.join(directory_path, "KPOINTS_OPT")
    use_hse06 = os.path.exists(kpoints_opt_path)

    # Extract the number of kpoints
    if use_hse06:
        kpointlist = root.find(".//eigenvalues_kpoints_opt[@comment='kpoints_opt']/kpoints/varray[@name='kpointlist']")
    else:
        kpointlist = root.find(".//varray[@name='kpointlist']")
    kpoints_number = len(kpointlist.findall("v"))

    # Extract eigenvalues and occupancy numbers
    eigen_matrix = []
    occu_matrix = []

    for kpoints_index in range(1, kpoints_number + 1):
        if use_hse06:
            xpath_expr = f"./calculation/projected_kpoints_opt/eigenvalues/array/set/set[@comment='spin 1']/set[@comment='kpoint {kpoints_index}']"
        else:
            xpath_expr = f".//set[@comment='kpoint {kpoints_index}']"

        kpoint_set = root.find(xpath_expr)
        eigen_column = []
        occu_column = []

        for eigen_occ_element in kpoint_set:
            values = list(map(float, eigen_occ_element.text.split()))
            eigen_column.append(values[0])
            occu_column.append(values[1])

        eigen_matrix.append(eigen_column)
        occu_matrix.append(occu_column)

    eigen_matrix = np.array(eigen_matrix).T
    occu_matrix = np.array(occu_matrix).T

    # Extract PDoS components
    energy_pdos_sum = None
    s_pdos_sum = None
    p_y_pdos_sum = None
    p_z_pdos_sum = None
    p_x_pdos_sum = None
    d_xy_pdos_sum = None
    d_yz_pdos_sum = None
    d_z2_pdos_sum = None
    d_zx_pdos_sum = None
    x2_y2_pdos_sum = None

    for ions_index in index:
        path_ions = f".//set[@comment='ion {ions_index}']/set[@comment='spin 1']/r"
        pdos_data = np.array([
            list(map(float, pdos_element.text.split()))
            for pdos_element in root.findall(path_ions)
        ])

        if energy_pdos_sum is None:
            energy_pdos_sum = pdos_data[:, 0]
            s_pdos_sum = pdos_data[:, 1]
            p_y_pdos_sum = pdos_data[:, 2]
            p_z_pdos_sum = pdos_data[:, 3]
            p_x_pdos_sum = pdos_data[:, 4]
            d_xy_pdos_sum = pdos_data[:, 5]
            d_yz_pdos_sum = pdos_data[:, 6]
            d_z2_pdos_sum = pdos_data[:, 7]
            d_zx_pdos_sum = pdos_data[:, 8]
            x2_y2_pdos_sum = pdos_data[:, 9]
        else:
            s_pdos_sum += pdos_data[:, 1]
            p_y_pdos_sum += pdos_data[:, 2]
            p_z_pdos_sum += pdos_data[:, 3]
            p_x_pdos_sum += pdos_data[:, 4]
            d_xy_pdos_sum += pdos_data[:, 5]
            d_yz_pdos_sum += pdos_data[:, 6]
            d_z2_pdos_sum += pdos_data[:, 7]
            d_zx_pdos_sum += pdos_data[:, 8]
            x2_y2_pdos_sum += pdos_data[:, 9]

    total_pdos_list = (
        s_pdos_sum + p_y_pdos_sum + p_z_pdos_sum + p_x_pdos_sum +
        d_xy_pdos_sum + d_yz_pdos_sum + d_z2_pdos_sum + d_zx_pdos_sum + x2_y2_pdos_sum
    )
    energy_pdos_shift = energy_pdos_sum - efermi
    integrated_pdos_list = np.trapz(total_pdos_list, x=energy_pdos_shift)

    return (
        efermi, total_ions, kpoints_number, eigen_matrix, occu_matrix,              # 0 ~ 4
        energy_pdos_shift, total_pdos_list, integrated_pdos_list,                   # 5 ~ 7
        energy_pdos_shift, s_pdos_sum, p_y_pdos_sum, p_z_pdos_sum, p_x_pdos_sum,    # 8 ~ 12
        d_xy_pdos_sum, d_yz_pdos_sum, d_z2_pdos_sum, d_zx_pdos_sum,                 # 13 ~ 16
        x2_y2_pdos_sum                                                              # 17
    )

def extract_dict_pdos(directory_path, index=None):
    """
    Extract PDOS for specified ions and return the result as a dictionary.

    Notes
    - This function depends on extract_index_pdos(directory_path, index).
    - For DOSCAR-based workflows, eigen_matrix and occu_matrix are typically None.
    - This function also reads TOTAL DOS from DOSCAR to provide:
      "total_dos" and "interstitial" (total_dos - total_pdos).

    Parameters
    ----------
    directory_path : str
        Path to the calculation directory.
    index : int, tuple, list, or None
        Ion selection passed to extract_index_pdos.

    Returns
    -------
    dict or None
        Dictionary containing PDOS arrays and useful aliases, or None if an error occurs.
    """
    try:
        pdos_data = extract_index_pdos(directory_path, index)
        if pdos_data is None:
            raise ValueError("extract_index_pdos returned None. Check the directory and indices.")
        if len(pdos_data) < 18:
            raise ValueError(f"extract_index_pdos returned incomplete data. Data length: {len(pdos_data)}")
        total_p_orbitals = pdos_data[10] + pdos_data[11] + pdos_data[12]
        total_d_orbitals = pdos_data[13] + pdos_data[14] + pdos_data[15] + pdos_data[16] + pdos_data[17]
        pdos_dict = {
            "efermi": pdos_data[0],
            "ions_number": pdos_data[1],
            "kpoints_number": pdos_data[2],
            "eigen_matrix": pdos_data[3],
            "occu_matrix": pdos_data[4],
            "dos_shifted_energy": pdos_data[5],
            "total_pdos": pdos_data[6],
            "integrated_pdos": pdos_data[7],
            "pdos_shifted_energy": pdos_data[8],
            "s": pdos_data[9],
            "p": total_p_orbitals,
            "d": total_d_orbitals,
            "p_y": pdos_data[10],
            "p_z": pdos_data[11],
            "p_x": pdos_data[12],
            "d_xy": pdos_data[13],
            "d_yz": pdos_data[14],
            "d_z2": pdos_data[15],
            "d_zx": pdos_data[16],
            "d_x2-y2": pdos_data[17],
        }
        dos_total_data = read_total_dos_from_doscar(directory_path)
        if dos_total_data is not None:
            efermi_dos, energy_dos_shift, total_dos = dos_total_data
            if "efermi" in pdos_dict and abs(pdos_dict["efermi"] - efermi_dos) > 1e-3:
                pass
            if len(energy_dos_shift) == len(pdos_dict["dos_shifted_energy"]):
                pdos_dict["dos_shifted_energy"] = energy_dos_shift
                pdos_dict["total_dos"] = total_dos
                pdos_dict["interstitial"] = total_dos - pdos_dict["total_pdos"]
            else:
                pdos_dict["total_dos"] = total_dos
                pdos_dict["interstitial"] = total_dos - pdos_dict["total_pdos"]
        else:
            pdos_dict["total_dos"] = None
            pdos_dict["interstitial"] = None
        aliases = {
            "total": "total_pdos",
            "integrated": "integrated_pdos",
            "x2-y2": "d_x2-y2",
            "d_x2_y2": "d_x2-y2",
            "d_xz": "d_zx",
        }
        for alias, key in aliases.items():
            pdos_dict[alias] = pdos_dict[key]
        pdos_dict["dos"] = pdos_dict["total_dos"] if pdos_dict.get("total_dos", None) is not None else pdos_dict["total_pdos"]
        return pdos_dict
    except Exception as e:
        print(f"Error in extract_dict_pdos: {e}")
        return None

def extract_dict_pdos_backup(directory_path, index=None):
    """
    Extract PDOS for specified ions and return the result as a dictionary.

    Notes
    - This function depends on extract_index_pdos(directory_path, index).
    - For DOSCAR-based workflows, eigen_matrix and occu_matrix are typically None.

    Parameters
    ----------
    directory_path : str
        Path to the calculation directory.
    index : int, tuple, list, or None
        Ion selection passed to extract_index_pdos.

    Returns
    -------
    dict or None
        Dictionary containing PDOS arrays and useful aliases, or None if an error occurs.
    """
    try:
        pdos_data = extract_index_pdos(directory_path, index)
        if pdos_data is None:
            raise ValueError("extract_index_pdos returned None. Check the directory and indices.")
        if len(pdos_data) < 18:
            raise ValueError(f"extract_index_pdos returned incomplete data. Data length: {len(pdos_data)}")

        total_p_orbitals = pdos_data[10] + pdos_data[11] + pdos_data[12]
        total_d_orbitals = pdos_data[13] + pdos_data[14] + pdos_data[15] + pdos_data[16] + pdos_data[17]

        pdos_dict = {
            "efermi": pdos_data[0],
            "ions_number": pdos_data[1],
            "kpoints_number": pdos_data[2],
            "eigen_matrix": pdos_data[3],
            "occu_matrix": pdos_data[4],
            "dos_shifted_energy": pdos_data[5],
            "total_pdos": pdos_data[6],
            "integrated_pdos": pdos_data[7],
            "pdos_shifted_energy": pdos_data[8],
            "s": pdos_data[9],
            "p": total_p_orbitals,
            "d": total_d_orbitals,
            "p_y": pdos_data[10],
            "p_z": pdos_data[11],
            "p_x": pdos_data[12],
            "d_xy": pdos_data[13],
            "d_yz": pdos_data[14],
            "d_z2": pdos_data[15],
            "d_zx": pdos_data[16],
            "d_x2-y2": pdos_data[17],
        }

        aliases = {
            "total": "total_pdos",
            "integrated": "integrated_pdos",
            "x2-y2": "d_x2-y2",
            "d_x2_y2": "d_x2-y2",
            "d_xz": "d_zx",
        }
        for alias, key in aliases.items():
            pdos_dict[alias] = pdos_dict[key]

        return pdos_dict

    except Exception as e:
        print(f"Error in extract_dict_pdos: {e}")
        return None

def extract_dict_pdos_old(directory_path, index=None):
    """
    Extract PDoS data for specified ions from vasprun.xml and return as a dictionary.

    Parameters:
        directory_path (str): Path to the directory containing vasprun.xml.
        index (int, list of int, tuple, or None): Specific ion index, list of indices, or ranges (tuples).
                                                  If None, extract for all ions.

    Returns:
        dict: Extracted data in a dictionary format, including total PDoS for p and d orbitals.
    """
    try:
        # Normalize the index and fetch raw PDoS data
        pdos_data = extract_index_pdos(directory_path, index)

        # Ensure pdos_data is valid
        if pdos_data is None:
            raise ValueError("extract_index_pdos returned None. Check the directory and indices.")
        if len(pdos_data) < 18:
            raise ValueError(f"extract_index_pdos returned incomplete data. Data length: {len(pdos_data)}")

        # Calculate total PDoS for p and d orbitals
        total_p_orbitals = pdos_data[10] + pdos_data[11] + pdos_data[12]  # p_y, p_z, p_x
        total_d_orbitals = (
            pdos_data[13] + pdos_data[14] + pdos_data[15] + pdos_data[16] + pdos_data[17]  # d_xy, d_yz, d_z2, d_zx, x2-y2
        )

        # Construct the PDoS dictionary
        pdos_dict = {
            "efermi": pdos_data[0],
            "ions_number": pdos_data[1],
            "kpoints_number": pdos_data[2],
            "eigen_matrix": pdos_data[3],
            "occu_matrix": pdos_data[4],
            "dos_shifted_energy": pdos_data[5],
            "total_pdos": pdos_data[6],
            "integrated_pdos": pdos_data[7],
            "pdos_shifted_energy": pdos_data[8],
            "s": pdos_data[9],
            "p": total_p_orbitals,
            "d": total_d_orbitals,
            "p_y": pdos_data[10],
            "p_z": pdos_data[11],
            "p_x": pdos_data[12],
            "d_xy": pdos_data[13],
            "d_yz": pdos_data[14],
            "d_z2": pdos_data[15],
            "d_zx": pdos_data[16],
            "d_x2-y2": pdos_data[17],
        }

        # Add aliases for common terms
        aliases = {
            "total": "total_pdos",
            "integrated": "integrated_pdos",
            "x2-y2": "d_x2-y2",
            "d_x2_y2": "d_x2-y2",
            "d_xz": "d_zx",
        }
        for alias, key in aliases.items():
            pdos_dict[alias] = pdos_dict[key]

        return pdos_dict

    except Exception as e:
        print(f"Error in extract_dict_pdos: {e}")
        return None

# plotting functions
def create_matters_pdos(matters_list):
    """
    Create a structured list of matters for plotting PDoS.
    Parameters:
        matters_list (list): List of configurations for PDoS extraction.
                             Each item is a list containing:
                             [label, directory, atoms, orbitals, line_color, line_style, line_weight, line_alpha].
    Returns:
        list: List of structured matters with extracted PDoS data.
    """
    # Default values for optional parameters
    default_values = {"line_color": "default","line_style": "solid","line_weight": 1.5,"line_alpha": 1.0}
    # Ensure input is a list of lists
    if isinstance(matters_list, list) and matters_list and not any(isinstance(i, list) for i in matters_list):
        source_data = matters_list[:]
        matters_list.clear()
        matters_list.append(source_data)
    structured_matters = []
    for matter_dir in matters_list:
        # Unpack the list with optional parameters
        label, directory, atoms, orbitals, *optional_params = matter_dir
        line_color = optional_params[0] if len(optional_params) > 0 else default_values["line_color"]
        line_style = optional_params[1] if len(optional_params) > 1 else default_values["line_style"]
        line_weight = optional_params[2] if len(optional_params) > 2 else default_values["line_weight"]
        line_alpha = optional_params[3] if len(optional_params) > 3 else default_values["line_alpha"]
        # Extract PDoS data
        pdos_data = extract_dict_pdos(directory, atoms)
        if pdos_data is None:
            print(f"Warning: Failed to extract PDoS data for {label} in directory {directory}. Skipping...")
            continue
        # Append structured matter list
        structured_matters.append([label, pdos_data, atoms, orbitals, line_color, line_style, line_weight, line_alpha])
    return structured_matters

def plot_pdos(title, *args, x_range=None, y_top=None):
    # General function to plot PDoS for one or multiple systems.
    help_info = """
    Usage: plot_pdos(title, *args, x_range=None, y_top=None)
    Example for single PDoS plot:
        systems = [["label", "path/to/directory", [indices], "orbital", "color", "linestyle", linewidth, alpha]]
        plot_pdos("Title", systems, x_range=6, y_top=12)
    Orbital labels include:
        - total_pdos (or total)
        - integrated_pdos (or integrated)
        - s, p, d, etc.
    """
    if title in ["help", "Help"]:
        print(help_info)
        return
    if not args:
        raise ValueError("At least one system must be provided in *args.")
    if len(args) == 1:
        return plot_single_pdos(title, matters_list=args[0], x_range=x_range, y_top=y_top)
    else:
        raise NotImplementedError("Multi-system PDoS plotting is not yet supported.")

def plot_single_pdos(title, matters_list=None, x_range=None, y_top=None):
    """
    Plot PDoS for a single system with individual settings for each orbital.

    Parameters:
        title (str): Title of the plot.
        x_range (float): Range of the x-axis (energy range).
        y_top (float): Maximum value of the y-axis.
        matters_list (list): List of configurations for each orbital. Each item is a list containing:
                             [label, directory, atoms, orbital, line_color, line_style, line_weight, line_alpha].
    """
    # Validate input
    if not matters_list or len(matters_list) == 0:
        raise ValueError("matters_list must contain at least one configuration.")

    # Figure Settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Color for Fermi energy line
    fermi_color = color_sampling("Violet")

    # Plot PDoS for each orbital with individual settings
    for matter in matters_list:
        label, directory, atoms, orbital, line_color, line_style, line_weight, line_alpha = matter
        # Extract PDoS data for this system
        pdos_data = extract_dict_pdos(directory, atoms)
        if pdos_data is None:
            print(f"Warning: Skipping {label} because PDoS data could not be extracted.")
            continue
        # Ensure the orbital exists in the PDoS data
        if orbital not in pdos_data:
            print(f"Warning: Orbital '{orbital}' not found for {label}. Skipping...")
            continue
        energy = pdos_data["pdos_shifted_energy"]
        plt.plot(energy, pdos_data[orbital],
                 color=color_sampling(line_color)[1],
                 linestyle=line_style, linewidth=line_weight, alpha=line_alpha,
                 label=f"{label}")
        
        # Total DoS
        # efermi, e_shift, dos_total = read_total_dos_from_doscar(directory)
        # pdos_total = pdos_data["total_pdos"]
        # interstitial = dos_total - pdos_total
        # plt.plot(e_shift, dos_total,  color=color_sampling(line_color)[1], linestyle=line_style, linewidth=line_weight, alpha=line_alpha,label="Total DOS")
        # plt.plot(e_shift, interstitial,  color=color_sampling(line_color)[1], linestyle=line_style, linewidth=line_weight, alpha=line_alpha, label="Interstitial (DOS - PDOS)")

    # Add Fermi energy line
    plt.axvline(x=0, linestyle="--", color=fermi_color[0], alpha=0.8, label="Fermi energy")
    if matters_list:  # Use the first valid pdos_data for Fermi energy annotation
        fermi_energy_text = f"Fermi energy\n({pdos_data['efermi']:.3f} eV)"
        plt.text(-x_range * 0.02, y_top * 0.98, fermi_energy_text,
                 fontsize=12, color=fermi_color[0], rotation=0, va="top", ha="right")

    # Plot settings
    plt.title(title)
    plt.xlabel("Energy (eV)")
    plt.ylabel("Density of States")
    plt.xlim(-x_range, x_range)
    plt.ylim(0, y_top)
    # plt.legend(loc="upper right")
    plt.legend(loc="best")
    plt.tight_layout()
    # plt.show()
