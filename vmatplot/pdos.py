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

def cal_type_pdos(directory_path):
    kpoints_file_path = os.path.join(directory_path, "KPOINTS")
    kpoints_opt_path = os.path.join(directory_path, "KPOINTS_OPT")
    if os.path.exists(kpoints_opt_path):
        return "GGA-PBE"
    elif os.path.exists(kpoints_file_path):
        return "HSE06"

def count_pdos_atoms(directory_path):
    """
    Get the total number of atoms (ions) from the vasprun.xml file in the specified folder.
    Parameters:
        directory_path (str): Path to the directory containing vasprun.xml.
    Returns:
        int: Total number of atoms.
    Raises:
        FileNotFoundError: If the vasprun.xml file does not exist.
        ValueError: If the atom count cannot be determined.
    """
    # Construct the path to vasprun.xml
    file_path = os.path.join(directory_path, "vasprun.xml")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file vasprun.xml does not exist in the directory: {directory_path}")
    # Parse vasprun.xml
    tree = ET.parse(file_path)
    root = tree.getroot()
    # Locate the <positions> section to count atoms
    positions_section = root.find(".//varray[@name='positions'][1]")
    if positions_section is None:
        raise ValueError("Failed to locate the positions section in vasprun.xml.")
    # Count the number of <v> elements in the positions section
    atom_count = len(positions_section.findall("v"))
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
def extract_pdos(directory_path):
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
def extract_segment_pdos(directory_path, start, end = None):
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
    # Construct the full path to the vasprun.xml file
    file_path = os.path.join(directory_path, "vasprun.xml")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Error: The file vasprun.xml does not exist in the directory {directory_path}.")

    # Parse the vasprun.xml file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Get total number of atoms
    total_ions = count_pdos_atoms(directory_path)

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

def extract_index_pdos(directory_path, index=None):
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

    # Get total number of atoms
    def count_pdos_atoms(directory_path):
        """Count total atoms in vasprun.xml"""
        positions_section = root.find(".//varray[@name='positions'][1]")
        if positions_section is None:
            raise ValueError("Failed to locate the positions section in vasprun.xml.")
        return len(positions_section.findall("v"))

    total_ions = count_pdos_atoms(directory_path)

    # Handle index logic: Normalize index to a list of integers
    if index in (None, [], "All", "all", "Total", "total"):  # Special cases: all ions
        index = list(range(1, total_ions + 1))
    elif isinstance(index, int):  # Single integer
        index = [index]
    elif isinstance(index, tuple):  # Single tuple (range)
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

def extract_dict_pdos(directory_path, index=None):
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
                 color=color_sampling(line_color)[0],
                 linestyle=line_style, linewidth=line_weight, alpha=line_alpha,
                 label=f"{label}")

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
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
