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

def cal_type(directory_path):
    kpoints_file_path = os.path.join(directory_path, "KPOINTS")
    kpoints_opt_path = os.path.join(directory_path, "KPOINTS_OPT")
    if os.path.exists(kpoints_opt_path):
        return "GGA-PBE"
    elif os.path.exists(kpoints_file_path):
        return "HSE06"

def extract_dos(directory_path):
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

        # plt.legend(loc="best")
        plt.legend(loc="upper right")
        plt.tight_layout()
