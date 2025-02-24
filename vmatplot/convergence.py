#### Convergence test
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0914

import xml.etree.ElementTree as ET
import os

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from vmatplot.commons import check_vasprun, identify_parameters, get_or_default
from vmatplot.algorithms import is_nested_list, fit_birch_murnaghan
from vmatplot.output_settings import canvas_setting, color_sampling

## Process and calculate data

def cal_cohesive_energy(atom_count, atom_energy, total_energy):
    cohesive_energy = (total_energy - atom_count * atom_energy)/atom_count
    return cohesive_energy

def summarize_parameters(directory=".", lattice_boundary=None):
    result_file = "energy_parameters.dat"
    result_file_path = os.path.join(directory, result_file)

    if directory.lower in ["help"]:
        print("Please use this function on the parent directory of the project's main folder.")
        return []

    if not os.path.exists(directory):
        os.makedirs(directory)

    dirs_to_walk = check_vasprun(directory)
    results = []
    TOLERANCE = 1e-6

    for work_dir in dirs_to_walk:
        params = identify_parameters(work_dir)
        if params:
            # Debugging output for extracted parameters
            # print(f"Extracted parameters from {work_dir}: {params}")
            # Lattice boundary check
            lattice_constant = params.get("lattice constant")
            if lattice_boundary is not None:
                lattice_start, lattice_end = lattice_boundary
                within_start = lattice_start in [None, ""] or lattice_constant >= lattice_start - TOLERANCE
                within_end = lattice_end in [None, ""] or lattice_constant <= lattice_end + TOLERANCE
                if not (within_start and within_end):
                    print(f"Skipping {work_dir} due to lattice boundary constraints.")
                    continue

            results.append(params)

    # Check if results are empty before proceeding to write the output
    if not results:
        print("No data extracted. Please check your directory and lattice boundary settings.")
        return results

    # Sort results by elapsed time (sec) if available
    results.sort(key=lambda x: x.get("elapsed time (sec)", float('inf')))

    # Write the sorted results to the output file
    try:
        with open(result_file_path, "w", encoding="utf-8") as f:
            # Write headers based on keys in the first result dictionary
            headers = "\t".join(results[0].keys())
            f.write(headers + "\n")
            for result in results:
                f.write("\t".join(str(result[key]) if result[key] is not None else 'None' for key in result) + "\n")
    except IOError as e:
        print(f"Error writing to file at {result_file_path}: {e}")

    return results

import os

def summarize_cohesive_energy_for_specific(full_cal, atom_cal):
    """
    Calculate the cohesive energy for a specific pair of VASP calculation directories.

    Parameters:
        full_cal (str): Path to the full calculation directory.
        atom_cal (str): Path to the atom calculation directory.

    Returns:
        dict: A dictionary containing:
              - "total atom count": Total number of atoms from the full calculation.
              - "total energy": Total energy from the full calculation.
              - "atom energy": Total energy from the atom calculation.
              - "cohesive energy": Computed cohesive energy.
              - "total kpoints": Total number of k-points.
              - "kpoints mesh": k-points mesh dimensions.
              - "energy cutoff (ENCUT)": Energy cutoff value.
              - "Scaling": Scaling factor.
              - "lattice constant": Lattice constant.
              
              Returns None if the parameters cannot be extracted.
    """
    # Extract parameters using identify_parameters.
    full_params = identify_parameters(full_cal)
    atom_params = identify_parameters(atom_cal)
    
    if full_params is None or atom_params is None:
        print("Error: One or both directories do not contain valid VASP calculation data.")
        return None

    try:
        # Compute cohesive energy: full energy minus (total atom count multiplied by atom energy).
        cohesive_energy = full_params["total energy"] - full_params["total atom count"] * atom_params["total energy"]
    except KeyError as e:
        print(f"Error: Missing key in parameters: {e}")
        return None

    result = {
        "total atom count": full_params["total atom count"],
        "total energy": full_params["total energy"],
        "atom energy": atom_params["total energy"],
        "cohesive energy": cohesive_energy,
        "total kpoints": full_params["total kpoints"],
        "kpoints mesh": full_params["kpoints mesh"],
        "energy cutoff (ENCUT)": full_params.get("energy cutoff (ENCUT)"),
        "Scaling": full_params.get("Scaling"),
        "lattice constant": full_params.get("lattice constant")
    }
    
    return result

def summarize_cohesive_energy_for_supdir(full_cal_dir, atom_cal_dir):
    """
    Calculate cohesive energies when both full_cal_dir and atom_cal_dir are parent directories.

    The function iterates through subdirectories in both parent directories, matches the corresponding
    full and atom calculation data, computes the cohesive energy for each matched pair, and writes the results
    to 'cohesive_energy.dat' in the atom_cal_dir.
    """
    result_file = "cohesive_energy.dat"
    result_file_path = os.path.join(atom_cal_dir, result_file)

    full_cal_data = []
    atom_cal_data = []

    for work_dir in check_vasprun(full_cal_dir):
        params = identify_parameters(work_dir)
        if params:
            full_cal_data.append(params)

    for work_dir in check_vasprun(atom_cal_dir):
        params = identify_parameters(work_dir)
        if params:
            atom_cal_data.append(params)

    results = []
    for full_params in full_cal_data:
        for atom_params in atom_cal_data:
            # Matching conditions: kpoints mesh, total kpoints, energy cutoff, and Scaling.
            if (full_params.get("kpoints mesh") == atom_params.get("kpoints mesh") and
                full_params.get("total kpoints") == atom_params.get("total kpoints") and
                full_params.get("energy cutoff (ENCUT)") == atom_params.get("energy cutoff (ENCUT)") and
                full_params.get("Scaling") == atom_params.get("Scaling")):
                
                cohesive_energy = full_params["total energy"] - full_params["total atom count"] * atom_params["total energy"]
                result = {
                    "total atom count": full_params["total atom count"],
                    "total energy": full_params["total energy"],
                    "atom energy": atom_params["total energy"],
                    "cohesive energy": cohesive_energy,
                    "total kpoints": full_params["total kpoints"],
                    "kpoints mesh": full_params["kpoints mesh"],
                    "energy cutoff (ENCUT)": full_params.get("energy cutoff (ENCUT)"),
                    "Scaling": full_params.get("Scaling"),
                    "lattice constant": full_params.get("lattice constant")
                }
                results.append(result)

    if not results:
        print("No matching data found. Please check your directories.")
        return results

    results.sort(key=lambda x: (x["total kpoints"], x.get("energy cutoff (ENCUT)", 0)))

    try:
        with open(result_file_path, "w", encoding="utf-8") as f:
            headers = "\t".join(results[0].keys())
            f.write(headers + "\n")
            for result in results:
                f.write("\t".join(str(result[key]) if result[key] is not None else "None" for key in results[0].keys()) + "\n")
    except IOError as e:
        print(f"Error writing to file at {result_file_path}: {e}")

    return results

def summarize_cohesive_energy_mixed(full_cal_supdir, atom_cal_dir):
    """
    Calculate cohesive energies when full_cal_supdir is a parent directory containing multiple
    full calculation subdirectories, and atom_cal_dir is a specific directory.

    For each subdirectory in full_cal_supdir, the function computes the cohesive energy using
    the atom calculation data from the single atom_cal_dir. The results are written to
    'cohesive_energy.dat' in the full_cal_supdir.

    Parameters:
        full_cal_supdir (str): Parent directory containing full calculation subdirectories.
        atom_cal_dir (str): Specific atom calculation directory.

    Returns:
        list: A list of dictionaries, each containing:
              - "total atom count", "total energy", "atom energy", "cohesive energy",
              - "total kpoints", "kpoints mesh", "energy cutoff (ENCUT)", "Scaling", and "lattice constant".
    """
    result_file = "cohesive_energy.dat"
    result_file_path = os.path.join(full_cal_supdir, result_file)

    full_cal_data = []
    for work_dir in check_vasprun(full_cal_supdir):
        params = identify_parameters(work_dir)
        if params:
            full_cal_data.append(params)

    atom_params = identify_parameters(atom_cal_dir)
    if atom_params is None:
        print("Error: Atom calculation directory does not contain valid VASP data.")
        return None

    results = []
    for full_params in full_cal_data:
        try:
            cohesive_energy = full_params["total energy"] - full_params["total atom count"] * atom_params["total energy"]
        except KeyError as e:
            print(f"Error: Missing key in parameters for a directory: {e}")
            continue

        result = {
            "total atom count": full_params["total atom count"],
            "total energy": full_params["total energy"],
            "atom energy": atom_params["total energy"],
            "cohesive energy": cohesive_energy,
            "total kpoints": full_params["total kpoints"],
            "kpoints mesh": full_params["kpoints mesh"],
            "energy cutoff (ENCUT)": full_params.get("energy cutoff (ENCUT)"),
            "Scaling": full_params.get("Scaling"),
            "lattice constant": full_params.get("lattice constant")
        }
        results.append(result)

    if not results:
        print("No valid full calculation data found. Please check your full calculation directory.")
        return results

    results.sort(key=lambda x: (x["total kpoints"], x.get("energy cutoff (ENCUT)", 0)))

    try:
        with open(result_file_path, "w", encoding="utf-8") as f:
            headers = "\t".join(results[0].keys())
            f.write(headers + "\n")
            for result in results:
                f.write("\t".join(str(result[key]) if result[key] is not None else "None" for key in results[0].keys()) + "\n")
    except IOError as e:
        print(f"Error writing to file at {result_file_path}: {e}")

    return results

def summarize_cohesive_energy(full_cal_dir, atom_cal_dir):
    """
    Automatically choose the appropriate cohesive energy summarization method based on the type of directories.

    - If both directories are specific calculation directories (i.e., containing 'vasprun.xml'),
      use summarize_cohesive_energy_for_specific.
    - If both directories are parent directories (i.e., no 'vasprun.xml' in the root), use
      summarize_cohesive_energy_for_supdir.
    - If full_cal_dir is a parent directory and atom_cal_dir is a specific directory, use
      summarize_cohesive_energy_mixed.
    - Otherwise, print an error message.

    Parameters:
        full_cal_dir (str): Path to the full calculation directory or parent directory.
        atom_cal_dir (str): Path to the atom calculation directory or parent directory.

    Returns:
        dict or list: The cohesive energy results as computed by the appropriate routine.
    """
    full_cal_specific = os.path.isfile(os.path.join(full_cal_dir, "vasprun.xml"))
    atom_cal_specific = os.path.isfile(os.path.join(atom_cal_dir, "vasprun.xml"))
    
    if full_cal_specific and atom_cal_specific:
        return summarize_cohesive_energy_for_specific(full_cal_dir, atom_cal_dir)
    elif (not full_cal_specific) and (not atom_cal_specific):
        return summarize_cohesive_energy_for_supdir(full_cal_dir, atom_cal_dir)
    elif (not full_cal_specific) and atom_cal_specific:
        return summarize_cohesive_energy_mixed(full_cal_dir, atom_cal_dir)
    else:
        print("Error: The combination of directories provided is not supported.")
        return None

def read_energy_parameters(data_path):
    help_info = "Usage: read_energy_parameters(data_path)\n" + \
                "data_path: Path to the data file containing energy and various VASP parameters.\n"

    # Check if the user asked for help
    if data_path.lower in ["help"]:
        print(help_info)
        return

    # Initialize a list to store the dictionary for each line in the file
    parameter_dicts = []

    # Open the data file and process each line as a dictionary entry
    with open(data_path, "r", encoding="utf-8") as data_file:
        # Read the header for keys
        headers = data_file.readline().strip().split("\t")

        # Define mappings for old and new .dat formats
        field_mappings = {
            "total atom count": "total atom count",
            "total energy": "Total Energy",
            "total kpoints": "Total Kpoints",
            "kpoints mesh": "Kpoints(X Y Z)",
            "lattice constant": "Lattice Constant",
            "symmetry precision (SYMPREC)": "SYMPREC",
            "energy cutoff (ENCUT)": "ENCUT",
            "k-point spacing (KSPACING)": "KSPACING",
            "volume": "VOLUME",
            "time step (POTIM)": "POTIM",
            "mixing parameter (AMIX)": "AMIX",
            "mixing parameter (BMIX)": "BMIX",
            "electronic convergence (EDIFF)": "EDIFF",
            "force convergence (EDIFFG)": "EDIFFG",
            "elapsed time (sec)": "elapsed time (sec)"
        }

        # Reverse mapping to accommodate both old and new .dat formats
        reverse_mapping = {v: k for k, v in field_mappings.items()}
        standardized_headers = [reverse_mapping.get(header, header) for header in headers]

        for line in data_file:
            values = line.strip().split("\t")
            entry = {}
            for header, value in zip(standardized_headers, values):
                # Convert "None" to None, tuples to tuples, and other values to float if possible
                if value == 'None':
                    entry[header] = None
                elif header == "kpoints mesh":  # Check for kpoints mesh in (x, y, z) format
                    entry[header] = tuple(map(int, value.strip("()").split(", ")))
                else:
                    try:
                        entry[header] = float(value)
                    except ValueError:
                        entry[header] = value  # Keep it as string if conversion fails
            parameter_dicts.append(entry)

    return parameter_dicts

## Energy versus parameters

def plot_energy_kpoints_single(suptitle, *args_list):
    help_info = "Usage: plot_energy_kpoints(suptitle, args_list)\n" + \
                "args_list: A list containing [info_suffix, source_data, kpoints_boundary, color_family].\n" + \
                "Example: plot_energy_kpoints(suptitle, ['Material Info', 'source_data_path', (start, end), 'blue'])\n"

    if not args_list or args_list[0] in ["HELP", "Help", "help"]:
        print(help_info)
        return

    # Unpack args_list
    # info_suffix, source_data, kpoints_boundary, color_family = args_list[0]
    args_info = args_list[0] + [None] * (7 - len(args_list[0]))

    # Unpack with ensured length
    info_suffix, source_data, kpoints_boundary, color_family, line_style, line_weight, line_alpha = args_info

    # Apply default values using get_or_default
    color_family = get_or_default(color_family, "default")
    line_style = get_or_default(line_style, "solid")
    line_weight = get_or_default(line_weight, 1.5)
    line_alpha = get_or_default(line_alpha, 1.0)

    # Figure Settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Apply ScalarFormatter with scientific notation limits
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    # Color calling
    colors = color_sampling(color_family)

    # Data input
    data_dict_list = read_energy_parameters(source_data)  # Now a list of dictionaries

    # Extract values based on keys in the dictionary
    total_kpoints = [d.get("total kpoints") for d in data_dict_list]
    energy = [d.get("total energy") for d in data_dict_list]
    sep_kpoints = [d.get("kpoints mesh") for d in data_dict_list]

    # Sort data based on total_kpoints to maintain order
    sorted_data = sorted(zip(total_kpoints, energy, sep_kpoints), key=lambda x: x[0])
    total_kpoints_sorted, energy_sorted, sep_kpoints_sorted = zip(*sorted_data)

    # Set title with info_suffix
    plt.title(f"{suptitle} {info_suffix}")
    plt.xlabel("K-points configuration")
    plt.ylabel("Energy (eV)")

    # Ensure boundary values are integers and handle None boundaries
    kpoints_start = min(total_kpoints_sorted) if not kpoints_boundary or kpoints_boundary[0] is None else int(kpoints_boundary[0])
    kpoints_end = max(total_kpoints_sorted) if not kpoints_boundary or kpoints_boundary[1] is None else int(kpoints_boundary[1])

    # Filter data within the specified boundary
    kpoints_indices = [
        index for index, val in enumerate(total_kpoints_sorted) if kpoints_start <= val <= kpoints_end
    ]
    total_kpoints_plot = [total_kpoints_sorted[index] for index in kpoints_indices]
    energy_plotting = [energy_sorted[index] for index in kpoints_indices]
    kpoints_labels_plot = [f"({x[0]}, {x[1]}, {x[2]})" for x in [sep_kpoints_sorted[index] for index in kpoints_indices]]

    # Plot with fixed spacing based on indices
    plt.plot(range(len(total_kpoints_plot)), energy_plotting, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha, label=f"Energy versus K-points {info_suffix}")
    plt.scatter(range(len(total_kpoints_plot)), energy_plotting, c=colors[1], s=line_weight*4, lw=line_weight, alpha=line_alpha, zorder=1)

    # Set custom tick labels for x-axis to show kpoints configurations
    plt.xticks(ticks=range(len(kpoints_labels_plot)), labels=kpoints_labels_plot, rotation=45, ha="right")

    plt.tight_layout()

def plot_energy_kpoints(suptitle, kpoints_list):
    """
    Generalized function to plot energy versus K-points configuration for multiple datasets.

    Parameters:
    - kpoints_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, kpoints_boundary, color_family, line_style, line_weight, line_alpha].
    """

    # Check if input is a single data set (either single list or a list with one sublist)
    if is_nested_list(kpoints_list) is False:
        return plot_energy_kpoints_single(suptitle, kpoints_list)
    elif isinstance(kpoints_list[0], list) and len(kpoints_list) == 1:
        return plot_energy_kpoints_single(suptitle, *kpoints_list)
    else: pass

    # Check if kpoints_list is a 2D list structure for multiple datasets
    for index, data in enumerate(kpoints_list):
        if not isinstance(data, list):
            print(f"Error: Item at index {index} in kpoints_list must be a list.")
            return
        if len(data) < 4:
            print(f"Warning: Item at index {index} has less than 4 elements. Missing elements will be filled with None.")
            kpoints_list[index] += [None] * (7 - len(data))

    # Set up figure and styling
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Apply scientific notation formatter for y-axis
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    # List to store legend handles for custom legend creation
    legend_handles = []
    global_kpoints_config = []  # List to collect unique kpoints configurations for x-axis labels
    total_kpoints_set = set()  # Collect unique total_kpoints across all datasets for sorting

    # Iterate over each dataset in kpoints_list to gather total_kpoints and kpoints configurations
    kpoints_config_map = {}
    for data in kpoints_list:
        # Unpack arguments and ensure length
        args_info = data + [None] * (7 - len(data))
        info_suffix, source_data, kpoints_boundary, color_family, line_style, line_weight, line_alpha = args_info

        # Apply default values using get_or_default
        color_family = get_or_default(color_family, "default")
        line_style = get_or_default(line_style, "solid")
        line_weight = get_or_default(line_weight, 1.5)
        line_alpha = get_or_default(line_alpha, 1.0)

        data_dict_list = read_energy_parameters(source_data)  # Now returns list of dicts
        total_kpoints = [d.get("total kpoints") for d in data_dict_list]
        kpoints_config = [d.get("kpoints mesh") for d in data_dict_list]  # Extract kpoints configurations in (X, Y, Z) format

        # Apply boundary to filter relevant k-points for each dataset
        kpoints_start = min(total_kpoints) if not kpoints_boundary or kpoints_boundary[0] is None else int(kpoints_boundary[0])
        kpoints_end = max(total_kpoints) if not kpoints_boundary or kpoints_boundary[1] is None else int(kpoints_boundary[1])

        # Filtered kpoints within boundary and update global sets
        for idx, total_kp in enumerate(total_kpoints):
            if kpoints_start <= total_kp <= kpoints_end:
                total_kpoints_set.add(total_kp)
                kpoints_config_map[total_kp] = f"({kpoints_config[idx][0]}, {kpoints_config[idx][1]}, {kpoints_config[idx][2]})"

    # Sort the global total_kpoints for unified x-axis
    sorted_total_kpoints = sorted(total_kpoints_set)
    global_kpoints_config = [kpoints_config_map[total_kp] for total_kp in sorted_total_kpoints]

    # Iterate over each dataset to align with global total_kpoints and plot
    for data in kpoints_list:
        # Unpack arguments and ensure length
        args_info = data + [None] * (7 - len(data))
        info_suffix, source_data, kpoints_boundary, color_family, line_style, line_weight, line_alpha = args_info

        # Apply default values using get_or_default
        color_family = get_or_default(color_family, "default")
        line_style = get_or_default(line_style, "solid")
        line_weight = get_or_default(line_weight, 1.5)
        line_alpha = get_or_default(line_alpha, 1.0)

        # Color for the current dataset
        colors = color_sampling(color_family)

        # Data input
        data_dict_list = read_energy_parameters(source_data)  # Now a list of dictionaries
        energy = [d.get("total energy") for d in data_dict_list]
        total_kpoints = [d.get("total kpoints") for d in data_dict_list]

        # Apply boundary values for the current dataset
        kpoints_start = min(total_kpoints) if not kpoints_boundary or kpoints_boundary[0] is None else int(kpoints_boundary[0])
        kpoints_end = max(total_kpoints) if not kpoints_boundary or kpoints_boundary[1] is None else int(kpoints_boundary[1])

        # Filter data within the specified boundary for the current dataset
        filtered_kpoints = [total_kp for total_kp in total_kpoints if kpoints_start <= total_kp <= kpoints_end]
        filtered_energy = [energy[idx] for idx, total_kp in enumerate(total_kpoints) if kpoints_start <= total_kp <= kpoints_end]

        # Align energies with the global sorted total_kpoints
        energy_aligned = [filtered_energy[filtered_kpoints.index(total_kp)] if total_kp in filtered_kpoints else np.nan for total_kp in sorted_total_kpoints]

        # Plotting with color, style, and unique label
        plt.plot(range(len(sorted_total_kpoints)), energy_aligned, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha)
        plt.scatter(range(len(sorted_total_kpoints)), energy_aligned, s=line_weight * 4, c=colors[1], alpha=line_alpha, zorder=1)

        # Add legend entry with custom handle
        legend_handle = mlines.Line2D([], [], color=colors[1], marker='o', markersize=6, linestyle=line_style,
                                      label=f"Energy versus K-points {info_suffix}")
        legend_handles.append(legend_handle)

    # Set labels and legend for multi-dataset
    plt.xlabel("K-points configurations")
    plt.ylabel("Energy (eV)")
    plt.xticks(ticks=range(len(global_kpoints_config)), labels=global_kpoints_config, rotation=45, ha="right")
    plt.title(f"{suptitle}")
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

def plot_energy_encut_single(suptitle, *args_list):
    help_info = (
        "Usage: plot_energy_encut(suptitle, args_list)\n"
        "args_list: A list containing [info_suffix, source_data, encut_boundary, color_family, line_style, line_weight, line_alpha].\n"
        "Example: plot_energy_encut(suptitle, ['Material Info', 'source_data_path', (start, end), 'violet', 'dashed', 2.0, 0.8])\n"
    )

    if not args_list or args_list[0] in ["HELP", "Help", "help"]:
        print(help_info)
        return

    # Unpack args_list and fill missing values with None
    args_info = args_list[0] + [None] * (7 - len(args_list[0]))
    info_suffix, source_data, encut_boundary, color_family, line_style, line_weight, line_alpha = args_info

    # Apply default values using get_or_default
    color_family = get_or_default(color_family, "default")
    line_style = get_or_default(line_style, "solid")
    line_weight = get_or_default(line_weight, 1.5)
    line_alpha = get_or_default(line_alpha, 1.0)

    # Figure Settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Apply ScalarFormatter with scientific notation limits
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    # Color calling
    colors = color_sampling(color_family)

    # Data input
    data_dict_list = read_energy_parameters(source_data)  # Now returns list of dicts

    # Extract values based on keys in the dictionary
    energy = [d.get("total energy", d.get("Total Energy")) for d in data_dict_list]
    encut_values = [d.get("energy cutoff (ENCUT)", d.get("ENCUT")) for d in data_dict_list]

    # Filter out None values from energy and ENCUT values
    filtered_data = [(enc, e) for enc, e in zip(encut_values, energy) if enc is not None and e is not None]
    if not filtered_data:
        print("No valid data found for plotting.")
        return

    # Unzip filtered data for ENCUT and energy
    encut_values, energy = zip(*filtered_data)

    # Sort data based on ENCUT values
    sorted_data = sorted(zip(encut_values, energy), key=lambda x: x[0])
    encut_sorted, energy_sorted = zip(*sorted_data)

    # Set title with info_suffix
    # plt.title(f"Energy versus energy cutoff {info_suffix}")
    plt.title(f"{suptitle} {info_suffix}")
    plt.xlabel("Energy cutoff (eV)")
    plt.ylabel("Energy (eV)")

    # Set boundaries for ENCUT
    encut_start = min(encut_sorted) if not encut_boundary or encut_boundary[0] is None else float(encut_boundary[0])
    encut_end = max(encut_sorted) if not encut_boundary or encut_boundary[1] is None else float(encut_boundary[1])

    # Filter data within the specified boundary
    encut_plotting = [enc for enc in encut_sorted if encut_start <= enc <= encut_end]
    energy_plotting = [energy_sorted[idx] for idx, enc in enumerate(encut_sorted) if encut_start <= enc <= encut_end]

    # Plotting
    plt.plot(encut_plotting, energy_plotting, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha)
    plt.scatter(encut_plotting, energy_plotting, s=line_weight * 4, c=colors[1], zorder=1)

    plt.tight_layout()

def plot_energy_encut(suptitle, encut_list):
    """
    Generalized function to plot energy versus ENCUT (energy cutoff) for multiple datasets.

    Parameters:
    - encut_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, encut_boundary, color_family, line_style, line_weight, line_alpha].
    """

    # Check if input is a single data set (either single list or a list with one sublist)
    if is_nested_list(encut_list) is False:
        return plot_energy_encut_single(suptitle, encut_list)
    elif isinstance(encut_list[0], list) and len(encut_list) == 1:
        return plot_energy_encut_single(suptitle, *encut_list)

    # Check if encut_list is a 2D list structure for multiple datasets
    for index, data in enumerate(encut_list):
        if not isinstance(data, list):
            print(f"Error: Item at index {index} in encut_list must be a list.")
            return
        if len(data) < 4:
            print(f"Warning: Item at index {index} has less than 4 elements. Missing elements will be filled with None.")
            encut_list[index] += [None] * (7 - len(data))

    # Set up figure and styling
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Apply scientific notation formatter for y-axis
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    # List to store legend handles for custom legend creation
    legend_handles = []
    all_encut_set = set()

    # Gather all unique ENCUT values
    for data in encut_list:
        args_info = data + [None] * (7 - len(data))
        info_suffix, source_data, encut_boundary, color_family, line_style, line_weight, line_alpha = args_info

        # Apply default values
        color_family = get_or_default(color_family, "default")
        line_style = get_or_default(line_style, "solid")
        line_weight = get_or_default(line_weight, 1.5)
        line_alpha = get_or_default(line_alpha, 1.0)

        data_dict_list = read_energy_parameters(source_data)
        encut_values = [d.get("energy cutoff (ENCUT)", d.get("ENCUT")) for d in data_dict_list]
        encut_start = min(encut_values) if not encut_boundary or encut_boundary[0] is None else float(encut_boundary[0])
        encut_end = max(encut_values) if not encut_boundary or encut_boundary[1] is None else float(encut_boundary[1])
        filtered_encut = [enc for enc in encut_values if encut_start <= enc <= encut_end]
        all_encut_set.update(filtered_encut)

    # Sort ENCUT values globally
    all_encut_sorted = sorted(all_encut_set)

    # Plot datasets
    for data in encut_list:
        args_info = data + [None] * (7 - len(data))
        info_suffix, source_data, encut_boundary, color_family, line_style, line_weight, line_alpha = args_info

        # Apply default values
        color_family = get_or_default(color_family, "default")
        line_style = get_or_default(line_style, "solid")
        line_weight = get_or_default(line_weight, 1.5)
        line_alpha = get_or_default(line_alpha, 1.0)

        data_dict_list = read_energy_parameters(source_data)
        energy = [d.get("total energy", d.get("Total Energy")) for d in data_dict_list]
        encut_values = [d.get("energy cutoff (ENCUT)", d.get("ENCUT")) for d in data_dict_list]

        encut_start = min(encut_values) if not encut_boundary or encut_boundary[0] is None else float(encut_boundary[0])
        encut_end = max(encut_values) if not encut_boundary or encut_boundary[1] is None else float(encut_boundary[1])
        filtered_encut = [enc for enc in encut_values if encut_start <= enc <= encut_end]
        filtered_energy = [energy[idx] for idx, enc in enumerate(encut_values) if encut_start <= enc <= encut_end]

        energy_aligned = [filtered_energy[filtered_encut.index(enc)] if enc in filtered_encut else np.nan for enc in all_encut_sorted]

        colors = color_sampling(color_family)
        plt.plot(all_encut_sorted, energy_aligned, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha)
        plt.scatter(all_encut_sorted, energy_aligned, s=line_weight * 4, c=colors[1], zorder=1)

        legend_handle = mlines.Line2D([], [], color=colors[1], marker='o', markersize=6, linestyle=line_style,
                                      label=f"Energy versus energy cutoff {info_suffix}")
        legend_handles.append(legend_handle)

    plt.xlabel("Energy cutoff (eV)")
    plt.ylabel("Energy (eV)")
    # plt.title("Energy versus energy cutoff")
    plt.title(f"{suptitle}")
    
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

def plot_energy_lattice_single(suptitle, *args_list):
    help_info = (
        "Usage: plot_energy_lattice_single(suptitle, args_list)\n"
        "args_list: A list containing [info_suffix, source_data, lattice_boundary, num_samples, color_family, line_style, line_weight, line_alpha].\n"
        "Example: plot_energy_lattice_single(suptitle, ['Material Info', 'source_data_path', (start, end), 11, 'green', 'dashed', 2.0, 0.8])\n"
    )

    # Check if the user requested help
    if not args_list or args_list[0] in ["HELP", "Help", "help"]:
        print(help_info)
        return

    # Unpack args_list and handle missing values
    args_info = args_list[0] + [None] * (8 - len(args_list[0]))
    info_suffix, source_data, lattice_boundary, num_samples, color_family, line_style, line_weight, line_alpha = args_info

    # Validate `num_samples`
    if num_samples is None or not isinstance(num_samples, int) or num_samples <= 0:
        num_samples = 100  # Default sample count

    # Apply default values using `get_or_default`
    color_family = get_or_default(color_family, "default")
    line_style = get_or_default(line_style, "solid")
    line_weight = get_or_default(line_weight, 1.5)
    line_alpha = get_or_default(line_alpha, 1.0)

    # Figure settings
    fig_setting = canvas_setting()
    params = fig_setting[2]
    plt.rcParams.update(params)
    fig, ax_kpoints = plt.subplots(figsize=fig_setting[0], dpi=fig_setting[1])
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Configure scientific notation for y-axis
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    ax_kpoints.yaxis.set_major_formatter(formatter)

    # Data input
    data_dict_list = read_energy_parameters(source_data)
    energy_values = [d.get("total energy", d.get("Total Energy")) for d in data_dict_list]
    lattice_constants = [d.get("lattice constant", d.get("Lattice Constant")) for d in data_dict_list]

    # Filter out None values
    filtered_data = [(lattice, energy) for lattice, energy in zip(lattice_constants, energy_values)
                     if lattice is not None and energy is not None]

    if not filtered_data:
        print("No valid data found for plotting.")
        return

    # Unzip filtered data and sort
    lattice_constants, energy_values = zip(*filtered_data)
    sorted_data = sorted(zip(lattice_constants, energy_values), key=lambda x: x[0])
    lattice_sorted, energy_sorted = zip(*sorted_data)

    # Define boundaries for lattice constants
    lattice_start = min(lattice_sorted) if lattice_boundary in [None, ""] else float(lattice_boundary[0] or min(lattice_sorted))
    lattice_end = max(lattice_sorted) if lattice_boundary in [None, ""] else float(lattice_boundary[1] or max(lattice_sorted))

    # Filter data within the specified boundary
    lattice_filtered, energy_filtered = zip(*[(l, e) for l, e in zip(lattice_sorted, energy_sorted) if lattice_start <= l <= lattice_end])

    # Estimate EOS parameters and fitted energy values using all data
    eos_params, resampled_lattice, fitted_energy = fit_birch_murnaghan(lattice_filtered, energy_filtered, sample_count=100)

    # Plot the fitted EOS curve
    ax_kpoints.plot(resampled_lattice, fitted_energy, color=color_sampling(color_family)[1], ls=line_style, lw=line_weight, alpha=line_alpha,
                    label=f"Fitted EOS curve {info_suffix}")

    # Select scatter sample points
    if num_samples >= len(lattice_filtered):
        scatter_lattice, scatter_energy = lattice_filtered, energy_filtered
    else:
        x_samples = np.linspace(lattice_start, lattice_end, num_samples)
        scatter_lattice, scatter_energy = zip(*[(l, e) for x in x_samples for l, e in zip(lattice_filtered, energy_filtered)
                                                 if np.abs(l - x) == min(np.abs(np.array(lattice_filtered) - x))])

    # Remove duplicate points
    unique_points = set()
    scatter_lattice_unique, scatter_energy_unique = [], []
    for x, y in zip(scatter_lattice, scatter_energy):
        if x not in unique_points:
            unique_points.add(x)
            scatter_lattice_unique.append(x)
            scatter_energy_unique.append(y)

    # Scatter sample data points
    ax_kpoints.scatter(scatter_lattice_unique, scatter_energy_unique, s=line_weight * 4, fc="#FFFFFF", ec=color_sampling(color_family)[1], alpha=line_alpha,
                       label=f"Sampled data {info_suffix}", zorder=2)

    # Mark the minimum energy point
    min_energy_idx = np.argmin(energy_filtered)
    ax_kpoints.scatter(lattice_filtered[min_energy_idx], energy_filtered[min_energy_idx], s=line_weight * 4, 
                       fc=color_sampling(color_family)[2], ec=color_sampling(color_family)[2],
                       label=f"Minimum energy at {lattice_filtered[min_energy_idx]:.6g} {info_suffix}", zorder=3)

    # Set labels, title, and legend
    ax_kpoints.set_xlabel(r"Lattice constant (Å)")
    ax_kpoints.set_ylabel(r"Energy (eV)")
    ax_kpoints.set_title(f"{suptitle} {info_suffix}")
    ax_kpoints.legend()
    plt.tight_layout()

def plot_energy_lattice(suptitle, lattice_list):
    """
    Generalized function to plot energy versus lattice constant for multiple datasets.

    Parameters:
    - lattice_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, lattice_boundary, num_samples, color_family, line_style, line_weight, line_alpha].
    """
    # Single dataset case
    if not isinstance(lattice_list[0], list):
        return plot_energy_lattice_single(suptitle, lattice_list)
    elif len(lattice_list) == 1:
        return plot_energy_lattice_single(suptitle, *lattice_list)

    # Validate input: Ensure lattice_list is a 2D list
    if not all(isinstance(item, list) for item in lattice_list):
        raise ValueError("Invalid input: lattice_list must be a list of lists.")

    # Ensure all inner lists have at least 8 elements
    for index, data in enumerate(lattice_list):
        if len(data) < 8:
            lattice_list[index] += [None] * (8 - len(data))

    # Figure settings
    fig_setting = canvas_setting()
    params = fig_setting[2]
    plt.rcParams.update(params)
    fig, ax = plt.subplots(figsize=fig_setting[0], dpi=fig_setting[1])
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)

    legend_handles = []

    # Process each dataset in lattice_list
    for idx, data in enumerate(lattice_list):
        # print(f"Processing dataset {idx}: {data}")

        # Unpack the data
        info_suffix, source_data, lattice_boundary, num_samples, color_family, line_style, line_weight, line_alpha = data

        # Validate `source_data`
        if not source_data or not isinstance(source_data, str):
            raise ValueError(f"Invalid source_data in dataset {idx}: {source_data}")

        # Validate `num_samples`
        if num_samples is None or not isinstance(num_samples, int) or num_samples <= 0:
            num_samples = 100  # Default sample count

        # Apply default values
        color_family = get_or_default(color_family, "default")
        line_style = get_or_default(line_style, "solid")
        line_weight = get_or_default(line_weight, 1.5)
        line_alpha = get_or_default(line_alpha, 1.0)

        # Read data from source_data
        try:
            data_dict_list = read_energy_parameters(source_data)
        except Exception as e:
            raise ValueError(f"Failed to read data from {source_data}: {e}")

        energy_values = [d.get("total energy", d.get("Total Energy")) for d in data_dict_list]
        lattice_constants = [d.get("lattice constant", d.get("Lattice Constant")) for d in data_dict_list]

        # Filter out None values
        filtered_data = [(lattice, energy) for lattice, energy in zip(lattice_constants, energy_values)
                         if lattice is not None and energy is not None]

        if not filtered_data:
            print(f"Warning: No valid data for {info_suffix}. Skipping this dataset.")
            continue

        # Unzip filtered data and sort
        lattice_constants, energy_values = zip(*filtered_data)
        sorted_data = sorted(zip(lattice_constants, energy_values), key=lambda x: x[0])
        lattice_sorted, energy_sorted = zip(*sorted_data)

        # Define boundaries for lattice constants
        lattice_start = min(lattice_sorted) if lattice_boundary in [None, ""] else float(lattice_boundary[0] or min(lattice_sorted))
        lattice_end = max(lattice_sorted) if lattice_boundary in [None, ""] else float(lattice_boundary[1] or max(lattice_sorted))

        # Filter data within the specified boundary
        lattice_filtered, energy_filtered = zip(*[(l, e) for l, e in zip(lattice_sorted, energy_sorted) if lattice_start <= l <= lattice_end])

        # Estimate EOS parameters and fitted energy values using all data
        eos_params, resampled_lattice, fitted_energy = fit_birch_murnaghan(lattice_filtered, energy_filtered, sample_count=100)

        # Plot the fitted EOS curve
        ax.plot(resampled_lattice, fitted_energy, color=color_sampling(color_family)[1], ls=line_style, lw=line_weight, alpha=line_alpha,
                label=f"Fitted EOS curve {info_suffix}")

        # Select scatter sample points
        if num_samples >= len(lattice_filtered):
            scatter_lattice, scatter_energy = lattice_filtered, energy_filtered
        else:
            x_samples = np.linspace(lattice_start, lattice_end, num_samples)
            scatter_lattice, scatter_energy = zip(*[(l, e) for x in x_samples for l, e in zip(lattice_filtered, energy_filtered)
                                                     if np.abs(l - x) == min(np.abs(np.array(lattice_filtered) - x))])

        # Scatter sample data points
        ax.scatter(scatter_lattice, scatter_energy, s=line_weight * 4, fc="#FFFFFF", ec=color_sampling(color_family)[1], alpha=line_alpha,
                   label=f"Sampled data {info_suffix}", zorder=2)

        # Mark the minimum energy point
        min_energy_idx = np.argmin(energy_filtered)
        ax.scatter(lattice_filtered[min_energy_idx], energy_filtered[min_energy_idx], s=line_weight * 4, 
                   fc=color_sampling(color_family)[2], ec=color_sampling(color_family)[2],
                   label=f"Minimum energy at {lattice_filtered[min_energy_idx]:.6g} {info_suffix}", zorder=3)

        # Add legend entries
        legend_handles.append(mlines.Line2D([], [], color=color_sampling(color_family)[1], linestyle=line_style,
                                            label=f"Fitted EOS curve {info_suffix}"))
        legend_handles.append(mlines.Line2D([], [], color=color_sampling(color_family)[1], marker='o', linestyle='None',
                                            label=f"Sampled data {info_suffix}"))
        legend_handles.append(mlines.Line2D([], [], color=color_sampling(color_family)[2], marker='o', linestyle='None',
                                            label=f"Minimum energy {info_suffix}"))

    ax.set_xlabel(r"Lattice constant (Å)")
    ax.set_ylabel(r"Energy (eV)")
    ax.set_title(f"{suptitle}")
    ax.legend(handles=legend_handles, loc="best")
    plt.tight_layout()


    """
    Plot multiple datasets: Energy vs a3.
    a3_list: A 2D list where each sublist contains
      [info_suffix, source_data, a3_boundary, num_samples, color_family, line_style, line_weight, line_alpha]
    """
    if not isinstance(a3_list[0], list):
        return plot_energy_a3_single(suptitle, a3_list)
    elif len(a3_list) == 1:
        return plot_energy_a3_single(suptitle, *a3_list)

    for index, data in enumerate(a3_list):
        if not isinstance(data, list):
            print(f"Error: Item at index {index} in a3_list must be a list.")
            return
        if len(data) < 8:
            print(f"Warning: Item at index {index} has less than 8 elements. Missing elements will be filled with None.")
            a3_list[index] += [None] * (8 - len(data))

    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    legend_handles = []
    global_a3_set = set()
    for data in a3_list:
        args_info = data + [None] * (8 - len(data))
        _, source_data, a3_boundary, _, _, _, _, _ = args_info
        data_dict_list = read_energy_parameters(source_data)
        a3_values = [d.get("a3") for d in data_dict_list]
        if not a3_values:
            continue
        a3_start = min(a3_values) if a3_boundary in [None, ""] else float(a3_boundary[0] or min(a3_values))
        a3_end = max(a3_values) if a3_boundary in [None, ""] else float(a3_boundary[1] or max(a3_values))
        for a3 in a3_values:
            if a3 is not None and a3_start <= a3 <= a3_end:
                global_a3_set.add(a3)
    sorted_global_a3 = sorted(global_a3_set)
    global_a3_labels = [f"{a3:.6g}" for a3 in sorted_global_a3]

    for data in a3_list:
        args_info = data + [None] * (8 - len(data))
        info_suffix, source_data, a3_boundary, num_samples, color_family, line_style, line_weight, line_alpha = args_info
        color_family = get_or_default(color_family, "default")
        line_style = get_or_default(line_style, "solid")
        line_weight = get_or_default(line_weight, 1.5)
        line_alpha = get_or_default(line_alpha, 1.0)
        colors = color_sampling(color_family)
        data_dict_list = read_energy_parameters(source_data)
        energy = [d.get("total energy", d.get("Total Energy")) for d in data_dict_list]
        a3_values = [d.get("a3") for d in data_dict_list]
        if not a3_values:
            continue
        a3_start = min(a3_values) if a3_boundary in [None, ""] else float(a3_boundary[0] or min(a3_values))
        a3_end = max(a3_values) if a3_boundary in [None, ""] else float(a3_boundary[1] or max(a3_values))
        filtered = [(a3, e) for a3, e in zip(a3_values, energy) if a3 is not None and a3_start <= a3 <= a3_end]
        if not filtered:
            continue
        a3_filtered, energy_filtered = zip(*sorted(filtered, key=lambda x: x[0]))
        eos_params, resampled_a3, fitted_energy = fit_birch_murnaghan(a3_filtered, energy_filtered, sample_count=100)
        fitted_interp = np.interp(sorted_global_a3, a3_filtered, fitted_energy)
        energy_interp = np.interp(sorted_global_a3, a3_filtered, energy_filtered)
        plt.plot(range(len(sorted_global_a3)), fitted_interp, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha,
                 label=f"Fitted EOS curve {info_suffix}")
        plt.scatter(range(len(sorted_global_a3)), energy_interp, s=line_weight * 4, c=colors[1], alpha=line_alpha, zorder=1)
        legend_handles.append(
            mlines.Line2D([], [], color=colors[1], marker='o', markersize=6, linestyle=line_style,
                          label=f"Energy vs a3 {info_suffix}")
        )

    plt.xlabel("a3 (Å)")
    plt.ylabel("Energy (eV)")
    plt.xticks(ticks=range(len(global_a3_labels)), labels=global_a3_labels, rotation=45, ha="right")
    plt.title(f"{suptitle}")
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

def plot_energy_scaling_single(suptitle, *args_list):
    help_info = (
        "Usage: plot_energy_scaling(suptitle, args_list)\n"
        "args_list: A list containing [info_suffix, source_data, scaling_boundary, color_family, line_style, line_weight, line_alpha].\n"
        "Example: plot_energy_scaling(suptitle, ['Material Info', 'source_data_path', (start, end), 'green', 'dashed', 2.0, 0.8])\n"
    )

    if not args_list or args_list[0] in ["HELP", "Help", "help"]:
        print(help_info)
        return

    # Unpack args_list and handle missing parameters
    args_info = args_list[0] + [None] * (7 - len(args_list[0]))
    info_suffix, source_data, scaling_boundary, color_family, line_style, line_weight, line_alpha = args_info

    # Apply default values using `get_or_default`
    color_family = get_or_default(color_family, "default")
    line_style   = get_or_default(line_style, "solid")
    line_weight  = get_or_default(line_weight, 1.5)
    line_alpha   = get_or_default(line_alpha, 1.0)

    # Figure settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Apply ScalarFormatter with scientific notation limits
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    # Color sampling
    colors = color_sampling(color_family)

    # Data input
    data_dict_list = read_energy_parameters(source_data)

    # Extract total energy and Scaling values
    energy = [d.get("total energy") for d in data_dict_list]
    scaling_values = [d.get("Scaling") for d in data_dict_list]

    # Filter out None values
    filtered_data = [(s, e) for s, e in zip(scaling_values, energy) if s is not None and e is not None]
    if not filtered_data:
        print("No valid data found for plotting.")
        return

    # Unzip and sort data by Scaling values
    scaling_values, energy = zip(*filtered_data)
    sorted_data = sorted(zip(scaling_values, energy), key=lambda x: x[0])
    scaling_sorted, energy_sorted = zip(*sorted_data)

    # Set boundaries for Scaling
    scaling_start = min(scaling_sorted) if not scaling_boundary or scaling_boundary[0] is None else float(scaling_boundary[0])
    scaling_end   = max(scaling_sorted) if not scaling_boundary or scaling_boundary[1] is None else float(scaling_boundary[1])

    # Filter data within the specified boundary
    scaling_filtered, energy_filtered = zip(*[
        (s, e) for s, e in zip(scaling_sorted, energy_sorted) if scaling_start <= s <= scaling_end
    ])

    # Plotting
    plt.plot(scaling_filtered, energy_filtered, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha,
             label=f"Energy versus Scaling {info_suffix}")
    plt.scatter(scaling_filtered, energy_filtered, s=line_weight * 4, c=colors[1], zorder=1, alpha=line_alpha)

    # Set labels and title (x-axis labels are not rotated)
    plt.title(f"{suptitle} {info_suffix}")
    plt.xlabel("Scaling")
    plt.ylabel("Energy (eV)")
    plt.legend(loc="best")
    plt.tight_layout()

def plot_energy_scaling(suptitle, scaling_list):
    """
    Generalized function to plot energy versus Scaling for multiple datasets.

    Parameters:
    - scaling_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, scaling_boundary, color_family, line_style, line_weight, line_alpha].
    """
    if not is_nested_list(scaling_list):
        return plot_energy_scaling_single(suptitle, scaling_list)
    elif len(scaling_list) == 1:
        return plot_energy_scaling_single(suptitle, *scaling_list)

    # Verify each dataset has correct structure
    for index, data in enumerate(scaling_list):
        if not isinstance(data, list):
            print(f"Error: Item at index {index} in scaling_list must be a list.")
            return
        if len(data) < 4:
            print(f"Warning: Item at index {index} has less than 4 elements. Missing elements will be filled with None.")
            scaling_list[index] += [None] * (7 - len(data))

    # Figure and styling settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    legend_handles = []
    global_scaling_set = set()
    scaling_config_map = {}

    # Collect global Scaling values across all datasets
    for data in scaling_list:
        args_info = data + [None] * (7 - len(data))
        _, source_data, scaling_boundary, color_family, line_style, line_weight, line_alpha = args_info

        color_family = get_or_default(color_family, "default")
        line_style   = get_or_default(line_style, "solid")
        line_weight  = get_or_default(line_weight, 1.5)
        line_alpha   = get_or_default(line_alpha, 1.0)

        data_dict_list = read_energy_parameters(source_data)
        scaling_values = [d.get("Scaling") for d in data_dict_list]
        scaling_start = min(scaling_values) if not scaling_boundary or scaling_boundary[0] is None else float(scaling_boundary[0])
        scaling_end   = max(scaling_values) if not scaling_boundary or scaling_boundary[1] is None else float(scaling_boundary[1])
        for val in scaling_values:
            if val is not None and scaling_start <= val <= scaling_end:
                global_scaling_set.add(val)
                scaling_config_map[val] = f"{val:.6g}"
    sorted_global_scaling = sorted(global_scaling_set)
    global_scaling_labels = [scaling_config_map[val] for val in sorted_global_scaling]

    # Plot each dataset aligned with the global Scaling values
    for data in scaling_list:
        args_info = data + [None] * (7 - len(data))
        info_suffix, source_data, scaling_boundary, color_family, line_style, line_weight, line_alpha = args_info

        color_family = get_or_default(color_family, "default")
        line_style   = get_or_default(line_style, "solid")
        line_weight  = get_or_default(line_weight, 1.5)
        line_alpha   = get_or_default(line_alpha, 1.0)
        colors = color_sampling(color_family)
        data_dict_list = read_energy_parameters(source_data)
        energy = [d.get("total energy") for d in data_dict_list]
        scaling_values = [d.get("Scaling") for d in data_dict_list]
        scaling_start = min(scaling_values) if not scaling_boundary or scaling_boundary[0] is None else float(scaling_boundary[0])
        scaling_end   = max(scaling_values) if not scaling_boundary or scaling_boundary[1] is None else float(scaling_boundary[1])
        filtered = [(s, e) for s, e in zip(scaling_values, energy) if s is not None and scaling_start <= s <= scaling_end]
        if not filtered:
            continue
        scaling_filtered, energy_filtered = zip(*sorted(filtered, key=lambda x: x[0]))

        # Align energies with global sorted Scaling values
        energy_aligned = [energy_filtered[scaling_filtered.index(val)] if val in scaling_filtered else np.nan for val in sorted_global_scaling]
        plt.plot(sorted_global_scaling, energy_aligned, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha)
        plt.scatter(sorted_global_scaling, energy_aligned, s=line_weight * 4, c=colors[1], zorder=1, alpha=line_alpha)

        legend_handles.append(
            mlines.Line2D([], [], color=colors[1], marker='o', markersize=6,
                          linestyle=line_style, label=f"Energy versus Scaling {info_suffix}")
        )

    plt.xlabel("Scaling")
    plt.ylabel("Energy (eV)")
    plt.xticks(ticks=range(len(global_scaling_labels)), labels=global_scaling_labels)
    plt.title(f"{suptitle}")
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

def plot_energy_a1_single(suptitle, *args_list):
    help_info = (
        "Usage: plot_energy_a1(suptitle, args_list)\n"
        "args_list: A list containing [info_suffix, source_data, a1_boundary, color_family, line_style, line_weight, line_alpha].\n"
        "Example: plot_energy_a1(suptitle, ['Material Info', 'source_data_path', (start, end), 'blue', 'solid', 1.5, 1.0])\n"
    )

    if not args_list or args_list[0] in ["HELP", "Help", "help"]:
        print(help_info)
        return

    # Unpack args_list and handle missing parameters
    args_info = args_list[0] + [None] * (7 - len(args_list[0]))
    info_suffix, source_data, a1_boundary, color_family, line_style, line_weight, line_alpha = args_info

    # Apply default values using `get_or_default`
    color_family = get_or_default(color_family, "default")
    line_style   = get_or_default(line_style, "solid")
    line_weight  = get_or_default(line_weight, 1.5)
    line_alpha   = get_or_default(line_alpha, 1.0)

    # Figure settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Apply ScalarFormatter with scientific notation limits
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    # Color sampling
    colors = color_sampling(color_family)

    # Data input
    data_dict_list = read_energy_parameters(source_data)

    # Extract total energy and a1 values
    energy = [d.get("total energy") for d in data_dict_list]
    a1_values = [d.get("a1") for d in data_dict_list]

    # Filter out None values
    filtered_data = [(val, e) for val, e in zip(a1_values, energy) if val is not None and e is not None]
    if not filtered_data:
        print("No valid data found for plotting.")
        return

    # Unzip and sort data by a1 values
    a1_values, energy = zip(*filtered_data)
    sorted_data = sorted(zip(a1_values, energy), key=lambda x: x[0])
    a1_sorted, energy_sorted = zip(*sorted_data)

    # Set boundaries for a1
    a1_start = min(a1_sorted) if not a1_boundary or a1_boundary[0] is None else float(a1_boundary[0])
    a1_end   = max(a1_sorted) if not a1_boundary or a1_boundary[1] is None else float(a1_boundary[1])

    # Filter data within the specified boundary
    a1_filtered, energy_filtered = zip(*[
        (val, e) for val, e in zip(a1_sorted, energy_sorted) if a1_start <= val <= a1_end
    ])

    # Plotting
    plt.plot(a1_filtered, energy_filtered, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha,
             label=f"Energy versus a1 {info_suffix}")
    plt.scatter(a1_filtered, energy_filtered, s=line_weight * 4, c=colors[1], zorder=1, alpha=line_alpha)

    # Set labels and title
    plt.title(f"{suptitle} {info_suffix}")
    plt.xlabel("a1 (Å)")
    plt.ylabel("Energy (eV)")
    plt.legend(loc="best")
    plt.tight_layout()

def plot_energy_a1(suptitle, a1_list):
    """
    Generalized function to plot energy versus a1 for multiple datasets.

    Parameters:
    - a1_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, a1_boundary, color_family, line_style, line_weight, line_alpha].
    """
    if not is_nested_list(a1_list):
        return plot_energy_a1_single(suptitle, a1_list)
    elif len(a1_list) == 1:
        return plot_energy_a1_single(suptitle, *a1_list)

    for index, data in enumerate(a1_list):
        if not isinstance(data, list):
            print(f"Error: Item at index {index} in a1_list must be a list.")
            return
        if len(data) < 4:
            print(f"Warning: Item at index {index} has less than 4 elements. Missing elements will be filled with None.")
            a1_list[index] += [None] * (7 - len(data))

    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    legend_handles = []
    global_a1_set = set()
    a1_config_map = {}

    # Gather all unique a1 values for the global x-axis
    for data in a1_list:
        args_info = data + [None] * (7 - len(data))
        _, source_data, a1_boundary, color_family, line_style, line_weight, line_alpha = args_info
        data_dict_list = read_energy_parameters(source_data)
        a1_values = [d.get("a1") for d in data_dict_list]
        a1_start = min(a1_values) if not a1_boundary or a1_boundary[0] is None else float(a1_boundary[0])
        a1_end   = max(a1_values) if not a1_boundary or a1_boundary[1] is None else float(a1_boundary[1])
        for val in a1_values:
            if val is not None and a1_start <= val <= a1_end:
                global_a1_set.add(val)
                a1_config_map[val] = f"{val:.6g}"
    sorted_global_a1 = sorted(global_a1_set)
    global_a1_labels = [a1_config_map[val] for val in sorted_global_a1]

    # Plot each dataset aligned with the global a1 values
    for data in a1_list:
        args_info = data + [None] * (7 - len(data))
        info_suffix, source_data, a1_boundary, color_family, line_style, line_weight, line_alpha = args_info

        color_family = get_or_default(color_family, "default")
        line_style   = get_or_default(line_style, "solid")
        line_weight  = get_or_default(line_weight, 1.5)
        line_alpha   = get_or_default(line_alpha, 1.0)
        colors = color_sampling(color_family)
        data_dict_list = read_energy_parameters(source_data)
        energy = [d.get("total energy") for d in data_dict_list]
        a1_values = [d.get("a1") for d in data_dict_list]
        a1_start = min(a1_values) if not a1_boundary or a1_boundary[0] is None else float(a1_boundary[0])
        a1_end   = max(a1_values) if not a1_boundary or a1_boundary[1] is None else float(a1_boundary[1])
        filtered = [(val, e) for val, e in zip(a1_values, energy) if val is not None and a1_start <= val <= a1_end]
        if not filtered:
            continue
        a1_filtered, energy_filtered = zip(*sorted(filtered, key=lambda x: x[0]))

        # Align energies with global sorted a1 values
        energy_aligned = [energy_filtered[a1_filtered.index(val)] if val in a1_filtered else np.nan for val in sorted_global_a1]
        plt.plot(sorted_global_a1, energy_aligned, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha,
                 label=f"Energy versus a1 {info_suffix}")
        plt.scatter(sorted_global_a1, energy_aligned, s=line_weight * 4, c=colors[1], alpha=line_alpha, zorder=1)
        legend_handles.append(
            mlines.Line2D([], [], color=colors[1], marker='o', markersize=6, linestyle=line_style,
                          label=f"Energy versus a1 {info_suffix}")
        )

    plt.xlabel("a1 (Å)")
    plt.ylabel("Energy (eV)")
    plt.xticks(ticks=range(len(global_a1_labels)), labels=global_a1_labels)
    plt.title(f"{suptitle}")
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

def plot_energy_a2_single(suptitle, *args_list):
    help_info = (
        "Usage: plot_energy_a2(suptitle, args_list)\n"
        "args_list: A list containing [info_suffix, source_data, a2_boundary, color_family, line_style, line_weight, line_alpha].\n"
        "Example: plot_energy_a2(suptitle, ['Material Info', 'source_data_path', (start, end), 'red', 'dashed', 2.0, 0.8])\n"
    )

    if not args_list or args_list[0] in ["HELP", "Help", "help"]:
        print(help_info)
        return

    # Unpack args_list and handle missing parameters
    args_info = args_list[0] + [None] * (7 - len(args_list[0]))
    info_suffix, source_data, a2_boundary, color_family, line_style, line_weight, line_alpha = args_info

    # Apply default values using `get_or_default`
    color_family = get_or_default(color_family, "default")
    line_style   = get_or_default(line_style, "solid")
    line_weight  = get_or_default(line_weight, 1.5)
    line_alpha   = get_or_default(line_alpha, 1.0)

    # Figure settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Apply ScalarFormatter with scientific notation limits
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    # Color sampling
    colors = color_sampling(color_family)

    # Data input
    data_dict_list = read_energy_parameters(source_data)

    # Extract total energy and a2 values
    energy = [d.get("total energy") for d in data_dict_list]
    a2_values = [d.get("a2") for d in data_dict_list]

    # Filter out None values
    filtered_data = [(val, e) for val, e in zip(a2_values, energy) if val is not None and e is not None]
    if not filtered_data:
        print("No valid data found for plotting.")
        return

    # Unzip and sort data by a2 values
    a2_values, energy = zip(*filtered_data)
    sorted_data = sorted(zip(a2_values, energy), key=lambda x: x[0])
    a2_sorted, energy_sorted = zip(*sorted_data)

    # Set boundaries for a2
    a2_start = min(a2_sorted) if not a2_boundary or a2_boundary[0] is None else float(a2_boundary[0])
    a2_end   = max(a2_sorted) if not a2_boundary or a2_boundary[1] is None else float(a2_boundary[1])

    # Filter data within the specified boundary
    a2_filtered, energy_filtered = zip(*[
        (val, e) for val, e in zip(a2_sorted, energy_sorted) if a2_start <= val <= a2_end
    ])

    # Plotting
    plt.plot(a2_filtered, energy_filtered, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha,
             label=f"Energy versus a2 {info_suffix}")
    plt.scatter(a2_filtered, energy_filtered, s=line_weight * 4, c=colors[1], zorder=1, alpha=line_alpha)

    # Set labels and title
    plt.title(f"{suptitle} {info_suffix}")
    plt.xlabel("a2 (Å)")
    plt.ylabel("Energy (eV)")
    plt.legend(loc="best")
    plt.tight_layout()

def plot_energy_a2(suptitle, a2_list):
    """
    Generalized function to plot energy versus a2 for multiple datasets.

    Parameters:
    - a2_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, a2_boundary, color_family, line_style, line_weight, line_alpha].
    """
    if not is_nested_list(a2_list):
        return plot_energy_a2_single(suptitle, a2_list)
    elif len(a2_list) == 1:
        return plot_energy_a2_single(suptitle, *a2_list)

    for index, data in enumerate(a2_list):
        if not isinstance(data, list):
            print(f"Error: Item at index {index} in a2_list must be a list.")
            return
        if len(data) < 4:
            print(f"Warning: Item at index {index} has less than 4 elements. Missing elements will be filled with None.")
            a2_list[index] += [None] * (7 - len(data))

    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    legend_handles = []
    global_a2_set = set()
    a2_config_map = {}

    # Gather all unique a2 values for the global x-axis
    for data in a2_list:
        args_info = data + [None] * (7 - len(data))
        _, source_data, a2_boundary, color_family, line_style, line_weight, line_alpha = args_info
        data_dict_list = read_energy_parameters(source_data)
        a2_values = [d.get("a2") for d in data_dict_list]
        a2_start = min(a2_values) if not a2_boundary or a2_boundary[0] is None else float(a2_boundary[0])
        a2_end   = max(a2_values) if not a2_boundary or a2_boundary[1] is None else float(a2_boundary[1])
        for val in a2_values:
            if val is not None and a2_start <= val <= a2_end:
                global_a2_set.add(val)
                a2_config_map[val] = f"{val:.6g}"
    sorted_global_a2 = sorted(global_a2_set)
    global_a2_labels = [a2_config_map[val] for val in sorted_global_a2]

    # Plot each dataset aligned with the global a2 values
    for data in a2_list:
        args_info = data + [None] * (7 - len(data))
        info_suffix, source_data, a2_boundary, color_family, line_style, line_weight, line_alpha = args_info

        color_family = get_or_default(color_family, "default")
        line_style   = get_or_default(line_style, "solid")
        line_weight  = get_or_default(line_weight, 1.5)
        line_alpha   = get_or_default(line_alpha, 1.0)
        colors = color_sampling(color_family)
        data_dict_list = read_energy_parameters(source_data)
        energy = [d.get("total energy") for d in data_dict_list]
        a2_values = [d.get("a2") for d in data_dict_list]
        a2_start = min(a2_values) if not a2_boundary or a2_boundary[0] is None else float(a2_boundary[0])
        a2_end   = max(a2_values) if not a2_boundary or a2_boundary[1] is None else float(a2_boundary[1])
        filtered = [(val, e) for val, e in zip(a2_values, energy) if val is not None and a2_start <= val <= a2_end]
        if not filtered:
            continue
        a2_filtered, energy_filtered = zip(*sorted(filtered, key=lambda x: x[0]))

        energy_aligned = [energy_filtered[a2_filtered.index(val)] if val in a2_filtered else np.nan for val in sorted_global_a2]
        plt.plot(sorted_global_a2, energy_aligned, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha,
                 label=f"Energy versus a2 {info_suffix}")
        plt.scatter(sorted_global_a2, energy_aligned, s=line_weight * 4, c=colors[1], alpha=line_alpha, zorder=1)
        legend_handles.append(
            mlines.Line2D([], [], color=colors[1], marker='o', markersize=6, linestyle=line_style,
                          label=f"Energy versus a2 {info_suffix}")
        )

    plt.xlabel("a2 (Å)")
    plt.ylabel("Energy (eV)")
    plt.xticks(ticks=range(len(global_a2_labels)), labels=global_a2_labels)
    plt.title(f"{suptitle}")
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

def plot_energy_a3_single(suptitle, *args_list):
    help_info = (
        "Usage: plot_energy_a3(suptitle, args_list)\n"
        "args_list: A list containing [info_suffix, source_data, a3_boundary, color_family, line_style, line_weight, line_alpha].\n"
        "Example: plot_energy_a3(suptitle, ['Material Info', 'source_data_path', (start, end), 'purple', 'dashed', 2.0, 0.8])\n"
    )

    if not args_list or args_list[0] in ["HELP", "Help", "help"]:
        print(help_info)
        return

    # Unpack args_list and handle missing parameters
    args_info = args_list[0] + [None] * (7 - len(args_list[0]))
    info_suffix, source_data, a3_boundary, color_family, line_style, line_weight, line_alpha = args_info

    # Apply default values using `get_or_default`
    color_family = get_or_default(color_family, "default")
    line_style   = get_or_default(line_style, "solid")
    line_weight  = get_or_default(line_weight, 1.5)
    line_alpha   = get_or_default(line_alpha, 1.0)

    # Figure settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Apply ScalarFormatter with scientific notation limits
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    # Color sampling
    colors = color_sampling(color_family)

    # Data input
    data_dict_list = read_energy_parameters(source_data)

    # Extract total energy and a3 values
    energy = [d.get("total energy") for d in data_dict_list]
    a3_values = [d.get("a3") for d in data_dict_list]

    # Filter out None values
    filtered_data = [(val, e) for val, e in zip(a3_values, energy) if val is not None and e is not None]
    if not filtered_data:
        print("No valid data found for plotting.")
        return

    # Unzip and sort data by a3 values
    a3_values, energy = zip(*filtered_data)
    sorted_data = sorted(zip(a3_values, energy), key=lambda x: x[0])
    a3_sorted, energy_sorted = zip(*sorted_data)

    # Set boundaries for a3
    a3_start = min(a3_sorted) if not a3_boundary or a3_boundary[0] is None else float(a3_boundary[0])
    a3_end   = max(a3_sorted) if not a3_boundary or a3_boundary[1] is None else float(a3_boundary[1])

    # Filter data within the specified boundary
    a3_filtered, energy_filtered = zip(*[
        (val, e) for val, e in zip(a3_sorted, energy_sorted) if a3_start <= val <= a3_end
    ])

    # Plotting
    plt.plot(a3_filtered, energy_filtered, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha,
             label=f"Energy versus a3 {info_suffix}")
    plt.scatter(a3_filtered, energy_filtered, s=line_weight * 4, c=colors[1], zorder=1, alpha=line_alpha)

    # Set labels and title
    plt.title(f"{suptitle} {info_suffix}")
    plt.xlabel("a3 (Å)")
    plt.ylabel("Energy (eV)")
    plt.legend(loc="best")
    plt.tight_layout()

def plot_energy_a3(suptitle, a3_list):
    """
    Generalized function to plot energy versus a3 for multiple datasets.

    Parameters:
    - a3_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, a3_boundary, color_family, line_style, line_weight, line_alpha].
    """
    if not is_nested_list(a3_list):
        return plot_energy_a3_single(suptitle, a3_list)
    elif len(a3_list) == 1:
        return plot_energy_a3_single(suptitle, *a3_list)

    for index, data in enumerate(a3_list):
        if not isinstance(data, list):
            print(f"Error: Item at index {index} in a3_list must be a list.")
            return
        if len(data) < 4:
            print(f"Warning: Item at index {index} has less than 4 elements. Missing elements will be filled with None.")
            a3_list[index] += [None] * (7 - len(data))

    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    legend_handles = []
    global_a3_set = set()
    a3_config_map = {}

    # Gather all unique a3 values for the global x-axis
    for data in a3_list:
        args_info = data + [None] * (7 - len(data))
        _, source_data, a3_boundary, color_family, line_style, line_weight, line_alpha = args_info
        data_dict_list = read_energy_parameters(source_data)
        a3_values = [d.get("a3") for d in data_dict_list]
        a3_start = min(a3_values) if not a3_boundary or a3_boundary[0] is None else float(a3_boundary[0])
        a3_end   = max(a3_values) if not a3_boundary or a3_boundary[1] is None else float(a3_boundary[1])
        for val in a3_values:
            if val is not None and a3_start <= val <= a3_end:
                global_a3_set.add(val)
                a3_config_map[val] = f"{val:.6g}"
    sorted_global_a3 = sorted(global_a3_set)
    global_a3_labels = [a3_config_map[val] for val in sorted_global_a3]

    # Plot each dataset aligned with the global a3 values
    for data in a3_list:
        args_info = data + [None] * (7 - len(data))
        info_suffix, source_data, a3_boundary, color_family, line_style, line_weight, line_alpha = args_info

        color_family = get_or_default(color_family, "default")
        line_style   = get_or_default(line_style, "solid")
        line_weight  = get_or_default(line_weight, 1.5)
        line_alpha   = get_or_default(line_alpha, 1.0)
        colors = color_sampling(color_family)
        data_dict_list = read_energy_parameters(source_data)
        energy = [d.get("total energy") for d in data_dict_list]
        a3_values = [d.get("a3") for d in data_dict_list]
        a3_start = min(a3_values) if not a3_boundary or a3_boundary[0] is None else float(a3_boundary[0])
        a3_end   = max(a3_values) if not a3_boundary or a3_boundary[1] is None else float(a3_boundary[1])
        filtered = [(val, e) for val, e in zip(a3_values, energy) if val is not None and a3_start <= val <= a3_end]
        if not filtered:
            continue
        a3_filtered, energy_filtered = zip(*sorted(filtered, key=lambda x: x[0]))

        energy_aligned = [energy_filtered[a3_filtered.index(val)] if val in a3_filtered else np.nan for val in sorted_global_a3]
        plt.plot(sorted_global_a3, energy_aligned, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha,
                 label=f"Energy versus a3 {info_suffix}")
        plt.scatter(sorted_global_a3, energy_aligned, s=line_weight * 4, c=colors[1], alpha=line_alpha, zorder=1)
        legend_handles.append(
            mlines.Line2D([], [], color=colors[1], marker='o', markersize=6, linestyle=line_style,
                          label=f"Energy versus a3 {info_suffix}")
        )

    plt.xlabel("a3 (Å)")
    plt.ylabel("Energy (eV)")
    plt.xticks(ticks=range(len(global_a3_labels)), labels=global_a3_labels)
    plt.title(f"{suptitle}")
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

def plot_energy_kpoints_encut_single(suptitle, kpoints_list, encut_list):
    """
    Plot energy versus K-points and energy cutoff on a shared y-axis with separate x-axes.

    Parameters:
    - kpoints_list: A list containing [info_suffix, source_data, kpoints_boundary, color_family, line_style, line_weight, line_alpha].
    - encut_list: A list containing [info_suffix, source_data, encut_boundary, color_family, line_style, line_weight, line_alpha].
    """

    # Unpack and handle missing values for kpoints_list
    kpoints_info = kpoints_list + [None] * (7 - len(kpoints_list))
    info_suffix_kpoints, kpoints_source_data, kpoints_boundary, kpoints_color_family, kpoints_line_style, kpoints_line_weight, kpoints_line_alpha = kpoints_info
    kpoints_color_family = get_or_default(kpoints_color_family, "default")
    kpoints_line_style = get_or_default(kpoints_line_style, "solid")
    kpoints_line_weight = get_or_default(kpoints_line_weight, 1.5)
    kpoints_line_alpha = get_or_default(kpoints_line_alpha, 1.0)

    # Unpack and handle missing values for encut_list
    encut_info = encut_list + [None] * (7 - len(encut_list))
    info_suffix_encut, encut_source_data, encut_boundary, encut_color_family, encut_line_style, encut_line_weight, encut_line_alpha = encut_info
    encut_color_family = get_or_default(encut_color_family, "default")
    encut_line_style = get_or_default(encut_line_style, "solid")
    encut_line_weight = get_or_default(encut_line_weight, 1.5)
    encut_line_alpha = get_or_default(encut_line_alpha, 1.0)

    # Set up figure and parameters
    fig_setting = canvas_setting()
    fig_size_adjusted = (fig_setting[0][0], fig_setting[0][1] * 1.25)
    fig, ax_kpoints = plt.subplots(figsize=fig_size_adjusted, dpi=fig_setting[1])
    ax_encut = ax_kpoints.twiny()  # Create the top x-axis

    # Update rcParams globally for plot settings
    params = fig_setting[2]
    plt.rcParams.update(params)

    # Configure tick direction and placement on both axes
    ax_kpoints.tick_params(direction="in")
    ax_encut.tick_params(direction="in")

    # Configure scientific notation for y-axis on K-points axis
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    ax_kpoints.yaxis.set_major_formatter(formatter)

    # Color sampling for K-points and ENCUT lines
    kpoints_colors = color_sampling(kpoints_color_family)
    encut_colors = color_sampling(encut_color_family)

    # K-points data processing
    kpoints_data = read_energy_parameters(kpoints_source_data)
    kpoints_energy = [d.get("total energy", d.get("Total Energy")) for d in kpoints_data]
    total_kpoints = [d.get("total kpoints", d.get("Total Kpoints")) for d in kpoints_data]
    directs_kpoints = [d.get("kpoints mesh", d.get("Kpoints(X Y Z)")) for d in kpoints_data]
    kpoints_labels = [f"({kp[0]}, {kp[1]}, {kp[2]})" for kp in directs_kpoints if kp is not None]

    # ENCUT data processing
    encut_data = read_energy_parameters(encut_source_data)
    encut_energy = [d.get("total energy", d.get("Total Energy")) for d in encut_data]
    encut_values = [d.get("energy cutoff (ENCUT)", d.get("ENCUT")) for d in encut_data]
    encut_filtered_data = [(enc, en) for enc, en in zip(encut_values, encut_energy) if enc is not None and en is not None]
    encut_sorted_data = sorted(encut_filtered_data, key=lambda x: x[0])
    encut_values_sorted, encut_energy_sorted = zip(*encut_sorted_data)

    # Set K-points boundary
    kpoints_start = int(kpoints_boundary[0]) if kpoints_boundary and kpoints_boundary[0] is not None else min(total_kpoints)
    kpoints_end = int(kpoints_boundary[1]) if kpoints_boundary and kpoints_boundary[1] is not None else max(total_kpoints)

    kpoints_indices = [
        index for index, val in enumerate(total_kpoints) if kpoints_start <= val <= kpoints_end
    ]
    kpoints_energy_plot = [kpoints_energy[index] for index in kpoints_indices]
    kpoints_labels_plot = [kpoints_labels[index] for index in kpoints_indices]

    # Set ENCUT boundary
    encut_start = float(encut_boundary[0]) if encut_boundary and encut_boundary[0] is not None else min(encut_values_sorted)
    encut_end = float(encut_boundary[1]) if encut_boundary and encut_boundary[1] is not None else max(encut_values_sorted)

    encut_indices = [
        index for index, val in enumerate(encut_values_sorted) if encut_start <= val <= encut_end
    ]
    encut_values_plot = [encut_values_sorted[index] for index in encut_indices]
    encut_energy_plot = [encut_energy_sorted[index] for index in encut_indices]

    # Plot K-points data with fixed spacing on x-axis
    ax_kpoints.plot(range(len(kpoints_energy_plot)), kpoints_energy_plot, c=kpoints_colors[1], ls=kpoints_line_style,
                    lw=kpoints_line_weight, alpha=kpoints_line_alpha)
    ax_kpoints.scatter(range(len(kpoints_energy_plot)), kpoints_energy_plot, s=kpoints_line_weight * 4, c=kpoints_colors[1],
                       zorder=1, alpha=kpoints_line_alpha)
    ax_kpoints.set_xlabel("K-points configuration", color=kpoints_colors[0])
    ax_kpoints.set_ylabel("Energy (eV)")
    ax_kpoints.set_xticks(range(len(kpoints_labels_plot)))
    ax_kpoints.set_xticklabels(kpoints_labels_plot, rotation=45, ha="right", color=kpoints_colors[0])

    # Plot ENCUT data on the top x-axis
    ax_encut.plot(encut_values_plot, encut_energy_plot, c=encut_colors[1], ls=encut_line_style, lw=encut_line_weight, alpha=encut_line_alpha)
    ax_encut.scatter(encut_values_plot, encut_energy_plot, s=encut_line_weight * 4, c=encut_colors[1], zorder=1, alpha=encut_line_alpha)
    ax_encut.set_xlabel("Energy cutoff (eV)", color=encut_colors[0])
    ax_encut.xaxis.set_label_position("top")
    ax_encut.xaxis.tick_top()
    for encut_label in ax_encut.get_xticklabels():
        encut_label.set_color(encut_colors[0])

    # Create unified legend
    kpoints_legend = mlines.Line2D([], [], color=kpoints_colors[1], marker='o', markersize=6, linestyle=kpoints_line_style,
                                   label=f"energy versus K-points {info_suffix_kpoints}")
    encut_legend = mlines.Line2D([], [], color=encut_colors[1], marker='o', markersize=6, linestyle=encut_line_style,
                                 label=f"energy versus energy cutoff {info_suffix_encut}")
    plt.legend(handles=[kpoints_legend, encut_legend], loc="best")

    plt.title(f"{suptitle}")
    plt.tight_layout()

def plot_energy_kpoints_encut(suptitle, kpoints_list_source, encut_list_source):
    """
    Generalized function to plot energy versus K-points and ENCUT (energy cutoff) for multiple datasets.

    Parameters:
    - kpoints_list_source: A list of lists, where each inner list contains:
      [info_suffix, source_data, kpoints_boundary, color_family, line_style, line_weight, line_alpha] for K-points data.
    - encut_list_source: A list of lists, where each inner list contains:
      [info_suffix, source_data, encut_boundary, color_family, line_style, line_weight, line_alpha] for ENCUT data.
    """

    # Ensure kpoints_list and encut_list are always nested lists
    kpoints_list = [kpoints_list_source] if not is_nested_list(kpoints_list_source) else kpoints_list_source
    encut_list = [encut_list_source] if not is_nested_list(encut_list_source) else encut_list_source

    # Handle single dataset cases
    if len(kpoints_list) == 1 and len(encut_list) == 1:
        return plot_energy_kpoints_encut_single(kpoints_list[0], encut_list[0])

    # Set up figure and parameters
    fig_setting = canvas_setting()
    fig_size_adjusted = (fig_setting[0][0], fig_setting[0][1] * 1.25)
    fig, ax_kpoints = plt.subplots(figsize=fig_size_adjusted, dpi=fig_setting[1])
    ax_encut = ax_kpoints.twiny()  # Create the top x-axis

    # Update rcParams globally for plot settings
    params = fig_setting[2]
    plt.rcParams.update(params)

    # Configure tick direction and placement on both axes
    ax_kpoints.tick_params(direction="in")
    ax_encut.tick_params(direction="in")

    # Configure scientific notation for y-axis on K-points axis
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    ax_kpoints.yaxis.set_major_formatter(formatter)

    # Initialize legend handles for unified legend creation
    legend_handles = []

    # Process K-points datasets
    global_kpoints = set()
    all_kpoints_labels = []
    for kpoints_data_item in kpoints_list:
        # Unpack and handle missing parameters
        kpoints_data_item += [None] * (7 - len(kpoints_data_item))
        info_suffix, kpoints_source_data, kpoints_boundary, kpoints_color_family, line_style, line_weight, line_alpha = kpoints_data_item
        kpoints_color_family = get_or_default(kpoints_color_family, "default")
        line_style = get_or_default(line_style, "solid")
        line_weight = get_or_default(line_weight, 1.5)
        line_alpha = get_or_default(line_alpha, 1.0)

        # K-points data processing
        kpoints_data = read_energy_parameters(kpoints_source_data)
        kpoints_energy = [d.get("total energy", d.get("Total Energy")) for d in kpoints_data]
        total_kpoints = [d.get("total kpoints", d.get("Total Kpoints")) for d in kpoints_data]
        directs_kpoints = [d.get("kpoints mesh", d.get("Kpoints(X Y Z)")) for d in kpoints_data]
        kpoints_labels = {kp: f"({coord[0]}, {coord[1]}, {coord[2]})" for kp, coord in zip(total_kpoints, directs_kpoints)}

        # Apply boundary to filter K-points
        kpoints_start = kpoints_boundary[0] if kpoints_boundary and kpoints_boundary[0] is not None else min(total_kpoints)
        kpoints_end = kpoints_boundary[1] if kpoints_boundary and kpoints_boundary[1] is not None else max(total_kpoints)
        filtered_data = [
            (kp, en, kpoints_labels[kp]) for kp, en in zip(total_kpoints, kpoints_energy)
            if kpoints_start <= kp <= kpoints_end
        ]

        # Sort filtered data and update global K-points
        filtered_data.sort(key=lambda x: x[0])
        filtered_kpoints, filtered_energy, filtered_labels = zip(*filtered_data)
        global_kpoints.update(filtered_kpoints)

        # Plot K-points data
        ax_kpoints.plot(range(len(filtered_kpoints)), filtered_energy, c=color_sampling(kpoints_color_family)[1],
                        ls=line_style, lw=line_weight, alpha=line_alpha)
        ax_kpoints.scatter(range(len(filtered_kpoints)), filtered_energy, s=line_weight * 4,
                           c=color_sampling(kpoints_color_family)[1], zorder=1, alpha=line_alpha)

        # Add to legend and labels
        legend_handles.append(mlines.Line2D([], [], color=color_sampling(kpoints_color_family)[1], marker='o', linestyle=line_style,
                                            label=f"Energy versus K-points {info_suffix}"))
        all_kpoints_labels.extend(filtered_labels)

    # Set x-axis labels for K-points
    all_kpoints_labels = sorted(set(all_kpoints_labels), key=all_kpoints_labels.index)
    ax_kpoints.set_xticks(range(len(all_kpoints_labels)))
    ax_kpoints.set_xticklabels(all_kpoints_labels, rotation=45, ha="right")
    ax_kpoints.set_xlabel("K-points configuration")
    ax_kpoints.set_ylabel("Energy (eV)")

    # Process ENCUT datasets
    for encut_data_item in encut_list:
        # Unpack and handle missing parameters
        encut_data_item += [None] * (7 - len(encut_data_item))
        info_suffix, encut_source_data, encut_boundary, encut_color_family, line_style, line_weight, line_alpha = encut_data_item
        encut_color_family = get_or_default(encut_color_family, "default")
        line_style = get_or_default(line_style, "solid")
        line_weight = get_or_default(line_weight, 1.5)
        line_alpha = get_or_default(line_alpha, 1.0)

        # ENCUT data processing
        encut_data = read_energy_parameters(encut_source_data)
        encut_energy = [d.get("total energy", d.get("Total Energy")) for d in encut_data]
        encut_values = [d.get("energy cutoff (ENCUT)", d.get("ENCUT")) for d in encut_data]
        encut_filtered_data = [(enc, en) for enc, en in zip(encut_values, encut_energy) if enc is not None and en is not None]
        encut_sorted_data = sorted(encut_filtered_data, key=lambda x: x[0])
        encut_values_sorted, encut_energy_sorted = zip(*encut_sorted_data)

        # Apply boundary to filter ENCUT
        encut_start = encut_boundary[0] if encut_boundary and encut_boundary[0] is not None else min(encut_values_sorted)
        encut_end = encut_boundary[1] if encut_boundary and encut_boundary[1] is not None else max(encut_values_sorted)
        encut_indices = [i for i, v in enumerate(encut_values_sorted) if encut_start <= v <= encut_end]
        encut_values_plot = [encut_values_sorted[i] for i in encut_indices]
        encut_energy_plot = [encut_energy_sorted[i] for i in encut_indices]

        # Plot ENCUT data
        ax_encut.plot(encut_values_plot, encut_energy_plot, c=color_sampling(encut_color_family)[1],
                      ls=line_style, lw=line_weight, alpha=line_alpha)
        ax_encut.scatter(encut_values_plot, encut_energy_plot, s=line_weight * 4,
                         c=color_sampling(encut_color_family)[1], zorder=1, alpha=line_alpha)

        # Add to legend
        legend_handles.append(mlines.Line2D([], [], color=color_sampling(encut_color_family)[1], marker='o', linestyle=line_style,
                                            label=f"Energy versus energy cutoff {info_suffix}"))

    # Set x-axis for ENCUT
    ax_encut.set_xlabel("Energy cutoff (eV)")
    ax_encut.xaxis.set_label_position("top")
    ax_encut.xaxis.tick_top()

    # Create unified legend
    ax_kpoints.legend(handles=legend_handles, loc="best")
    plt.title(f"{suptitle}")
    plt.tight_layout()

## Cohesive energy versus parameters

def plot_cohesive_energy_kpoints_single(suptitle, *args_list):
    help_info = (
        "Usage: plot_cohesive_energy_kpoints(suptitle, args_list)\n"
        "args_list: A list containing [info_suffix, source_data, kpoints_boundary, color_family, line_style, line_weight, line_alpha].\n"
        "Example: plot_cohesive_energy_kpoints(suptitle, ['Material Info', 'source_data_path', (start, end), 'blue', 'dashed', 2.0, 0.8])\n"
    )

    if not args_list or args_list[0] in ["HELP", "Help", "help"]:
        print(help_info)
        return

    # Unpack args_list and handle missing parameters
    args_info = args_list[0] + [None] * (7 - len(args_list[0]))
    info_suffix, source_data, kpoints_boundary, color_family, line_style, line_weight, line_alpha = args_info

    # Apply default values using `get_or_default`
    color_family = get_or_default(color_family, "default")
    line_style = get_or_default(line_style, "solid")
    line_weight = get_or_default(line_weight, 1.5)
    line_alpha = get_or_default(line_alpha, 1.0)

    # Figure settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Apply ScalarFormatter with scientific notation limits
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    # Color sampling
    colors = color_sampling(color_family)

    # Data input
    data_dict_list = read_energy_parameters(source_data)  # Now a list of dictionaries

    # Extract values based on keys in the dictionary
    total_kpoints = [d.get("total kpoints") for d in data_dict_list]
    cohesive_energy = [d.get("cohesive energy") for d in data_dict_list]
    sep_kpoints = [d.get("kpoints mesh") for d in data_dict_list]

    # Filter out None values
    filtered_data = [(k, e, kp) for k, e, kp in zip(total_kpoints, cohesive_energy, sep_kpoints) if k is not None and e is not None]
    if not filtered_data:
        print("No valid data found for plotting.")
        return

    # Sort data based on total_kpoints
    sorted_data = sorted(filtered_data, key=lambda x: x[0])
    total_kpoints_sorted, cohesive_energy_sorted, sep_kpoints_sorted = zip(*sorted_data)

    # Set title with info_suffix
    plt.title(f"{suptitle} {info_suffix}")
    plt.xlabel("K-points configuration")
    plt.ylabel("Cohesive energy (eV/atom)")

    # Ensure boundary values are integers and handle None boundaries
    kpoints_start = min(total_kpoints_sorted) if not kpoints_boundary or kpoints_boundary[0] is None else int(kpoints_boundary[0])
    kpoints_end = max(total_kpoints_sorted) if not kpoints_boundary or kpoints_boundary[1] is None else int(kpoints_boundary[1])

    # Filter data within the specified boundary
    kpoints_indices = [
        index for index, val in enumerate(total_kpoints_sorted) if kpoints_start <= val <= kpoints_end
    ]
    total_kpoints_plot = [total_kpoints_sorted[index] for index in kpoints_indices]
    cohesive_energy_plotting = [cohesive_energy_sorted[index] for index in kpoints_indices]
    kpoints_labels_plot = [f"({x[0]}, {x[1]}, {x[2]})" for x in [sep_kpoints_sorted[index] for index in kpoints_indices]]

    # Plot with fixed spacing based on indices
    plt.plot(range(len(total_kpoints_plot)), cohesive_energy_plotting, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha,
             label=f"Cohesive energy versus K-points {info_suffix}")
    plt.scatter(range(len(total_kpoints_plot)), cohesive_energy_plotting, s=line_weight * 4, c=colors[1], zorder=1, alpha=line_alpha)

    # Set custom tick labels for x-axis to show K-points configurations
    plt.xticks(ticks=range(len(kpoints_labels_plot)), labels=kpoints_labels_plot, rotation=45, ha="right")

    plt.legend(loc="best")
    plt.tight_layout()

def plot_cohesive_energy_kpoints(suptitle, kpoints_list):
    """
    Generalized function to plot cohesive energy versus K-points configuration for multiple datasets.

    Parameters:
    - kpoints_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, kpoints_boundary, color_family, line_style, line_weight, line_alpha].
    """

    # Check if input is a single data set (either single list or a list with one sublist)
    if not is_nested_list(kpoints_list):
        return plot_cohesive_energy_kpoints_single(suptitle, kpoints_list)
    elif len(kpoints_list) == 1:
        return plot_cohesive_energy_kpoints_single(suptitle, *kpoints_list)

    # Verify that each dataset in kpoints_list has the correct structure
    for index, data in enumerate(kpoints_list):
        if not isinstance(data, list):
            print(f"Error: Item at index {index} in kpoints_list must be a list.")
            return
        if len(data) < 4:
            print(f"Warning: Item at index {index} has less than 4 elements. Missing elements will be filled with None.")
            kpoints_list[index] += [None] * (7 - len(data))

    # Set up figure and styling
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Apply scientific notation formatter for y-axis
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    # Initialize legend handles
    legend_handles = []
    global_kpoints_config = []  # List to collect unique kpoints configurations for x-axis labels
    total_kpoints_set = set()  # Collect unique total_kpoints across all datasets for sorting

    # Iterate over each dataset in kpoints_list
    kpoints_config_map = {}
    for data in kpoints_list:
        # Unpack and handle missing parameters
        args_info = data + [None] * (7 - len(data))
        info_suffix, source_data, kpoints_boundary, color_family, line_style, line_weight, line_alpha = args_info
        color_family = get_or_default(color_family, "default")
        line_style = get_or_default(line_style, "solid")
        line_weight = get_or_default(line_weight, 1.5)
        line_alpha = get_or_default(line_alpha, 1.0)

        # Data processing
        data_dict_list = read_energy_parameters(source_data)
        total_kpoints = [d.get("total kpoints") for d in data_dict_list]
        cohesive_energy = [d.get("cohesive energy") for d in data_dict_list]
        kpoints_config = [d.get("kpoints mesh") for d in data_dict_list]

        # Apply boundary to filter relevant k-points
        kpoints_start = min(total_kpoints) if not kpoints_boundary or kpoints_boundary[0] is None else int(kpoints_boundary[0])
        kpoints_end = max(total_kpoints) if not kpoints_boundary or kpoints_boundary[1] is None else int(kpoints_boundary[1])

        # Filter and update global data
        for idx, total_kp in enumerate(total_kpoints):
            if kpoints_start <= total_kp <= kpoints_end:
                total_kpoints_set.add(total_kp)
                kpoints_config_map[total_kp] = f"({kpoints_config[idx][0]}, {kpoints_config[idx][1]}, {kpoints_config[idx][2]})"

    # Sort global total_kpoints for unified x-axis
    sorted_total_kpoints = sorted(total_kpoints_set)
    global_kpoints_config = [kpoints_config_map[total_kp] for total_kp in sorted_total_kpoints]

    # Iterate over each dataset to align with global total_kpoints and plot
    for data in kpoints_list:
        args_info = data + [None] * (7 - len(data))
        info_suffix, source_data, kpoints_boundary, color_family, line_style, line_weight, line_alpha = args_info

        # Data processing
        data_dict_list = read_energy_parameters(source_data)
        total_kpoints = [d.get("total kpoints") for d in data_dict_list]
        cohesive_energy = [d.get("cohesive energy") for d in data_dict_list]

        # Apply boundary to filter relevant data
        kpoints_start = min(total_kpoints) if not kpoints_boundary or kpoints_boundary[0] is None else int(kpoints_boundary[0])
        kpoints_end = max(total_kpoints) if not kpoints_boundary or kpoints_boundary[1] is None else int(kpoints_boundary[1])
        filtered_kpoints = [kp for kp in total_kpoints if kpoints_start <= kp <= kpoints_end]
        filtered_cohesive_energy = [cohesive_energy[idx] for idx, kp in enumerate(total_kpoints) if kpoints_start <= kp <= kpoints_end]

        # Align cohesive energies with global sorted total_kpoints
        cohesive_energy_aligned = [
            filtered_cohesive_energy[filtered_kpoints.index(kp)] if kp in filtered_kpoints else np.nan
            for kp in sorted_total_kpoints
        ]

        # Plot data
        colors = color_sampling(color_family)
        plt.plot(range(len(sorted_total_kpoints)), cohesive_energy_aligned, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha)
        plt.scatter(range(len(sorted_total_kpoints)), cohesive_energy_aligned, s=line_weight * 4, c=colors[1], alpha=line_alpha, zorder=1)

        # Add legend entry
        legend_handles.append(
            mlines.Line2D([], [], color=colors[1], marker='o', linestyle=line_style, markersize=6,
                          label=f"Cohesive energy versus K-points {info_suffix}")
        )

    # Configure axis labels, title, and legend
    plt.xlabel("K-points configurations")
    plt.ylabel("Cohesive energy (eV/atom)")
    plt.xticks(ticks=range(len(global_kpoints_config)), labels=global_kpoints_config, rotation=45, ha="right")
    plt.title(f"{suptitle}")
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

def plot_cohesive_energy_encut_single(suptitle, *args_list):
    help_info = (
        "Usage: plot_cohesive_energy_encut(suptitle, args_list)\n"
        "args_list: A list containing [info_suffix, source_data, encut_boundary, color_family, line_style, line_weight, line_alpha].\n"
        "Example: plot_cohesive_energy_encut(suptitle, ['Material Info', 'source_data_path', (start, end), 'violet', 'dashed', 2.0, 0.8])\n"
    )

    if not args_list or args_list[0] in ["HELP", "Help", "help"]:
        print(help_info)
        return

    # Unpack args_list and handle missing parameters
    args_info = args_list[0] + [None] * (7 - len(args_list[0]))
    info_suffix, source_data, encut_boundary, color_family, line_style, line_weight, line_alpha = args_info

    # Apply default values using `get_or_default`
    color_family = get_or_default(color_family, "default")
    line_style = get_or_default(line_style, "solid")
    line_weight = get_or_default(line_weight, 1.5)
    line_alpha = get_or_default(line_alpha, 1.0)

    # Figure settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Apply ScalarFormatter with scientific notation limits
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    # Color sampling
    colors = color_sampling(color_family)

    # Data input
    data_dict_list = read_energy_parameters(source_data)

    # Extract cohesive energy and ENCUT values
    cohesive_energy = [d.get("cohesive energy") for d in data_dict_list]
    encut_values = [d.get("energy cutoff (ENCUT)") for d in data_dict_list]

    # Filter out None values
    filtered_data = [(enc, e) for enc, e in zip(encut_values, cohesive_energy) if enc is not None and e is not None]
    if not filtered_data:
        print("No valid data found for plotting.")
        return

    # Unzip and sort data by ENCUT values
    encut_values, cohesive_energy = zip(*filtered_data)
    sorted_data = sorted(zip(encut_values, cohesive_energy), key=lambda x: x[0])
    encut_sorted, cohesive_energy_sorted = zip(*sorted_data)

    # Set boundaries for ENCUT
    encut_start = min(encut_sorted) if not encut_boundary or encut_boundary[0] is None else float(encut_boundary[0])
    encut_end = max(encut_sorted) if not encut_boundary or encut_boundary[1] is None else float(encut_boundary[1])

    # Filter data within the specified boundary
    encut_filtered, energy_filtered = zip(*[
        (enc, energy) for enc, energy in zip(encut_sorted, cohesive_energy_sorted)
        if encut_start <= enc <= encut_end
    ])

    # Plotting
    plt.plot(encut_filtered, energy_filtered, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha,
             label=f"Cohesive energy versus energy cutoff {info_suffix}")
    plt.scatter(encut_filtered, energy_filtered, s=line_weight * 4, c=colors[1], zorder=1, alpha=line_alpha)

    # Set labels and title
    plt.title(f"{suptitle} {info_suffix}")
    plt.xlabel("Energy cutoff (eV)")
    plt.ylabel("Cohesive energy (eV/atom)")
    plt.legend(loc="best")
    plt.tight_layout()

def plot_cohesive_energy_encut(suptitle, encut_list):
    """
    Generalized function to plot cohesive energy versus ENCUT (energy cutoff) for multiple datasets.

    Parameters:
    - encut_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, encut_boundary, color_family, line_style, line_weight, line_alpha].
    """

    # Check if input is a single data set (either single list or a list with one sublist)
    if not is_nested_list(encut_list):
        return plot_cohesive_energy_encut_single(suptitle, encut_list)
    elif len(encut_list) == 1:
        return plot_cohesive_energy_encut_single(suptitle, *encut_list)

    # Verify that each dataset in encut_list has the correct structure
    for index, data in enumerate(encut_list):
        if not isinstance(data, list):
            print(f"Error: Item at index {index} in encut_list must be a list.")
            return
        if len(data) < 4:
            print(f"Warning: Item at index {index} has less than 4 elements. Missing elements will be filled with None.")
            encut_list[index] += [None] * (7 - len(data))

    # Set up figure and styling
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Apply scientific notation formatter for y-axis
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    # Initialize legend handles for custom legend creation
    legend_handles = []
    global_encut_set = set()  # Collect unique ENCUT values across all datasets

    # Iterate over each dataset in encut_list to gather ENCUT values
    for data in encut_list:
        # Unpack and handle missing parameters
        args_info = data + [None] * (7 - len(data))
        info_suffix, source_data, encut_boundary, color_family, line_style, line_weight, line_alpha = args_info
        color_family = get_or_default(color_family, "default")
        line_style = get_or_default(line_style, "solid")
        line_weight = get_or_default(line_weight, 1.5)
        line_alpha = get_or_default(line_alpha, 1.0)

        # Data processing
        data_dict_list = read_energy_parameters(source_data)
        encut_values = [d.get("energy cutoff (ENCUT)") for d in data_dict_list]

        # Apply boundary to filter ENCUT values
        encut_start = min(encut_values) if not encut_boundary or encut_boundary[0] is None else float(encut_boundary[0])
        encut_end = max(encut_values) if not encut_boundary or encut_boundary[1] is None else float(encut_boundary[1])
        filtered_encut = [enc for enc in encut_values if encut_start <= enc <= encut_end]

        # Add filtered ENCUT values to global set
        global_encut_set.update(filtered_encut)

    # Sort global ENCUT values for unified x-axis
    global_encut_sorted = sorted(global_encut_set)

    # Iterate over each dataset to align with global ENCUT and plot
    for data in encut_list:
        args_info = data + [None] * (7 - len(data))
        info_suffix, source_data, encut_boundary, color_family, line_style, line_weight, line_alpha = args_info

        # Data processing
        data_dict_list = read_energy_parameters(source_data)
        cohesive_energy = [d.get("cohesive energy") for d in data_dict_list]
        encut_values = [d.get("energy cutoff (ENCUT)") for d in data_dict_list]

        # Apply boundary values
        encut_start = min(encut_values) if not encut_boundary or encut_boundary[0] is None else float(encut_boundary[0])
        encut_end = max(encut_values) if not encut_boundary or encut_boundary[1] is None else float(encut_boundary[1])

        # Filter data within boundary
        filtered_encut = [enc for enc in encut_values if encut_start <= enc <= encut_end]
        cohesive_energy_filtered = [cohesive_energy[idx] for idx, enc in enumerate(encut_values) if encut_start <= enc <= encut_end]

        # Align cohesive energies with global sorted ENCUT
        cohesive_energy_aligned = [
            cohesive_energy_filtered[filtered_encut.index(enc)] if enc in filtered_encut else np.nan
            for enc in global_encut_sorted
        ]

        # Plot data
        colors = color_sampling(color_family)
        plt.plot(global_encut_sorted, cohesive_energy_aligned, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha)
        plt.scatter(global_encut_sorted, cohesive_energy_aligned, s=line_weight * 4, c=colors[1], zorder=1, alpha=line_alpha)

        # Add legend entry
        legend_handles.append(
            mlines.Line2D([], [], color=colors[1], marker='o', linestyle=line_style, markersize=6,
                          label=f"Cohesive energy versus energy cutoff {info_suffix}")
        )

    # Set labels, title, and legend
    plt.xlabel("Energy cutoff (eV)")
    plt.ylabel("Cohesive energy (eV/atom)")
    plt.title(f"{suptitle}")
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

def plot_cohesive_energy_lattice_single(suptitle, *args_list):
    help_info = (
        "Usage: plot_cohesive_energy_lattice_single(suptitle, args_list)\n"
        "args_list: A list containing [info_suffix, source_data, lattice_boundary, num_samples, color_family, line_style, line_weight, line_alpha].\n"
        "Example: plot_cohesive_energy_lattice_single(suptitle, ['Material Info', 'source_data_path', (start, end), 11, 'green', 'dashed', 2.0, 0.8])\n"
    )

    # Check if the user requested help
    if not args_list or args_list[0] in ["HELP", "Help", "help"]:
        print(help_info)
        return

    # Unpack args_list and handle missing parameters
    args_info = args_list[0] + [None] * (8 - len(args_list[0]))
    info_suffix, source_data, lattice_boundary, num_samples, color_family, line_style, line_weight, line_alpha = args_info

    # Validate `num_samples`
    if num_samples is None or not isinstance(num_samples, int) or num_samples <= 0:
        num_samples = 100  # Default sample count

    # Apply default values
    color_family = get_or_default(color_family, "default")
    line_style = get_or_default(line_style, "solid")
    line_weight = get_or_default(line_weight, 1.5)
    line_alpha = get_or_default(line_alpha, 1.0)

    # Figure settings
    fig_setting = canvas_setting()
    params = fig_setting[2]
    plt.rcParams.update(params)
    fig, ax = plt.subplots(figsize=fig_setting[0], dpi=fig_setting[1])
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Configure scientific notation for y-axis
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)

    # Data input
    data_dict_list = read_energy_parameters(source_data)
    cohesive_energy_values = [d.get("cohesive energy") for d in data_dict_list]
    lattice_constants = [d.get("lattice constant") for d in data_dict_list]

    # Filter out None values
    filtered_data = [(lattice, energy) for lattice, energy in zip(lattice_constants, cohesive_energy_values)
                     if lattice is not None and energy is not None]

    if not filtered_data:
        print("No valid data found for plotting.")
        return

    # Unzip filtered data and sort
    lattice_constants, cohesive_energy_values = zip(*filtered_data)
    sorted_data = sorted(zip(lattice_constants, cohesive_energy_values), key=lambda x: x[0])
    lattice_sorted, cohesive_sorted = zip(*sorted_data)

    # Define boundaries for lattice constants
    lattice_start = min(lattice_sorted) if lattice_boundary in [None, ""] else float(lattice_boundary[0] or min(lattice_sorted))
    lattice_end = max(lattice_sorted) if lattice_boundary in [None, ""] else float(lattice_boundary[1] or max(lattice_sorted))

    # Filter data within the specified boundary
    lattice_filtered, cohesive_filtered = zip(*[(l, e) for l, e in zip(lattice_sorted, cohesive_sorted) if lattice_start <= l <= lattice_end])

    # Estimate EOS parameters and fitted energy values using all data
    eos_params, resampled_lattice, fitted_energy = fit_birch_murnaghan(lattice_filtered, cohesive_filtered, sample_count=100)

    # Plot the fitted EOS curve
    ax.plot(resampled_lattice, fitted_energy, color=color_sampling(color_family)[1], ls=line_style, lw=line_weight, alpha=line_alpha,
            label=f"Fitted EOS curve {info_suffix}")

    # Select scatter sample points
    if num_samples >= len(lattice_filtered):
        scatter_lattice, scatter_energy = lattice_filtered, cohesive_filtered
    else:
        x_samples = np.linspace(lattice_start, lattice_end, num_samples)
        scatter_lattice, scatter_energy = zip(*[(l, e) for x in x_samples for l, e in zip(lattice_filtered, cohesive_filtered)
                                                 if np.abs(l - x) == min(np.abs(np.array(lattice_filtered) - x))])

    # Scatter sample data points
    ax.scatter(scatter_lattice, scatter_energy, s=line_weight * 4, fc="#FFFFFF", ec=color_sampling(color_family)[1], alpha=line_alpha,
               label=f"Sampled data {info_suffix}", zorder=2)

    # Mark the minimum cohesive energy point
    min_energy_idx = np.argmin(cohesive_filtered)
    ax.scatter(lattice_filtered[min_energy_idx], cohesive_filtered[min_energy_idx], s=line_weight * 4,
               fc=color_sampling(color_family)[2], ec=color_sampling(color_family)[2],
               label=f"Minimum energy at {lattice_filtered[min_energy_idx]:.6g} {info_suffix}", zorder=3)

    # Set labels, title, and legend
    ax.set_xlabel(r"Lattice constant (Å)")
    ax.set_ylabel(r"Cohesive energy (eV/atom)")
    ax.set_title(f"{suptitle} {info_suffix}")
    ax.legend()
    plt.tight_layout()

def plot_cohesive_energy_lattice(suptitle, lattice_list):
    """
    Generalized function to plot cohesive energy versus lattice constant for multiple datasets.

    Parameters:
    - lattice_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, lattice_boundary, num_samples, color_family, line_style, line_weight, line_alpha].
    """
    # Single dataset case
    if not isinstance(lattice_list[0], list):
        return plot_cohesive_energy_lattice_single(suptitle, lattice_list)
    elif len(lattice_list) == 1:
        return plot_cohesive_energy_lattice_single(suptitle, *lattice_list)

    # Ensure all inner lists have at least 8 elements
    for index, data in enumerate(lattice_list):
        if len(data) < 8:
            lattice_list[index] += [None] * (8 - len(data))

    # Figure settings
    fig_setting = canvas_setting()
    params = fig_setting[2]
    plt.rcParams.update(params)
    fig, ax = plt.subplots(figsize=fig_setting[0], dpi=fig_setting[1])
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Configure scientific notation for y-axis
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)

    # Process each dataset in lattice_list
    for idx, data in enumerate(lattice_list):
        info_suffix, source_data, lattice_boundary, num_samples, color_family, line_style, line_weight, line_alpha = data

        # Validate `source_data`
        if not source_data or not isinstance(source_data, str):
            print(f"Invalid source_data for dataset {idx}. Skipping.")
            continue

        # Validate `num_samples`
        if num_samples is None or not isinstance(num_samples, int) or num_samples <= 0:
            num_samples = 100  # Default sample count

        # Apply default values
        color_family = get_or_default(color_family, "default")
        line_style = get_or_default(line_style, "solid")
        line_weight = get_or_default(line_weight, 1.5)
        line_alpha = get_or_default(line_alpha, 1.0)

        # Read data
        data_dict_list = read_energy_parameters(source_data)
        cohesive_energy_values = [d.get("cohesive energy") for d in data_dict_list]
        lattice_constants = [d.get("lattice constant") for d in data_dict_list]

        # Filter and sort data
        filtered_data = [(lattice, energy) for lattice, energy in zip(lattice_constants, cohesive_energy_values)
                         if lattice is not None and energy is not None]
        if not filtered_data:
            print(f"Warning: No valid data for dataset {info_suffix}. Skipping.")
            continue
        lattice_sorted, cohesive_sorted = zip(*sorted(filtered_data))

        # Define boundaries
        lattice_start = min(lattice_sorted) if lattice_boundary in [None, ""] else float(lattice_boundary[0])
        lattice_end = max(lattice_sorted) if lattice_boundary in [None, ""] else float(lattice_boundary[1])

        # Filter data within boundary
        lattice_filtered = [l for l in lattice_sorted if lattice_start <= l <= lattice_end]
        cohesive_filtered = [cohesive_sorted[i] for i, l in enumerate(lattice_sorted) if lattice_start <= l <= lattice_end]

        if not lattice_filtered:
            print(f"No data within the specified lattice boundary for dataset {info_suffix}.")
            continue

        # Estimate EOS parameters and fitted energy values using all data
        eos_params, resampled_lattice, fitted_energy = fit_birch_murnaghan(lattice_filtered, cohesive_filtered, sample_count=100)

        # Plot the fitted EOS curve
        ax.plot(resampled_lattice, fitted_energy, color=color_sampling(color_family)[1], ls=line_style, lw=line_weight, alpha=line_alpha,
                label=f"Fitted EOS curve {info_suffix}")

        # Select scatter sample points
        if num_samples >= len(lattice_filtered):
            scatter_lattice, scatter_energy = lattice_filtered, cohesive_filtered
        else:
            x_samples = np.linspace(lattice_start, lattice_end, num_samples)
            scatter_lattice, scatter_energy = zip(*[(l, e) for x in x_samples for l, e in zip(lattice_filtered, cohesive_filtered)
                                                     if np.abs(l - x) == min(np.abs(np.array(lattice_filtered) - x))])

        # Scatter sample data points
        ax.scatter(scatter_lattice, scatter_energy, s=line_weight * 4, fc="#FFFFFF", ec=color_sampling(color_family)[1], alpha=line_alpha,
                   label=f"Sampled data {info_suffix}", zorder=2)

        # Mark the minimum cohesive energy point
        min_energy_idx = np.argmin(cohesive_filtered)
        ax.scatter(lattice_filtered[min_energy_idx], cohesive_filtered[min_energy_idx], s=line_weight * 4,
                   fc=color_sampling(color_family)[2], ec=color_sampling(color_family)[2],
                   label=f"Minimum energy at {lattice_filtered[min_energy_idx]:.6g} {info_suffix}", zorder=3)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="best")

    # Set labels and title
    ax.set_xlabel("Lattice constant (Å)")
    ax.set_ylabel("Cohesive energy (eV/atom)")
    ax.set_title(suptitle)
    plt.tight_layout()

def plot_cohesive_energy_scaling_single(suptitle, *args_list):
    help_info = (
        "Usage: plot_cohesive_energy_scaling(suptitle, args_list)\n"
        "args_list: A list containing [info_suffix, source_data, scaling_boundary, color_family, line_style, line_weight, line_alpha].\n"
        "Example: plot_cohesive_energy_scaling(suptitle, ['Material Info', 'source_data_path', (start, end), 'green', 'dashed', 2.0, 0.8])\n"
    )

    if not args_list or args_list[0] in ["HELP", "Help", "help"]:
        print(help_info)
        return

    # Unpack args_list and handle missing parameters
    args_info = args_list[0] + [None] * (7 - len(args_list[0]))
    info_suffix, source_data, scaling_boundary, color_family, line_style, line_weight, line_alpha = args_info

    # Apply default values using `get_or_default`
    color_family = get_or_default(color_family, "default")
    line_style   = get_or_default(line_style, "solid")
    line_weight  = get_or_default(line_weight, 1.5)
    line_alpha   = get_or_default(line_alpha, 1.0)

    # Figure settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Apply ScalarFormatter with scientific notation limits
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    # Color sampling
    colors = color_sampling(color_family)

    # Data input
    data_dict_list = read_energy_parameters(source_data)

    # Extract cohesive energy and Scaling values
    cohesive_energy = [d.get("cohesive energy") for d in data_dict_list]
    scaling_values = [d.get("Scaling") for d in data_dict_list]

    # Filter out None values
    filtered_data = [(s, e) for s, e in zip(scaling_values, cohesive_energy) if s is not None and e is not None]
    if not filtered_data:
        print("No valid data found for plotting.")
        return

    # Unzip and sort data by Scaling values
    scaling_values, cohesive_energy = zip(*filtered_data)
    sorted_data = sorted(zip(scaling_values, cohesive_energy), key=lambda x: x[0])
    scaling_sorted, cohesive_energy_sorted = zip(*sorted_data)

    # Set boundaries for Scaling
    scaling_start = min(scaling_sorted) if not scaling_boundary or scaling_boundary[0] is None else float(scaling_boundary[0])
    scaling_end   = max(scaling_sorted) if not scaling_boundary or scaling_boundary[1] is None else float(scaling_boundary[1])

    # Filter data within the specified boundary
    scaling_filtered, energy_filtered = zip(*[
        (s, e) for s, e in zip(scaling_sorted, cohesive_energy_sorted) if scaling_start <= s <= scaling_end
    ])

    # Plotting
    plt.plot(scaling_filtered, energy_filtered, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha,
             label=f"Cohesive energy versus Scaling {info_suffix}")
    plt.scatter(scaling_filtered, energy_filtered, s=line_weight * 4, c=colors[1], zorder=1, alpha=line_alpha)

    # Set labels and title
    plt.title(f"{suptitle} {info_suffix}")
    plt.xlabel("Scaling")
    plt.ylabel("Cohesive energy (eV/atom)")
    plt.legend(loc="best")
    plt.tight_layout()


def plot_cohesive_energy_scaling(suptitle, scaling_list):
    """
    Generalized function to plot cohesive energy versus Scaling for multiple datasets.

    Parameters:
    - scaling_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, scaling_boundary, color_family, line_style, line_weight, line_alpha].
    """
    if not is_nested_list(scaling_list):
        return plot_cohesive_energy_scaling_single(suptitle, scaling_list)
    elif len(scaling_list) == 1:
        return plot_cohesive_energy_scaling_single(suptitle, *scaling_list)

    # Verify structure of each dataset
    for index, data in enumerate(scaling_list):
        if not isinstance(data, list):
            print(f"Error: Item at index {index} in scaling_list must be a list.")
            return
        if len(data) < 4:
            print(f"Warning: Item at index {index} has less than 4 elements. Missing elements will be filled with None.")
            scaling_list[index] += [None] * (7 - len(data))

    # Figure settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    legend_handles = []
    global_scaling_set = set()
    scaling_config_map = {}

    # Gather global Scaling values from all datasets
    for data in scaling_list:
        args_info = data + [None] * (7 - len(data))
        _, source_data, scaling_boundary, color_family, line_style, line_weight, line_alpha = args_info
        color_family = get_or_default(color_family, "default")
        line_style   = get_or_default(line_style, "solid")
        line_weight  = get_or_default(line_weight, 1.5)
        line_alpha   = get_or_default(line_alpha, 1.0)
        data_dict_list = read_energy_parameters(source_data)
        scaling_values = [d.get("Scaling") for d in data_dict_list]
        scaling_start = min(scaling_values) if not scaling_boundary or scaling_boundary[0] is None else float(scaling_boundary[0])
        scaling_end   = max(scaling_values) if not scaling_boundary or scaling_boundary[1] is None else float(scaling_boundary[1])
        for val in scaling_values:
            if val is not None and scaling_start <= val <= scaling_end:
                global_scaling_set.add(val)
                scaling_config_map[val] = f"{val:.6g}"
    sorted_global_scaling = sorted(global_scaling_set)
    global_scaling_labels = [scaling_config_map[val] for val in sorted_global_scaling]

    # Plot each dataset aligned with the global Scaling values
    for data in scaling_list:
        args_info = data + [None] * (7 - len(data))
        info_suffix, source_data, scaling_boundary, color_family, line_style, line_weight, line_alpha = args_info
        color_family = get_or_default(color_family, "default")
        line_style   = get_or_default(line_style, "solid")
        line_weight  = get_or_default(line_weight, 1.5)
        line_alpha   = get_or_default(line_alpha, 1.0)
        colors = color_sampling(color_family)
        data_dict_list = read_energy_parameters(source_data)
        energy = [d.get("total energy") for d in data_dict_list]
        scaling_values = [d.get("Scaling") for d in data_dict_list]
        scaling_start = min(scaling_values) if not scaling_boundary or scaling_boundary[0] is None else float(scaling_boundary[0])
        scaling_end   = max(scaling_values) if not scaling_boundary or scaling_boundary[1] is None else float(scaling_boundary[1])
        filtered = [(s, e) for s, e in zip(scaling_values, energy) if s is not None and scaling_start <= s <= scaling_end]
        if not filtered:
            continue
        scaling_filtered, energy_filtered = zip(*sorted(filtered, key=lambda x: x[0]))
        energy_aligned = [energy_filtered[scaling_filtered.index(val)] if val in scaling_filtered else np.nan for val in sorted_global_scaling]
        plt.plot(sorted_global_scaling, energy_aligned, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha)
        plt.scatter(sorted_global_scaling, energy_aligned, s=line_weight * 4, c=colors[1], zorder=1, alpha=line_alpha)
        legend_handles.append(
            mlines.Line2D([], [], color=colors[1], marker='o', markersize=6, linestyle=line_style,
                          label=f"Cohesive energy versus Scaling {info_suffix}")
        )

    plt.xlabel("Scaling")
    plt.ylabel("Cohesive energy (eV/atom)")
    plt.xticks(ticks=range(len(global_scaling_labels)), labels=global_scaling_labels)
    plt.title(f"{suptitle}")
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

def plot_cohesive_energy_a1_single(suptitle, *args_list):
    help_info = (
        "Usage: plot_cohesive_energy_a1(suptitle, args_list)\n"
        "args_list: A list containing [info_suffix, source_data, a1_boundary, color_family, line_style, line_weight, line_alpha].\n"
        "Example: plot_cohesive_energy_a1(suptitle, ['Material Info', 'source_data_path', (start, end), 'blue', 'solid', 1.5, 1.0])\n"
    )

    if not args_list or args_list[0] in ["HELP", "Help", "help"]:
        print(help_info)
        return

    # Unpack args_list and handle missing parameters
    args_info = args_list[0] + [None] * (7 - len(args_list[0]))
    info_suffix, source_data, a1_boundary, color_family, line_style, line_weight, line_alpha = args_info

    # Apply default values using `get_or_default`
    color_family = get_or_default(color_family, "default")
    line_style   = get_or_default(line_style, "solid")
    line_weight  = get_or_default(line_weight, 1.5)
    line_alpha   = get_or_default(line_alpha, 1.0)

    # Figure settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Apply ScalarFormatter with scientific notation limits
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    # Color sampling
    colors = color_sampling(color_family)

    # Data input
    data_dict_list = read_energy_parameters(source_data)

    # Extract cohesive energy and a1 values
    cohesive_energy = [d.get("cohesive energy") for d in data_dict_list]
    a1_values = [d.get("a1") for d in data_dict_list]

    # Filter out None values
    filtered_data = [(val, e) for val, e in zip(a1_values, cohesive_energy) if val is not None and e is not None]
    if not filtered_data:
        print("No valid data found for plotting.")
        return

    # Unzip and sort data by a1 values
    a1_values, cohesive_energy = zip(*filtered_data)
    sorted_data = sorted(zip(a1_values, cohesive_energy), key=lambda x: x[0])
    a1_sorted, cohesive_energy_sorted = zip(*sorted_data)

    # Set boundaries for a1
    a1_start = min(a1_sorted) if not a1_boundary or a1_boundary[0] is None else float(a1_boundary[0])
    a1_end   = max(a1_sorted) if not a1_boundary or a1_boundary[1] is None else float(a1_boundary[1])

    # Filter data within the specified boundary
    a1_filtered, energy_filtered = zip(*[
        (val, e) for val, e in zip(a1_sorted, cohesive_energy_sorted) if a1_start <= val <= a1_end
    ])

    # Plotting
    plt.plot(a1_filtered, energy_filtered, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha,
             label=f"Cohesive energy versus a1 {info_suffix}")
    plt.scatter(a1_filtered, energy_filtered, s=line_weight * 4, c=colors[1], zorder=1, alpha=line_alpha)

    # Set labels and title
    plt.title(f"{suptitle} {info_suffix}")
    plt.xlabel("a1 (Å)")
    plt.ylabel("Cohesive energy (eV/atom)")
    plt.legend(loc="best")
    plt.tight_layout()

def plot_cohesive_energy_a1(suptitle, a1_list):
    """
    Generalized function to plot cohesive energy versus a1 for multiple datasets.

    Parameters:
    - a1_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, a1_boundary, color_family, line_style, line_weight, line_alpha].
    """
    if not is_nested_list(a1_list):
        return plot_cohesive_energy_a1_single(suptitle, a1_list)
    elif len(a1_list) == 1:
        return plot_cohesive_energy_a1_single(suptitle, *a1_list)

    for index, data in enumerate(a1_list):
        if not isinstance(data, list):
            print(f"Error: Item at index {index} in a1_list must be a list.")
            return
        if len(data) < 4:
            print(f"Warning: Item at index {index} has less than 4 elements. Missing elements will be filled with None.")
            a1_list[index] += [None] * (7 - len(data))

    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    legend_handles = []
    global_a1_set = set()
    a1_config_map = {}

    # Gather all unique a1 values for the global x-axis
    for data in a1_list:
        args_info = data + [None] * (7 - len(data))
        _, source_data, a1_boundary, color_family, line_style, line_weight, line_alpha = args_info
        data_dict_list = read_energy_parameters(source_data)
        a1_values = [d.get("a1") for d in data_dict_list]
        a1_start = min(a1_values) if not a1_boundary or a1_boundary[0] is None else float(a1_boundary[0])
        a1_end   = max(a1_values) if not a1_boundary or a1_boundary[1] is None else float(a1_boundary[1])
        for val in a1_values:
            if val is not None and a1_start <= val <= a1_end:
                global_a1_set.add(val)
                a1_config_map[val] = f"{val:.6g}"
    sorted_global_a1 = sorted(global_a1_set)
    global_a1_labels = [a1_config_map[val] for val in sorted_global_a1]

    # Plot each dataset aligned with the global a1 values
    for data in a1_list:
        args_info = data + [None] * (7 - len(data))
        info_suffix, source_data, a1_boundary, color_family, line_style, line_weight, line_alpha = args_info

        color_family = get_or_default(color_family, "default")
        line_style   = get_or_default(line_style, "solid")
        line_weight  = get_or_default(line_weight, 1.5)
        line_alpha   = get_or_default(line_alpha, 1.0)
        colors = color_sampling(color_family)
        data_dict_list = read_energy_parameters(source_data)
        energy = [d.get("total energy") for d in data_dict_list]
        a1_values = [d.get("a1") for d in data_dict_list]
        a1_start = min(a1_values) if not a1_boundary or a1_boundary[0] is None else float(a1_boundary[0])
        a1_end   = max(a1_values) if not a1_boundary or a1_boundary[1] is None else float(a1_boundary[1])
        filtered = [(val, e) for val, e in zip(a1_values, energy) if val is not None and a1_start <= val <= a1_end]
        if not filtered:
            continue
        a1_filtered, energy_filtered = zip(*sorted(filtered, key=lambda x: x[0]))
        energy_aligned = [energy_filtered[a1_filtered.index(val)] if val in a1_filtered else np.nan for val in sorted_global_a1]
        plt.plot(sorted_global_a1, energy_aligned, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha,
                 label=f"Cohesive energy versus a1 {info_suffix}")
        plt.scatter(sorted_global_a1, energy_aligned, s=line_weight * 4, c=colors[1], alpha=line_alpha, zorder=1)
        legend_handles.append(
            mlines.Line2D([], [], color=colors[1], marker='o', markersize=6, linestyle=line_style,
                          label=f"Cohesive energy versus a1 {info_suffix}")
        )

    plt.xlabel("a1 (Å)")
    plt.ylabel("Cohesive energy (eV/atom)")
    plt.xticks(ticks=range(len(global_a1_labels)), labels=global_a1_labels)
    plt.title(f"{suptitle}")
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

def plot_cohesive_energy_a2_single(suptitle, *args_list):
    help_info = (
        "Usage: plot_cohesive_energy_a2(suptitle, args_list)\n"
        "args_list: A list containing [info_suffix, source_data, a2_boundary, color_family, line_style, line_weight, line_alpha].\n"
        "Example: plot_cohesive_energy_a2(suptitle, ['Material Info', 'source_data_path', (start, end), 'red', 'dashed', 2.0, 0.8])\n"
    )

    if not args_list or args_list[0] in ["HELP", "Help", "help"]:
        print(help_info)
        return

    # Unpack args_list and handle missing parameters
    args_info = args_list[0] + [None] * (7 - len(args_list[0]))
    info_suffix, source_data, a2_boundary, color_family, line_style, line_weight, line_alpha = args_info

    # Apply default values using `get_or_default`
    color_family = get_or_default(color_family, "default")
    line_style   = get_or_default(line_style, "solid")
    line_weight  = get_or_default(line_weight, 1.5)
    line_alpha   = get_or_default(line_alpha, 1.0)

    # Figure settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Apply ScalarFormatter with scientific notation limits
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    # Color sampling
    colors = color_sampling(color_family)

    # Data input
    data_dict_list = read_energy_parameters(source_data)

    # Extract cohesive energy and a2 values
    cohesive_energy = [d.get("cohesive energy") for d in data_dict_list]
    a2_values = [d.get("a2") for d in data_dict_list]

    # Filter out None values
    filtered_data = [(val, e) for val, e in zip(a2_values, cohesive_energy) if val is not None and e is not None]
    if not filtered_data:
        print("No valid data found for plotting.")
        return

    # Unzip and sort data by a2 values
    a2_values, cohesive_energy = zip(*filtered_data)
    sorted_data = sorted(zip(a2_values, cohesive_energy), key=lambda x: x[0])
    a2_sorted, cohesive_energy_sorted = zip(*sorted_data)

    # Set boundaries for a2
    a2_start = min(a2_sorted) if not a2_boundary or a2_boundary[0] is None else float(a2_boundary[0])
    a2_end   = max(a2_sorted) if not a2_boundary or a2_boundary[1] is None else float(a2_boundary[1])

    # Filter data within the specified boundary
    a2_filtered, energy_filtered = zip(*[
        (val, e) for val, e in zip(a2_sorted, cohesive_energy_sorted) if a2_start <= val <= a2_end
    ])

    # Plotting
    plt.plot(a2_filtered, energy_filtered, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha,
             label=f"Cohesive energy versus a2 {info_suffix}")
    plt.scatter(a2_filtered, energy_filtered, s=line_weight * 4, c=colors[1], zorder=1, alpha=line_alpha)

    # Set labels and title
    plt.title(f"{suptitle} {info_suffix}")
    plt.xlabel("a2 (Å)")
    plt.ylabel("Cohesive energy (eV/atom)")
    plt.legend(loc="best")
    plt.tight_layout()

def plot_cohesive_energy_a2(suptitle, a2_list):
    """
    Generalized function to plot cohesive energy versus a2 for multiple datasets.

    Parameters:
    - a2_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, a2_boundary, color_family, line_style, line_weight, line_alpha].
    """
    if not is_nested_list(a2_list):
        return plot_cohesive_energy_a2_single(suptitle, a2_list)
    elif len(a2_list) == 1:
        return plot_cohesive_energy_a2_single(suptitle, *a2_list)

    for index, data in enumerate(a2_list):
        if not isinstance(data, list):
            print(f"Error: Item at index {index} in a2_list must be a list.")
            return
        if len(data) < 4:
            print(f"Warning: Item at index {index} has less than 4 elements. Missing elements will be filled with None.")
            a2_list[index] += [None] * (7 - len(data))

    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    legend_handles = []
    global_a2_set = set()
    a2_config_map = {}

    # Gather all unique a2 values for the global x-axis
    for data in a2_list:
        args_info = data + [None] * (7 - len(data))
        _, source_data, a2_boundary, color_family, line_style, line_weight, line_alpha = args_info
        data_dict_list = read_energy_parameters(source_data)
        a2_values = [d.get("a2") for d in data_dict_list]
        a2_start = min(a2_values) if not a2_boundary or a2_boundary[0] is None else float(a2_boundary[0])
        a2_end   = max(a2_values) if not a2_boundary or a2_boundary[1] is None else float(a2_boundary[1])
        for val in a2_values:
            if val is not None and a2_start <= val <= a2_end:
                global_a2_set.add(val)
                a2_config_map[val] = f"{val:.6g}"
    sorted_global_a2 = sorted(global_a2_set)
    global_a2_labels = [a2_config_map[val] for val in sorted_global_a2]

    # Plot each dataset aligned with the global a2 values
    for data in a2_list:
        args_info = data + [None] * (7 - len(data))
        info_suffix, source_data, a2_boundary, color_family, line_style, line_weight, line_alpha = args_info

        color_family = get_or_default(color_family, "default")
        line_style   = get_or_default(line_style, "solid")
        line_weight  = get_or_default(line_weight, 1.5)
        line_alpha   = get_or_default(line_alpha, 1.0)
        colors = color_sampling(color_family)
        data_dict_list = read_energy_parameters(source_data)
        energy = [d.get("total energy") for d in data_dict_list]
        a2_values = [d.get("a2") for d in data_dict_list]
        a2_start = min(a2_values) if not a2_boundary or a2_boundary[0] is None else float(a2_boundary[0])
        a2_end   = max(a2_values) if not a2_boundary or a2_boundary[1] is None else float(a2_boundary[1])
        filtered = [(val, e) for val, e in zip(a2_values, energy) if val is not None and a2_start <= val <= a2_end]
        if not filtered:
            continue
        a2_filtered, energy_filtered = zip(*sorted(filtered, key=lambda x: x[0]))
        energy_aligned = [energy_filtered[a2_filtered.index(val)] if val in a2_filtered else np.nan for val in sorted_global_a2]
        plt.plot(sorted_global_a2, energy_aligned, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha,
                 label=f"Cohesive energy versus a2 {info_suffix}")
        plt.scatter(sorted_global_a2, energy_aligned, s=line_weight * 4, c=colors[1], alpha=line_alpha, zorder=1)
        legend_handles.append(
            mlines.Line2D([], [], color=colors[1], marker='o', markersize=6, linestyle=line_style,
                          label=f"Cohesive energy versus a2 {info_suffix}")
        )

    plt.xlabel("a2 (Å)")
    plt.ylabel("Cohesive energy (eV/atom)")
    plt.xticks(ticks=range(len(global_a2_labels)), labels=global_a2_labels)
    plt.title(f"{suptitle}")
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

def plot_cohesive_energy_a3_single(suptitle, *args_list):
    help_info = (
        "Usage: plot_cohesive_energy_a3(suptitle, args_list)\n"
        "args_list: A list containing [info_suffix, source_data, a3_boundary, color_family, line_style, line_weight, line_alpha].\n"
        "Example: plot_cohesive_energy_a3(suptitle, ['Material Info', 'source_data_path', (start, end), 'purple', 'dashed', 2.0, 0.8])\n"
    )

    if not args_list or args_list[0] in ["HELP", "Help", "help"]:
        print(help_info)
        return

    # Unpack args_list and handle missing parameters
    args_info = args_list[0] + [None] * (7 - len(args_list[0]))
    info_suffix, source_data, a3_boundary, color_family, line_style, line_weight, line_alpha = args_info

    # Apply default values using `get_or_default`
    color_family = get_or_default(color_family, "default")
    line_style   = get_or_default(line_style, "solid")
    line_weight  = get_or_default(line_weight, 1.5)
    line_alpha   = get_or_default(line_alpha, 1.0)

    # Figure settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Apply ScalarFormatter with scientific notation limits
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    # Color sampling
    colors = color_sampling(color_family)

    # Data input
    data_dict_list = read_energy_parameters(source_data)

    # Extract cohesive energy and a3 values
    cohesive_energy = [d.get("cohesive energy") for d in data_dict_list]
    a3_values = [d.get("a3") for d in data_dict_list]

    # Filter out None values
    filtered_data = [(val, e) for val, e in zip(a3_values, cohesive_energy) if val is not None and e is not None]
    if not filtered_data:
        print("No valid data found for plotting.")
        return

    # Unzip and sort data by a3 values
    a3_values, cohesive_energy = zip(*filtered_data)
    sorted_data = sorted(zip(a3_values, cohesive_energy), key=lambda x: x[0])
    a3_sorted, cohesive_energy_sorted = zip(*sorted_data)

    # Set boundaries for a3
    a3_start = min(a3_sorted) if not a3_boundary or a3_boundary[0] is None else float(a3_boundary[0])
    a3_end   = max(a3_sorted) if not a3_boundary or a3_boundary[1] is None else float(a3_boundary[1])

    # Filter data within the specified boundary
    a3_filtered, energy_filtered = zip(*[
        (val, e) for val, e in zip(a3_sorted, cohesive_energy_sorted) if a3_start <= val <= a3_end
    ])

    # Plotting
    plt.plot(a3_filtered, energy_filtered, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha,
             label=f"Cohesive energy versus a3 {info_suffix}")
    plt.scatter(a3_filtered, energy_filtered, s=line_weight * 4, c=colors[1], zorder=1, alpha=line_alpha)

    # Set labels and title
    plt.title(f"{suptitle} {info_suffix}")
    plt.xlabel("a3 (Å)")
    plt.ylabel("Cohesive energy (eV/atom)")
    plt.legend(loc="best")
    plt.tight_layout()


def plot_cohesive_energy_a3(suptitle, a3_list):
    """
    Generalized function to plot cohesive energy versus a3 for multiple datasets.

    Parameters:
    - a3_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, a3_boundary, color_family, line_style, line_weight, line_alpha].
    """
    if not is_nested_list(a3_list):
        return plot_cohesive_energy_a3_single(suptitle, a3_list)
    elif len(a3_list) == 1:
        return plot_cohesive_energy_a3_single(suptitle, *a3_list)

    for index, data in enumerate(a3_list):
        if not isinstance(data, list):
            print(f"Error: Item at index {index} in a3_list must be a list.")
            return
        if len(data) < 4:
            print(f"Warning: Item at index {index} has less than 4 elements. Missing elements will be filled with None.")
            a3_list[index] += [None] * (7 - len(data))

    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_formatter(formatter)

    legend_handles = []
    global_a3_set = set()
    a3_config_map = {}

    # Gather all unique a3 values for the global x-axis
    for data in a3_list:
        args_info = data + [None] * (7 - len(data))
        _, source_data, a3_boundary, color_family, line_style, line_weight, line_alpha = args_info
        data_dict_list = read_energy_parameters(source_data)
        a3_values = [d.get("a3") for d in data_dict_list]
        a3_start = min(a3_values) if not a3_boundary or a3_boundary[0] is None else float(a3_boundary[0])
        a3_end   = max(a3_values) if not a3_boundary or a3_boundary[1] is None else float(a3_boundary[1])
        for val in a3_values:
            if val is not None and a3_start <= val <= a3_end:
                global_a3_set.add(val)
                a3_config_map[val] = f"{val:.6g}"
    sorted_global_a3 = sorted(global_a3_set)
    global_a3_labels = [a3_config_map[val] for val in sorted_global_a3]

    # Plot each dataset aligned with the global a3 values
    for data in a3_list:
        args_info = data + [None] * (7 - len(data))
        info_suffix, source_data, a3_boundary, color_family, line_style, line_weight, line_alpha = args_info

        color_family = get_or_default(color_family, "default")
        line_style   = get_or_default(line_style, "solid")
        line_weight  = get_or_default(line_weight, 1.5)
        line_alpha   = get_or_default(line_alpha, 1.0)
        colors = color_sampling(color_family)
        data_dict_list = read_energy_parameters(source_data)
        energy = [d.get("total energy") for d in data_dict_list]
        a3_values = [d.get("a3") for d in data_dict_list]
        a3_start = min(a3_values) if not a3_boundary or a3_boundary[0] is None else float(a3_boundary[0])
        a3_end   = max(a3_values) if not a3_boundary or a3_boundary[1] is None else float(a3_boundary[1])
        filtered = [(val, e) for val, e in zip(a3_values, energy) if val is not None and a3_start <= val <= a3_end]
        if not filtered:
            continue
        a3_filtered, energy_filtered = zip(*sorted(filtered, key=lambda x: x[0]))
        energy_aligned = [energy_filtered[a3_filtered.index(val)] if val in a3_filtered else np.nan for val in sorted_global_a3]
        plt.plot(sorted_global_a3, energy_aligned, c=colors[1], ls=line_style, lw=line_weight, alpha=line_alpha,
                 label=f"Cohesive energy versus a3 {info_suffix}")
        plt.scatter(sorted_global_a3, energy_aligned, s=line_weight * 4, c=colors[1], alpha=line_alpha, zorder=1)
        legend_handles.append(
            mlines.Line2D([], [], color=colors[1], marker='o', markersize=6, linestyle=line_style,
                          label=f"Cohesive energy versus a3 {info_suffix}")
        )

    plt.xlabel("a3 (Å)")
    plt.ylabel("Cohesive energy (eV/atom)")
    plt.xticks(ticks=range(len(global_a3_labels)), labels=global_a3_labels)
    plt.title(f"{suptitle}")
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

def plot_cohesive_energy_kpoints_encut_single(suptitle, kpoints_list, encut_list):
    """
    Plot cohesive energy versus K-points and ENCUT (energy cutoff) for a single dataset.

    Parameters:
    - kpoints_list: A list containing [info_suffix, source_data, kpoints_boundary, color_family, line_style, line_weight, line_alpha].
    - encut_list: A list containing [info_suffix, source_data, encut_boundary, color_family, line_style, line_weight, line_alpha].
    """
    # Unpack arguments for K-points
    args_kpoints = kpoints_list + [None] * (7 - len(kpoints_list))
    info_suffix_kpoints, kpoints_source_data, kpoints_boundary, kpoints_color_family, kpoints_line_style, kpoints_line_weight, kpoints_line_alpha = args_kpoints

    # Apply defaults for K-points parameters
    kpoints_color_family = get_or_default(kpoints_color_family, "default")
    kpoints_line_style = get_or_default(kpoints_line_style, "solid")
    kpoints_line_weight = get_or_default(kpoints_line_weight, 1.5)
    kpoints_line_alpha = get_or_default(kpoints_line_alpha, 1.0)

    # Unpack arguments for ENCUT
    args_encut = encut_list + [None] * (7 - len(encut_list))
    info_suffix_encut, encut_source_data, encut_boundary, encut_color_family, encut_line_style, encut_line_weight, encut_line_alpha = args_encut

    # Apply defaults for ENCUT parameters
    encut_color_family = get_or_default(encut_color_family, "default")
    encut_line_style = get_or_default(encut_line_style, "solid")
    encut_line_weight = get_or_default(encut_line_weight, 1.5)
    encut_line_alpha = get_or_default(encut_line_alpha, 1.0)

    # Set up figure and parameters
    fig_setting = canvas_setting()
    fig_size_adjusted = (fig_setting[0][0], fig_setting[0][1] * 1.25)
    fig, ax_kpoints = plt.subplots(figsize=fig_size_adjusted, dpi=fig_setting[1])
    ax_encut = ax_kpoints.twiny()  # Create the top x-axis

    # Update rcParams globally for plot settings
    params = fig_setting[2]
    plt.rcParams.update(params)

    # Configure tick direction and placement on both axes
    ax_kpoints.tick_params(direction="in")
    ax_encut.tick_params(direction="in")

    # Configure scientific notation for y-axis on K-points axis
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    ax_kpoints.yaxis.set_major_formatter(formatter)

    # Color sampling for K-points and ENCUT lines
    kpoints_colors = color_sampling(kpoints_color_family)
    encut_colors = color_sampling(encut_color_family)

    # K-points data processing for cohesive energy
    kpoints_data = read_energy_parameters(kpoints_source_data)
    cohesive_energy_kpoints = [d.get("cohesive energy") for d in kpoints_data]
    total_kpoints = [d.get("total kpoints") for d in kpoints_data]
    directs_kpoints = [d.get("kpoints mesh") for d in kpoints_data]
    kpoints_labels = [f"({kp[0]}, {kp[1]}, {kp[2]})" for kp in directs_kpoints]

    # ENCUT data processing for cohesive energy
    encut_data = read_energy_parameters(encut_source_data)
    cohesive_energy_encut = [d.get("cohesive energy") for d in encut_data]
    encut_values = [d.get("energy cutoff (ENCUT)") for d in encut_data]
    encut_filtered_data = [(enc, en) for enc, en in zip(encut_values, cohesive_energy_encut) if enc is not None and en is not None]
    encut_sorted_data = sorted(encut_filtered_data, key=lambda x: x[0])
    encut_values_sorted, cohesive_energy_encut_sorted = zip(*encut_sorted_data)

    # Set K-points boundary
    kpoints_start = int(kpoints_boundary[0]) if kpoints_boundary and kpoints_boundary[0] is not None else min(total_kpoints)
    kpoints_end = int(kpoints_boundary[1]) if kpoints_boundary and kpoints_boundary[1] is not None else max(total_kpoints)

    kpoints_indices = [index for index, val in enumerate(total_kpoints) if kpoints_start <= val <= kpoints_end]
    cohesive_energy_kpoints_plot = [cohesive_energy_kpoints[index] for index in kpoints_indices]
    kpoints_labels_plot = [kpoints_labels[index] for index in kpoints_indices]

    # Set ENCUT boundary
    encut_start = float(encut_boundary[0]) if encut_boundary and encut_boundary[0] is not None else min(encut_values_sorted)
    encut_end = float(encut_boundary[1]) if encut_boundary and encut_boundary[1] is not None else max(encut_values_sorted)

    encut_indices = [index for index, val in enumerate(encut_values_sorted) if encut_start <= val <= encut_end]
    encut_values_plot = [encut_values_sorted[index] for index in encut_indices]
    cohesive_energy_encut_plot = [cohesive_energy_encut_sorted[index] for index in encut_indices]

    # Plot K-points data with fixed spacing on x-axis
    ax_kpoints.plot(range(len(cohesive_energy_kpoints_plot)), cohesive_energy_kpoints_plot,
                    c=kpoints_colors[1], ls=kpoints_line_style, lw=kpoints_line_weight, alpha=kpoints_line_alpha)
    ax_kpoints.scatter(range(len(cohesive_energy_kpoints_plot)), cohesive_energy_kpoints_plot,
                       s=kpoints_line_weight * 4, c=kpoints_colors[1], alpha=kpoints_line_alpha, zorder=1)
    ax_kpoints.set_xlabel("K-points configuration", color=kpoints_colors[0])
    ax_kpoints.set_ylabel("Cohesive energy (eV/atom)")
    ax_kpoints.set_xticks(range(len(kpoints_labels_plot)))
    ax_kpoints.set_xticklabels(kpoints_labels_plot, rotation=45, ha="right", color=kpoints_colors[0])

    # Plot ENCUT data on the top x-axis
    ax_encut.plot(encut_values_plot, cohesive_energy_encut_plot,
                  c=encut_colors[1], ls=encut_line_style, lw=encut_line_weight, alpha=encut_line_alpha)
    ax_encut.scatter(encut_values_plot, cohesive_energy_encut_plot,
                     s=encut_line_weight * 4, c=encut_colors[1], alpha=encut_line_alpha, zorder=1)
    ax_encut.set_xlabel("Energy cutoff (eV)", color=encut_colors[0])
    ax_encut.xaxis.set_label_position("top")
    ax_encut.xaxis.tick_top()
    for encut_label in ax_encut.get_xticklabels():
        encut_label.set_color(encut_colors[0])

    # Create unified legend
    kpoints_legend = mlines.Line2D([], [], color=kpoints_colors[1], marker='o', markersize=6,
                                   linestyle=kpoints_line_style, label=f"Cohesive energy versus K-points {info_suffix_kpoints}")
    encut_legend = mlines.Line2D([], [], color=encut_colors[1], marker='o', markersize=6,
                                 linestyle=encut_line_style, label=f"Cohesive energy versus energy cutoff {info_suffix_encut}")
    plt.legend(handles=[kpoints_legend, encut_legend], loc="best")

    plt.title(f"{suptitle}")
    plt.tight_layout()

def plot_cohesive_energy_kpoints_encut(suptitle, kpoints_list_source, encut_list_source):
    """
    Generalized function to plot cohesive energy versus K-points and ENCUT (energy cutoff) for multiple datasets.

    Parameters:
    - kpoints_list_source: A list of lists, where each inner list contains:
      [info_suffix, source_data, kpoints_boundary, color_family, line_style, line_weight, line_alpha].
    - encut_list_source: A list of lists, where each inner list contains:
      [info_suffix, source_data, encut_boundary, color_family, line_style, line_weight, line_alpha].
    """

    # Ensure kpoints_list and encut_list are always nested lists
    kpoints_list = [kpoints_list_source] if not is_nested_list(kpoints_list_source) else kpoints_list_source
    encut_list = [encut_list_source] if not is_nested_list(encut_list_source) else encut_list_source

    # Downgrade system to handle single data cases
    if len(kpoints_list) == 1 and len(encut_list) == 1:
        return plot_cohesive_energy_kpoints_encut_single(suptitle, kpoints_list[0], encut_list[0])

    # Set up figure and parameters
    fig_setting = canvas_setting()
    fig_size_adjusted = (fig_setting[0][0], fig_setting[0][1] * 1.25)
    fig, ax_kpoints = plt.subplots(figsize=fig_size_adjusted, dpi=fig_setting[1])
    ax_encut = ax_kpoints.twiny()  # Create the top x-axis

    # Update rcParams globally for plot settings
    params = fig_setting[2]
    plt.rcParams.update(params)

    # Configure tick direction and placement on both axes
    ax_kpoints.tick_params(direction="in")
    ax_encut.tick_params(direction="in")

    # Configure scientific notation for y-axis on K-points axis
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    ax_kpoints.yaxis.set_major_formatter(formatter)

    # Initialize legend handles for unified legend creation
    legend_handles = []

    # Collect all filtered K-points and ENCUT for global alignment
    global_kpoints = set()
    global_encut_values = set()

    # Process K-points datasets for cohesive energy
    for kpoints_data_item in kpoints_list:
        args_kpoints = kpoints_data_item + [None] * (7 - len(kpoints_data_item))
        info_suffix, source_data, kpoints_boundary, color_family, line_style, line_weight, line_alpha = args_kpoints

        # Apply defaults
        color_family = get_or_default(color_family, "default")
        line_style = get_or_default(line_style, "solid")
        line_weight = get_or_default(line_weight, 1.5)
        line_alpha = get_or_default(line_alpha, 1.0)

        # Data processing
        kpoints_data = read_energy_parameters(source_data)
        cohesive_energy_kpoints = [d.get("cohesive energy") for d in kpoints_data]
        total_kpoints = [d.get("total kpoints") for d in kpoints_data]
        directs_kpoints = [d.get("kpoints mesh") for d in kpoints_data]
        kpoints_labels = {kp: f"({coord[0]}, {coord[1]}, {coord[2]})" for kp, coord in zip(total_kpoints, directs_kpoints)}

        # Apply boundaries
        kpoints_start = kpoints_boundary[0] if kpoints_boundary and kpoints_boundary[0] is not None else min(total_kpoints)
        kpoints_end = kpoints_boundary[1] if kpoints_boundary and kpoints_boundary[1] is not None else max(total_kpoints)

        filtered_data = [
            (kp, en, kpoints_labels[kp]) for kp, en in zip(total_kpoints, cohesive_energy_kpoints)
            if kpoints_start <= kp <= kpoints_end
        ]

        # Sort and unpack
        filtered_data.sort(key=lambda x: x[0])
        filtered_kpoints, filtered_energy, filtered_labels = zip(*filtered_data)

        # Update global K-points
        global_kpoints.update(filtered_kpoints)

        # Plot K-points
        ax_kpoints.plot(range(len(filtered_kpoints)), filtered_energy, c=color_sampling(color_family)[1],
                        lw=line_weight, ls=line_style, alpha=line_alpha)
        ax_kpoints.scatter(range(len(filtered_kpoints)), filtered_energy, s=line_weight * 4,
                           c=color_sampling(color_family)[1], alpha=line_alpha, zorder=1)

        # Add legend
        legend_handles.append(mlines.Line2D([], [], color=color_sampling(color_family)[1], marker='o', markersize=6,
                                            linestyle=line_style, label=f"Cohesive energy versus K-points {info_suffix}"))

    # Process ENCUT datasets for cohesive energy
    for encut_data_item in encut_list:
        args_encut = encut_data_item + [None] * (7 - len(encut_data_item))
        info_suffix, source_data, encut_boundary, color_family, line_style, line_weight, line_alpha = args_encut

        # Apply defaults
        color_family = get_or_default(color_family, "default")
        line_style = get_or_default(line_style, "solid")
        line_weight = get_or_default(line_weight, 1.5)
        line_alpha = get_or_default(line_alpha, 1.0)

        # Data processing
        encut_data = read_energy_parameters(source_data)
        cohesive_energy_encut = [d.get("cohesive energy") for d in encut_data]
        encut_values = [d.get("energy cutoff (ENCUT)") for d in encut_data]

        encut_filtered_data = [(enc, en) for enc, en in zip(encut_values, cohesive_energy_encut) if enc is not None and en is not None]
        encut_sorted_data = sorted(encut_filtered_data, key=lambda x: x[0])
        encut_values_sorted, cohesive_energy_encut_sorted = zip(*encut_sorted_data)

        # Apply boundaries
        encut_start = encut_boundary[0] if encut_boundary and encut_boundary[0] is not None else min(encut_values_sorted)
        encut_end = encut_boundary[1] if encut_boundary and encut_boundary[1] is not None else max(encut_values_sorted)

        encut_filtered_indices = [
            index for index, val in enumerate(encut_values_sorted) if encut_start <= val <= encut_end
        ]
        encut_values_plot = [encut_values_sorted[index] for index in encut_filtered_indices]
        cohesive_energy_encut_plot = [cohesive_energy_encut_sorted[index] for index in encut_filtered_indices]
        global_encut_values.update(encut_values_plot)

        # Plot ENCUT
        ax_encut.plot(encut_values_plot, cohesive_energy_encut_plot, c=color_sampling(color_family)[1],
                      lw=line_weight, ls=line_style, alpha=line_alpha)
        ax_encut.scatter(encut_values_plot, cohesive_energy_encut_plot, s=line_weight * 4,
                         c=color_sampling(color_family)[1], alpha=line_alpha, zorder=1)

        # Add legend
        legend_handles.append(mlines.Line2D([], [], color=color_sampling(color_family)[1], marker='o', markersize=6,
                                            linestyle=line_style, label=f"Cohesive energy versus ENCUT {info_suffix}"))

    # Configure axis labels and legend
    ax_kpoints.set_xlabel("K-points configuration")
    ax_kpoints.set_ylabel("Cohesive energy (eV/atom)")
    ax_encut.set_xlabel("Energy cutoff (eV)")
    ax_encut.xaxis.set_label_position("top")
    ax_encut.xaxis.tick_top()
    ax_kpoints.legend(handles=legend_handles, loc="best")

    plt.title(f"{suptitle}")
    plt.tight_layout()

# Scheduling functions

def plot_energy_parameters(suptitle, parameters, *args):
    if isinstance(parameters, str):
        param = parameters.lower()
        args_list = args[0] if len(args) == 1 and isinstance(args[0], list) else list(args)
        if param in ["kpoint", "kpoints", "k-point", "k-points"]:
            return plot_energy_kpoints(suptitle, args_list)
        elif param in ["encut", "energy cutoff", "energy_cutoff", "energy-cutoff"]:
            return plot_energy_encut(suptitle, args_list)
        elif param in ["lattice", "lattice constant"]:
            return plot_energy_lattice(suptitle, args_list)
        elif param in ["scaling"]:
            return plot_energy_scaling(suptitle, args_list)
        elif param in ["a1"]:
            return plot_energy_a1(suptitle, args_list)
        elif param in ["a2"]:
            return plot_energy_a2(suptitle, args_list)
        elif param in ["a3"]:
            return plot_energy_a3(suptitle, args_list)
        else:
            print("Parameter type not recognized or incorrect arguments. To be continued.")
    elif isinstance(parameters, (tuple, list)):
        # If the list or tuple length is 1, treat it as a single string case
        if len(parameters) == 1:
            return plot_energy_parameters(suptitle, parameters[0], *args)
        # Handle combination of two parameters
        elif len(parameters) == 2:
            first_param, second_param = parameters
            # Standardize parameter names for easy comparison
            first_param = first_param.lower()
            second_param = second_param.lower()
            if first_param in ["kpoints", "k-point", "kpoints configuration"] and second_param in ["encut", "energy cutoff", "energy_cutoff", "energy-cutoff"]:
                return plot_energy_kpoints_encut(suptitle, *args)
            elif first_param in ["encut", "energy cutoff", "energy_cutoff", "energy-cutoff"] and second_param in ["kpoints", "k-point", "kpoints configuration"]:
                print("Reordering parameters to [\"kpoints\", \"encut\"] and plotting.")
                # Reorder parameters to call plot_energy_kpoints_encut
                return plot_energy_kpoints_encut(suptitle, args[1], args[0])
            else: print("Combination of parameters not recognized.")
        else: print("Error: Only one or two parameters are allowed for plot types.")
    else: print("Invalid input type for parameters.")

def plot_energy_parameter(*args):
    # Alias for single-parameter usage
    return plot_energy_parameters(*args)

def plot_cohesive_energy_parameters(suptitle, parameters, *args):
    if isinstance(parameters, str):
        param = parameters.lower()
        args_list = args[0] if len(args) == 1 and isinstance(args[0], list) else list(args)
        if param in ["kpoint", "kpoints", "k-point", "k-points"]:
            return plot_cohesive_energy_kpoints(suptitle, args_list)
        elif param in ["encut", "energy cutoff", "energy_cutoff", "energy-cutoff"]:
            return plot_cohesive_energy_encut(suptitle, args_list)
        elif param in ["lattice", "lattice constant"]:
            return plot_cohesive_energy_lattice(suptitle, args_list)
        elif param in ["scaling"]:
            return plot_cohesive_energy_scaling(suptitle, args_list)
        elif param in ["a1"]:
            return plot_cohesive_energy_a1(suptitle, args_list)
        elif param in ["a2"]:
            return plot_cohesive_energy_a2(suptitle, args_list)
        elif param in ["a3"]:
            return plot_cohesive_energy_a3(suptitle, args_list)
        else:
            print("Parameter type not recognized or incorrect arguments. To be continued.")
    elif isinstance(parameters, (tuple, list)):
        # If the list or tuple length is 1, treat it as a single string case
        if len(parameters) == 1:
            return plot_cohesive_energy_parameters(suptitle, parameters[0], *args)
        # Handle combination of two parameters
        elif len(parameters) == 2:
            first_param, second_param = parameters
            # Standardize parameter names for easy comparison
            first_param = first_param.lower()
            second_param = second_param.lower()
            if first_param in ["kpoints", "k-point", "kpoints configuration"] and second_param in ["encut", "energy cutoff", "energy_cutoff", "energy-cutoff"]:
                return plot_cohesive_energy_kpoints_encut(suptitle, *args)
            elif first_param in ["encut", "energy cutoff", "energy_cutoff", "energy-cutoff"] and second_param in ["kpoints", "k-point", "kpoints configuration"]:
                print("Reordering parameters to [\"kpoints\", \"encut\"] and plotting.")
                # Reorder parameters to call plot_cohesive_energy_kpoints_encut
                return plot_cohesive_energy_kpoints_encut(suptitle, args[1], args[0])
            else:
                print("Combination of parameters not recognized.")
        else:
            print("Error: Only one or two parameters are allowed for plot types.")
    else:
        print("Invalid input type for parameters.")

def plot_cohesive_energy_parameter(*args):
    # Alias for single-parameter usage
    return plot_cohesive_energy_parameters(*args)
