#### Convergence test
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0914

import xml.etree.ElementTree as ET
import os

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from vmatplot.commons import check_vasprun, identify_parameters
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

def summarize_cohesive_energy(full_cal_dir, atom_cal_dir):
    result_file = "cohesive_energy.dat"
    result_file_path = os.path.join(atom_cal_dir, result_file)

    # Extract data from both directories
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

    # Match and calculate cohesive energy
    results = []
    for full_params in full_cal_data:
        for atom_params in atom_cal_data:
            # Matching conditions now include energy cutoff and Scaling
            if (full_params['kpoints mesh'] == atom_params['kpoints mesh'] and
                full_params['total kpoints'] == atom_params['total kpoints'] and
                full_params.get("energy cutoff (ENCUT)") == atom_params.get("energy cutoff (ENCUT)") and
                full_params.get("Scaling") == atom_params.get("Scaling")):
                
                cohesive_energy = full_params['total energy'] - full_params['total atom count'] * atom_params['total energy']
                result = {
                    "total atom count": full_params['total atom count'],
                    "total energy": full_params['total energy'],
                    "atom energy": atom_params['total energy'],
                    "cohesive energy": cohesive_energy,
                    "total kpoints": full_params['total kpoints'],
                    "kpoints mesh": full_params['kpoints mesh'],
                    "energy cutoff (ENCUT)": full_params.get("energy cutoff (ENCUT)"),
                    "Scaling": full_params.get("Scaling"),  # Include Scaling in the result output
                    "lattice constant": full_params.get("lattice constant")
                }
                results.append(result)

    # Check if results are empty before proceeding to write the output
    if not results:
        print("No matching data found. Please check your directories.")
        return results

    # Sort results by total kpoints first, then by energy cutoff (ENCUT)
    results.sort(key=lambda x: (x['total kpoints'], x['energy cutoff (ENCUT)']))

    # Write results to cohesive_energy.dat file
    try:
        with open(result_file_path, "w", encoding="utf-8") as f:
            # Write headers based on keys in the first result dictionary
            headers = "\t".join(results[0].keys())
            f.write(headers + "\n")
            for result in results:
                f.write("\t".join(
                    str(result[key]) if result[key] is not None else 'None' for key in results[0].keys()
                ) + "\n")
    except IOError as e:
        print(f"Error writing to file at {result_file_path}: {e}")

    return results

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

def plot_energy_kpoints_single(*args_list):
    help_info = "Usage: plot_energy_kpoints(args_list)\n" + \
                "args_list: A list containing [info_suffix, source_data, kpoints_boundary, color_family].\n" + \
                "Example: plot_energy_kpoints(['Material Info', 'source_data_path', (start, end), 'blue'])\n"

    if not args_list or args_list[0] in ["HELP", "Help", "help"]:
        print(help_info)
        return

    # Unpack args_list
    info_suffix, source_data, kpoints_boundary, color_family = args_list[0]

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
    plt.title(f"Energy versus K-points {info_suffix}")
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
    plt.plot(range(len(total_kpoints_plot)), energy_plotting, c=colors[1], lw=1.5, label=f"Energy versus K-points ({info_suffix})")
    plt.scatter(range(len(total_kpoints_plot)), energy_plotting, s=6, c=colors[1], zorder=1)

    # Set custom tick labels for x-axis to show kpoints configurations
    plt.xticks(ticks=range(len(kpoints_labels_plot)), labels=kpoints_labels_plot, rotation=45, ha="right")

    plt.tight_layout()

def plot_energy_kpoints(kpoints_list):
    """
    Generalized function to plot energy versus K-points configuration for multiple datasets.

    Parameters:
    - kpoints_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, kpoints_boundary, color_family].
    """
    
    # Check if input is a single data set (either single list or a list with one sublist)
    if is_nested_list(kpoints_list) is False:
        return plot_energy_kpoints_single(kpoints_list)
    elif isinstance(kpoints_list[0], list) and len(kpoints_list) == 1:
        return plot_energy_kpoints_single(*kpoints_list)
    else:
        pass

    # Check if kpoints_list is a 2D list structure for multiple datasets
    if not all(isinstance(data, list) and len(data) == 4 for data in kpoints_list):
        print("Error: Each item in kpoints_list must be a list with [info_suffix, source_data, kpoints_boundary, color_family].")
        return
    
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
        info_suffix, source_data, kpoints_boundary, color_family = data
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
        info_suffix, source_data, kpoints_boundary, color_family = data
        
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
        
        # Plotting with color and unique label
        plt.plot(range(len(sorted_total_kpoints)), energy_aligned, c=colors[1], lw=1.5)
        plt.scatter(range(len(sorted_total_kpoints)), energy_aligned, s=6, c=colors[1], zorder=1)

        # Add legend entry with custom handle
        legend_handle = mlines.Line2D([], [], color=colors[1], marker='o', markersize=6, linestyle='-', 
                                      label=f"Energy versus K-points {info_suffix}")
        legend_handles.append(legend_handle)

    # Set labels and legend for multi-dataset
    plt.xlabel("K-points configurations")
    plt.ylabel("Energy (eV)")
    plt.xticks(ticks=range(len(global_kpoints_config)), labels=global_kpoints_config, rotation=45, ha="right")
    plt.title("Energy versus K-points")
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

def plot_energy_encut_single(*args_list):
    help_info = "Usage: plot_energy_encut(args_list)\n" + \
                "args_list: A list containing [info_suffix, source_data, encut_boundary, color_family].\n" + \
                "Example: plot_energy_encut(['Material Info', 'source_data_path', (start, end), 'violet'])\n"

    if not args_list or args_list[0] in ["HELP", "Help", "help"]:
        print(help_info)
        return

    # Unpack args_list
    info_suffix, source_data, encut_boundary, color_family = args_list[0]

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
    plt.title(f"Energy versus energy cutoff {info_suffix}")
    plt.xlabel("Energy cutoff (eV)")
    plt.ylabel("Energy (eV)")

    # Set boundaries for ENCUT
    if encut_boundary in [None, ""]:
        encut_start = min(encut_sorted)
        encut_end = max(encut_sorted)
    else:
        encut_start = float(encut_boundary[0]) if encut_boundary[0] else min(encut_sorted)
        encut_end = float(encut_boundary[1]) if encut_boundary[1] else max(encut_sorted)

    # Find indices for the specified boundary in sorted data
    start_index = next((i for i, val in enumerate(encut_sorted) if val >= encut_start), 0)
    end_index = next((i for i, val in enumerate(encut_sorted) if val > encut_end), len(encut_sorted)) - 1

    # Prepare ENCUT values and energy values for plotting within specified boundary
    encut_plotting = encut_sorted[start_index:end_index + 1]
    energy_plotting = energy_sorted[start_index:end_index + 1]

    # Plotting
    plt.scatter(encut_plotting, energy_plotting, s=6, c=colors[1], zorder=1, label=f"Energy versus energy cutoff ({info_suffix})")
    plt.plot(encut_plotting, energy_plotting, c=colors[1], lw=1.5)

    plt.tight_layout()

def plot_energy_encut(encut_list):
    """
    Generalized function to plot energy versus ENCUT (energy cutoff) for multiple datasets.

    Parameters:
    - encut_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, encut_boundary, color_family].
    """

    # Check if input is a single data set (either single list or a list with one sublist)
    if is_nested_list(encut_list) is False:
        return plot_energy_encut_single(encut_list)
    elif isinstance(encut_list[0], list) and len(encut_list) == 1:
        return plot_energy_encut_single(*encut_list)  # Calls single data plot function directly
    else:
        pass

    # Check if encut_list is a 2D list structure for multiple datasets
    if not all(isinstance(data, list) and len(data) == 4 for data in encut_list):
        print("Error: Each item in encut_list must be a list with [info_suffix, source_data, encut_boundary, color_family].")
        return

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
    all_encut_set = set()  # Collect unique ENCUT values across all datasets

    # Iterate over each dataset in encut_list to gather ENCUT values
    for data in encut_list:
        info_suffix, source_data, encut_boundary, color_family = data
        data_dict_list = read_energy_parameters(source_data)  # Now returns list of dicts
        encut_values = [d.get("energy cutoff (ENCUT)", d.get("ENCUT")) for d in data_dict_list]

        # Apply boundary to filter relevant ENCUT values for each dataset
        encut_start = min(encut_values) if not encut_boundary or encut_boundary[0] is None else float(encut_boundary[0])
        encut_end = max(encut_values) if not encut_boundary or encut_boundary[1] is None else float(encut_boundary[1])
        filtered_encut = [enc for enc in encut_values if encut_start <= enc <= encut_end]

        # Add filtered ENCUT values to global set
        all_encut_set.update(filtered_encut)

    # Sort the global ENCUT values for unified x-axis
    all_encut_sorted = sorted(all_encut_set)

    # Iterate over each dataset to align with global ENCUT and plot
    for data in encut_list:
        info_suffix, source_data, encut_boundary, color_family = data

        # Color for the current dataset
        colors = color_sampling(color_family)

        # Data input
        data_dict_list = read_energy_parameters(source_data)
        energy = [d.get("total energy", d.get("Total Energy")) for d in data_dict_list]
        encut_values = [d.get("energy cutoff (ENCUT)", d.get("ENCUT")) for d in data_dict_list]

        # Apply boundary values for the current dataset
        encut_start = min(encut_values) if not encut_boundary or encut_boundary[0] is None else float(encut_boundary[0])
        encut_end = max(encut_values) if not encut_boundary or encut_boundary[1] is None else float(encut_boundary[1])
        
        # Filter data within the specified boundary for the current dataset
        encut_filtered = [enc for enc in encut_values if encut_start <= enc <= encut_end]
        energy_filtered = [energy[idx] for idx, enc in enumerate(encut_values) if encut_start <= enc <= encut_end]
        
        # Create a mapping for the filtered data to align with the global ENCUT
        energy_aligned = [energy_filtered[encut_filtered.index(enc)] if enc in encut_filtered else np.nan for enc in all_encut_sorted]

        # Plotting with color and unique label
        plt.plot(all_encut_sorted, energy_aligned, c=colors[1], lw=1.5)
        plt.scatter(all_encut_sorted, energy_aligned, s=6, c=colors[1], zorder=1)

        # Add legend entry with custom handle
        legend_handle = mlines.Line2D([], [], color=colors[1], marker='o', markersize=6, linestyle='-', 
                                      label=f"Energy versus energy cutoff {info_suffix}")
        legend_handles.append(legend_handle)

    # Set labels and legend for multi-dataset
    plt.xlabel("Energy cutoff (eV)")
    plt.ylabel("Energy (eV)")
    plt.title("Energy versus energy cutoff")
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

def plot_energy_lattice_single(*args_list):
    help_info = (
        "Usage: plot_energy_lattice_single(args_list)\n"
        "args_list: A list containing [info_suffix, source_data, lattice_boundary, color_family, num_samples].\n"
        "Example: plot_energy_lattice_single(['Material Info', 'source_data_path', (start, end), 'green', 11])\n"
    )

    # Check if the user requested help
    if not args_list or args_list[0] in ["HELP", "Help", "help"]:
        print(help_info)
        return

    # Unpack args_list
    info_suffix, source_data, lattice_boundary, color_family, num_samples = args_list[0]

    # Figure settings
    fig_setting = canvas_setting()
    fig, ax_kpoints = plt.subplots(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Color selection
    colors = color_sampling(color_family)

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
    if lattice_boundary in [None, ""]:
        lattice_start = min(lattice_sorted)
        lattice_end = max(lattice_sorted)
    else:
        lattice_start = float(lattice_boundary[0]) if lattice_boundary[0] else min(lattice_sorted)
        lattice_end = float(lattice_boundary[1]) if lattice_boundary[1] else max(lattice_sorted)

    # Filter data within the specified boundary
    lattice_filtered = []
    energy_filtered = []
    for lattice, energy in zip(lattice_sorted, energy_sorted):
        if lattice_start <= lattice <= lattice_end:
            lattice_filtered.append(lattice)
            energy_filtered.append(energy)

    # Estimate EOS parameters and fitted energy values using all data
    eos_params, resampled_lattice, fitted_energy = fit_birch_murnaghan(lattice_filtered, energy_filtered, sample_count=100)

    # Plot the fitted EOS curve
    ax_kpoints.plot(resampled_lattice, fitted_energy, color=colors[1], lw=1.5, label=f"Fitted EOS Curve {info_suffix}")

    # Select scatter sample points based on approximately equal intervals in x-axis values
    if num_samples is None or num_samples >= len(lattice_filtered):
        scatter_lattice = lattice_filtered
        scatter_energy = energy_filtered
    else:
        # Define equally spaced x-axis values within the lattice boundary
        x_samples = np.linspace(lattice_start, lattice_end, num_samples)
        scatter_lattice = []
        scatter_energy = []
        for x in x_samples:
            # Find the data point closest to the x sample
            idx = (np.abs(np.array(lattice_filtered) - x)).argmin()
            scatter_lattice.append(lattice_filtered[idx])
            scatter_energy.append(energy_filtered[idx])

    # Remove duplicate points (if any)
    unique_points = set()
    scatter_lattice_unique = []
    scatter_energy_unique = []
    for x, y in zip(scatter_lattice, scatter_energy):
        if x not in unique_points:
            unique_points.add(x)
            scatter_lattice_unique.append(x)
            scatter_energy_unique.append(y)

    # Scatter sample data points
    ax_kpoints.scatter(scatter_lattice_unique, scatter_energy_unique, s=48, fc="#FFFFFF", ec=colors[1], label=f"Sampled data {info_suffix}", zorder=2)

    # Find and mark the minimum energy point from the filtered data
    min_energy_idx = np.argmin(energy_filtered)
    ax_kpoints.scatter(lattice_filtered[min_energy_idx], energy_filtered[min_energy_idx], s=48, fc=colors[2], ec=colors[2], label=f"Minimum Energy {info_suffix}", zorder=3)

    # Set labels, title, and legend
    ax_kpoints.set_xlabel(r"Lattice constant (Å)")
    ax_kpoints.set_ylabel(r"Energy (eV)")
    ax_kpoints.set_title(f"Energy versus lattice constant {info_suffix}")
    ax_kpoints.legend()
    plt.tight_layout()

def plot_energy_lattice(lattice_list):
    """
    Generalized function to plot energy versus lattice constant for multiple datasets.

    Parameters:
    - lattice_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, lattice_boundary, color_family, num_samples].
    """
    # Single dataset case
    if not isinstance(lattice_list[0], list):
        return plot_energy_lattice_single(lattice_list)
    elif len(lattice_list) == 1:
        return plot_energy_lattice_single(*lattice_list)

    # Multi-dataset case: Verify each inner list has the correct format
    if not all(isinstance(data, list) and len(data) == 5 for data in lattice_list):
        print("Error: Each item in lattice_list must be a list with [info_suffix, source_data, lattice_boundary, color_family, num_samples].")
        return

    fig_setting = canvas_setting()
    fig, ax = plt.subplots(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Scientific notation for y-axis
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)

    legend_handles = []

    for data in lattice_list:
        info_suffix, source_data, lattice_boundary, color_family, num_samples = data
        colors = color_sampling(color_family)

        # Data input
        data_dict_list = read_energy_parameters(source_data)
        energy_values = [d.get("total energy", d.get("Total Energy")) for d in data_dict_list]
        lattice_constants = [d.get("lattice constant", d.get("Lattice Constant")) for d in data_dict_list]

        # Filter out None values
        filtered_data = [(lattice, energy) for lattice, energy in zip(lattice_constants, energy_values)
                         if lattice is not None and energy is not None]

        if not filtered_data:
            print(f"No valid data found for plotting for dataset {info_suffix}.")
            continue

        # Unzip filtered data and sort
        lattice_constants, energy_values = zip(*filtered_data)
        sorted_data = sorted(zip(lattice_constants, energy_values), key=lambda x: x[0])
        lattice_sorted, energy_sorted = zip(*sorted_data)

        # Define boundaries for lattice constants
        if lattice_boundary in [None, ""]:
            lattice_start = min(lattice_sorted)
            lattice_end = max(lattice_sorted)
        else:
            lattice_start = float(lattice_boundary[0]) if lattice_boundary[0] else min(lattice_sorted)
            lattice_end = float(lattice_boundary[1]) if lattice_boundary[1] else max(lattice_sorted)

        # Filter data within the specified boundary
        lattice_filtered = []
        energy_filtered = []
        for lattice, energy in zip(lattice_sorted, energy_sorted):
            if lattice_start <= lattice <= lattice_end:
                lattice_filtered.append(lattice)
                energy_filtered.append(energy)

        if not lattice_filtered:
            print(f"No data within the specified lattice boundary for dataset {info_suffix}.")
            continue

        # Estimate EOS parameters and fitted energy values using all data
        eos_params, resampled_lattice, fitted_energy = fit_birch_murnaghan(lattice_filtered, energy_filtered, sample_count=100)

        # Plot the fitted EOS curve
        ax.plot(resampled_lattice, fitted_energy, color=colors[1], lw=1.5, label=f"Fitted EOS Curve {info_suffix}")

        # Select scatter sample points based on approximately equal intervals in x-axis values
        if num_samples is None or num_samples >= len(lattice_filtered):
            scatter_lattice = lattice_filtered
            scatter_energy = energy_filtered
        else:
            # Define equally spaced x-axis values within the lattice boundary
            x_samples = np.linspace(lattice_start, lattice_end, num_samples)
            scatter_lattice = []
            scatter_energy = []
            for x in x_samples:
                # Find the data point closest to the x sample
                idx = (np.abs(np.array(lattice_filtered) - x)).argmin()
                scatter_lattice.append(lattice_filtered[idx])
                scatter_energy.append(energy_filtered[idx])

        # Remove duplicate points (if any)
        unique_points = set()
        scatter_lattice_unique = []
        scatter_energy_unique = []
        for x, y in zip(scatter_lattice, scatter_energy):
            if x not in unique_points:
                unique_points.add(x)
                scatter_lattice_unique.append(x)
                scatter_energy_unique.append(y)

        # Scatter sample data points
        ax.scatter(scatter_lattice_unique, scatter_energy_unique, s=48, fc="#FFFFFF", ec=colors[1], label=f"Sampled data {info_suffix}", zorder=2)

        # Find and mark the minimum energy point from the filtered data
        min_energy_idx = np.argmin(energy_filtered)
        ax.scatter(lattice_filtered[min_energy_idx], energy_filtered[min_energy_idx], s=48, fc=colors[2], ec=colors[2], label=f"Minimum Energy {info_suffix}", zorder=3)

        # Add legend entry
        legend_handle = mlines.Line2D([], [], color=colors[1], marker='o', markersize=6, linestyle='-',
                                      label=f"Dataset {info_suffix}")
        legend_handles.append(legend_handle)

    # Set labels and legend for multi-dataset
    ax.set_xlabel("Lattice constant (Å)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Energy versus lattice constant")
    ax.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

def plot_energy_kpoints_encut_single(kpoints_list, encut_list):
    # Unpack lists to retrieve the new info_suffix (materials information)
    info_suffix_kpoints, kpoints_source_data, kpoints_boundary, kpoints_color_family = kpoints_list
    info_suffix_encut, encut_source_data, encut_boundary, encut_color_family = encut_list

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
    kpoints_labels = [f"({kp[0]}, {kp[1]}, {kp[2]})" for kp in directs_kpoints]

    # ENCUT data processing
    encut_data = read_energy_parameters(encut_source_data)
    encut_energy = [d.get("total energy", d.get("Total Energy")) for d in encut_data]
    encut_values = [d.get("energy cutoff (ENCUT)", d.get("ENCUT")) for d in encut_data]
    encut_filtered_data = [(enc, en) for enc, en in zip(encut_values, encut_energy) if enc is not None and en is not None]
    encut_sorted_data = sorted(encut_filtered_data, key=lambda x: x[0])
    encut_values_sorted, encut_energy_sorted = zip(*encut_sorted_data)

    # Set K-points boundary
    kpoints_start = int(kpoints_boundary[0]) if kpoints_boundary[0] is not None else min(total_kpoints)
    kpoints_end = int(kpoints_boundary[1]) if kpoints_boundary[1] is not None else max(total_kpoints)

    kpoints_indices = [
        index for index, val in enumerate(total_kpoints) if kpoints_start <= val <= kpoints_end
    ]
    kpoints_energy_plot = [kpoints_energy[index] for index in kpoints_indices]
    kpoints_labels_plot = [kpoints_labels[index] for index in kpoints_indices]

    # Set ENCUT boundary
    encut_start = float(encut_boundary[0]) if encut_boundary[0] is not None else min(encut_values_sorted)
    encut_end = float(encut_boundary[1]) if encut_boundary[1] is not None else max(encut_values_sorted)

    encut_indices = [
        index for index, val in enumerate(encut_values_sorted) if encut_start <= val <= encut_end
    ]
    encut_values_plot = [encut_values_sorted[index] for index in encut_indices]
    encut_energy_plot = [encut_energy_sorted[index] for index in encut_indices]

    # Plot K-points data with fixed spacing on x-axis
    ax_kpoints.plot(range(len(kpoints_energy_plot)), kpoints_energy_plot, c=kpoints_colors[1], lw=1.5)
    ax_kpoints.scatter(range(len(kpoints_energy_plot)), kpoints_energy_plot, s=6, c=kpoints_colors[1], zorder=1)
    ax_kpoints.set_xlabel("K-points configuration", color=kpoints_colors[0])
    ax_kpoints.set_ylabel("Energy (eV)")
    ax_kpoints.set_xticks(range(len(kpoints_labels_plot)))
    ax_kpoints.set_xticklabels(kpoints_labels_plot, rotation=45, ha="right", color=kpoints_colors[0])

    # Plot ENCUT data on the top x-axis
    ax_encut.plot(encut_values_plot, encut_energy_plot, c=encut_colors[1], lw=1.5)
    ax_encut.scatter(encut_values_plot, encut_energy_plot, s=6, c=encut_colors[1], zorder=1)
    ax_encut.set_xlabel("Energy cutoff (eV)", color=encut_colors[0])
    ax_encut.xaxis.set_label_position("top")
    ax_encut.xaxis.tick_top()
    for encut_label in ax_encut.get_xticklabels():
        encut_label.set_color(encut_colors[0])

    # Create unified legend
    kpoints_legend = mlines.Line2D([], [], color=kpoints_colors[1], marker='o', markersize=6, linestyle='-', 
                                   label=f"energy versus K-points {info_suffix_kpoints}")
    encut_legend = mlines.Line2D([], [], color=encut_colors[1], marker='o', markersize=6, linestyle='-', 
                                 label=f"energy versus energy cutoff {info_suffix_encut}")
    plt.legend(handles=[kpoints_legend, encut_legend], loc="best")

    plt.title("Energy versus K-points and energy cutoff")
    plt.tight_layout()

def plot_energy_kpoints_encut(kpoints_list_source, encut_list_source):
    """
    Generalized function to plot energy versus K-points and ENCUT (energy cutoff) for multiple datasets.

    Parameters:
    - kpoints_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, kpoints_boundary, color_family] for k-points data.
    - encut_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, encut_boundary, color_family] for ENCUT data.
    """

    # Ensure kpoints_list and encut_list are always nested lists
    kpoints_list = [kpoints_list_source] if not is_nested_list(kpoints_list_source) else kpoints_list_source
    encut_list = [encut_list_source] if not is_nested_list(encut_list_source) else encut_list_source

    # Downgrade system to handle single data cases
    kpoints_list_downgrad, encut_list_downgrad = [], []
    if is_nested_list(kpoints_list_source) == False:
        kpoints_list_downgrad = kpoints_list_source
    elif len(kpoints_list_source) == 1:
        kpoints_list_downgrad = kpoints_list_source[0]
    
    if is_nested_list(encut_list_source) == False:
        encut_list_downgrad = encut_list_source
    elif len(encut_list_source) == 1:
        encut_list_downgrad = encut_list_source[0]
    
    if encut_list_downgrad != [] and kpoints_list_downgrad != []:
        return plot_energy_kpoints_encut_single(kpoints_list_downgrad, encut_list_downgrad)
    else:
        pass

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

    # Collect all filtered K-points for global alignment
    global_kpoints = set()

    # Process K-points datasets
    all_kpoints_labels = []
    for kpoints_data_item in kpoints_list:
        info_suffix, kpoints_source_data, kpoints_boundary, kpoints_color_family = kpoints_data_item
        kpoints_colors = color_sampling(kpoints_color_family)

        # K-points data processing with dynamic column names
        kpoints_data = read_energy_parameters(kpoints_source_data)
        kpoints_energy = [d.get("total energy", d.get("Total Energy")) for d in kpoints_data]
        total_kpoints = [d.get("total kpoints", d.get("Total Kpoints")) for d in kpoints_data]
        directs_kpoints = [d.get("kpoints mesh", d.get("Kpoints(X Y Z)")) for d in kpoints_data]
        kpoints_labels = {kp: f"({coord[0]}, {coord[1]}, {coord[2]})" for kp, coord in zip(total_kpoints, directs_kpoints)}

        # Apply boundary to filter K-points and energy values
        kpoints_start = kpoints_boundary[0] if kpoints_boundary[0] is not None else min(total_kpoints)
        kpoints_end = kpoints_boundary[1] if kpoints_boundary[1] is not None else max(total_kpoints)
        filtered_data = [
            (kp, en, kpoints_labels[kp]) for kp, en in zip(total_kpoints, kpoints_energy)
            if kpoints_start <= kp <= kpoints_end
        ]

        # Sort filtered data by total_kpoints and unpack
        filtered_data.sort(key=lambda x: x[0])
        filtered_kpoints, filtered_energy, filtered_labels = zip(*filtered_data)

        # Update global_kpoints for consistent alignment across datasets
        global_kpoints.update(filtered_kpoints)

        # Plot K-points data
        ax_kpoints.plot(range(len(filtered_kpoints)), filtered_energy, c=kpoints_colors[1], lw=1.5)
        ax_kpoints.scatter(range(len(filtered_kpoints)), filtered_energy, s=6, c=kpoints_colors[1], zorder=1)

        # Add to legend handles
        legend_handles.append(mlines.Line2D([], [], color=kpoints_colors[1], marker='o', markersize=6, linestyle='-', label=f"Energy versus K-points {info_suffix}"))
        all_kpoints_labels.extend(filtered_labels)

    # Set x-axis labels for K-points, ensuring unique and ordered labels
    all_kpoints_labels = sorted(set(all_kpoints_labels), key=all_kpoints_labels.index)
    ax_kpoints.set_xticks(range(len(all_kpoints_labels)))
    ax_kpoints.set_xticklabels(all_kpoints_labels, rotation=45, ha="right")
    ax_kpoints.set_xlabel("K-points configuration")
    ax_kpoints.set_ylabel("Energy (eV)")

    # Process ENCUT datasets
    all_encut_values = []
    for encut_data_item in encut_list:
        info_suffix, encut_source_data, encut_boundary, encut_color_family = encut_data_item
        encut_colors = color_sampling(encut_color_family)

        # ENCUT data processing with dynamic column names
        encut_data = read_energy_parameters(encut_source_data)
        encut_energy = [d.get("total energy", d.get("Total Energy")) for d in encut_data]
        encut_values = [d.get("energy cutoff (ENCUT)", d.get("ENCUT")) for d in encut_data]
        encut_filtered_data = [(enc, en) for enc, en in zip(encut_values, encut_energy) if enc is not None and en is not None]
        encut_sorted_data = sorted(encut_filtered_data, key=lambda x: x[0])
        encut_values_sorted, encut_energy_sorted = zip(*encut_sorted_data)

        # Set ENCUT boundary
        encut_start = float(encut_boundary[0]) if encut_boundary[0] is not None else min(encut_values_sorted)
        encut_end = float(encut_boundary[1]) if encut_boundary[1] is not None else max(encut_values_sorted)

        # Filter sorted data within boundary
        encut_indices = [
            index for index, val in enumerate(encut_values_sorted) if encut_start <= val <= encut_end
        ]
        encut_values_plot = [encut_values_sorted[index] for index in encut_indices]
        encut_energy_plot = [encut_energy_sorted[index] for index in encut_indices]
        all_encut_values.extend(encut_values_plot)

        # Plot ENCUT data on the top x-axis
        ax_encut.plot(encut_values_plot, encut_energy_plot, c=encut_colors[1], lw=1.5)
        ax_encut.scatter(encut_values_plot, encut_energy_plot, s=6, c=encut_colors[1], zorder=1)

        # Add to legend handles
        legend_handles.append(mlines.Line2D([], [], color=encut_colors[1], marker='o', markersize=6, linestyle='-', label=f"Energy versus energy cutoff {info_suffix}"))

    # Set top x-axis labels for ENCUT, using MaxNLocator for fewer ticks
    ax_encut.set_xlabel("Energy cutoff (eV)")
    ax_encut.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
    ax_encut.xaxis.set_label_position("top")
    ax_encut.xaxis.tick_top()

    # Create unified legend
    plt.legend(handles=legend_handles, loc="best")
    plt.title("Energy versus K-points and energy cutoff")
    plt.tight_layout()

## Cohesive energy versus parameters

def plot_cohesive_energy_kpoints_single(*args_list):
    help_info = "Usage: plot_cohesive_energy_kpoints(args_list)\n" + \
                "args_list: A list containing [info_suffix, source_data, kpoints_boundary, color_family].\n" + \
                "Example: plot_cohesive_energy_kpoints(['Material Info', 'source_data_path', (start, end), 'blue'])\n"

    if not args_list or args_list[0] in ["HELP", "Help", "help"]:
        print(help_info)
        return

    # Unpack args_list
    info_suffix, source_data, kpoints_boundary, color_family = args_list[0]

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
    cohesive_energy = [d.get("cohesive energy") for d in data_dict_list]
    sep_kpoints = [d.get("kpoints mesh") for d in data_dict_list]

    # Sort data based on total_kpoints to maintain order
    sorted_data = sorted(zip(total_kpoints, cohesive_energy, sep_kpoints), key=lambda x: x[0])
    total_kpoints_sorted, cohesive_energy_sorted, sep_kpoints_sorted = zip(*sorted_data)

    # Set title with info_suffix
    plt.title(f"Cohesive energy versus K-points {info_suffix}")
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
    plt.plot(range(len(total_kpoints_plot)), cohesive_energy_plotting, c=colors[1], lw=1.5, label=f"Cohesive energy versus K-points ({info_suffix})")
    plt.scatter(range(len(total_kpoints_plot)), cohesive_energy_plotting, s=6, c=colors[1], zorder=1)

    # Set custom tick labels for x-axis to show kpoints configurations
    plt.xticks(ticks=range(len(kpoints_labels_plot)), labels=kpoints_labels_plot, rotation=45, ha="right")

    plt.tight_layout()

def plot_cohesive_energy_kpoints(kpoints_list):
    """
    Generalized function to plot cohesive energy versus K-points configuration for multiple datasets.

    Parameters:
    - kpoints_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, kpoints_boundary, color_family].
    """
    
    # Check if input is a single data set (either single list or a list with one sublist)
    if is_nested_list(kpoints_list) is False:
        return plot_cohesive_energy_kpoints_single(kpoints_list)
    elif isinstance(kpoints_list[0], list) and len(kpoints_list) == 1:
        return plot_cohesive_energy_kpoints_single(*kpoints_list)
    else:
        pass

    # Check if kpoints_list is a 2D list structure for multiple datasets
    if not all(isinstance(data, list) and len(data) == 4 for data in kpoints_list):
        print("Error: Each item in kpoints_list must be a list with [info_suffix, source_data, kpoints_boundary, color_family].")
        return
    
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
        info_suffix, source_data, kpoints_boundary, color_family = data
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
        info_suffix, source_data, kpoints_boundary, color_family = data
        
        # Color for the current dataset
        colors = color_sampling(color_family)

        # Data input
        data_dict_list = read_energy_parameters(source_data)  # Now a list of dictionaries
        cohesive_energy = [d.get("cohesive energy") for d in data_dict_list]
        total_kpoints = [d.get("total kpoints") for d in data_dict_list]

        # Apply boundary values for the current dataset
        kpoints_start = min(total_kpoints) if not kpoints_boundary or kpoints_boundary[0] is None else int(kpoints_boundary[0])
        kpoints_end = max(total_kpoints) if not kpoints_boundary or kpoints_boundary[1] is None else int(kpoints_boundary[1])
        
        # Filter data within the specified boundary for the current dataset
        filtered_kpoints = [total_kp for total_kp in total_kpoints if kpoints_start <= total_kp <= kpoints_end]
        filtered_cohesive_energy = [cohesive_energy[idx] for idx, total_kp in enumerate(total_kpoints) if kpoints_start <= total_kp <= kpoints_end]
        
        # Align cohesive energies with the global sorted total_kpoints
        cohesive_energy_aligned = [filtered_cohesive_energy[filtered_kpoints.index(total_kp)] if total_kp in filtered_kpoints else np.nan for total_kp in sorted_total_kpoints]
        
        # Plotting with color and unique label
        plt.plot(range(len(sorted_total_kpoints)), cohesive_energy_aligned, c=colors[1], lw=1.5)
        plt.scatter(range(len(sorted_total_kpoints)), cohesive_energy_aligned, s=6, c=colors[1], zorder=1)

        # Add legend entry with custom handle
        legend_handle = mlines.Line2D([], [], color=colors[1], marker='o', markersize=6, linestyle='-', 
                                      label=f"Cohesive energy versus K-points {info_suffix}")
        legend_handles.append(legend_handle)

    # Set labels and legend for multi-dataset
    plt.xlabel("K-points configurations")
    plt.ylabel("Cohesive energy (eV/atom)")
    plt.xticks(ticks=range(len(global_kpoints_config)), labels=global_kpoints_config, rotation=45, ha="right")
    plt.title("Cohesive energy versus K-points")
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

def plot_cohesive_energy_encut_single(*args_list):
    help_info = "Usage: plot_cohesive_energy_encut(args_list)\n" + \
                "args_list: A list containing [info_suffix, source_data, encut_boundary, color_family].\n" + \
                "Example: plot_cohesive_energy_encut(['Material Info', 'source_data_path', (start, end), 'violet'])\n"

    if not args_list or args_list[0] in ["HELP", "Help", "help"]:
        print(help_info)
        return

    # Unpack args_list
    info_suffix, source_data, encut_boundary, color_family = args_list[0]

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
    data_dict_list = read_energy_parameters(source_data)

    # Extract cohesive energy and ENCUT values
    cohesive_energy = [d.get("cohesive energy") for d in data_dict_list]
    encut_values = [d.get("energy cutoff (ENCUT)") for d in data_dict_list]

    # Filter out None values from cohesive energy and ENCUT values
    filtered_data = [(enc, e) for enc, e in zip(encut_values, cohesive_energy) if enc is not None and e is not None]
    if not filtered_data:
        print("No valid data found for plotting.")
        return

    # Unzip filtered data for ENCUT and cohesive energy
    encut_values, cohesive_energy = zip(*filtered_data)

    # Sort data based on ENCUT values
    sorted_data = sorted(zip(encut_values, cohesive_energy), key=lambda x: x[0])
    encut_sorted, cohesive_energy_sorted = zip(*sorted_data)

    # Set title with info_suffix
    plt.title(f"Cohesive energy versus energy cutoff {info_suffix}")
    plt.xlabel("Energy cutoff (eV)")
    plt.ylabel("Cohesive energy (eV/atom)")

    # Set boundaries for ENCUT
    encut_start = min(encut_sorted) if encut_boundary is None or encut_boundary[0] is None else float(encut_boundary[0])
    encut_end = max(encut_sorted) if encut_boundary is None or encut_boundary[1] is None else float(encut_boundary[1])

    # Filter data within the specified boundary
    start_index = next((i for i, val in enumerate(encut_sorted) if val >= encut_start), 0)
    end_index = next((i for i, val in enumerate(encut_sorted) if val > encut_end), len(encut_sorted)) - 1

    # Prepare ENCUT values and cohesive energy values for plotting within specified boundary
    encut_plotting = encut_sorted[start_index:end_index + 1]
    cohesive_energy_plotting = cohesive_energy_sorted[start_index:end_index + 1]

    # Plotting
    plt.scatter(encut_plotting, cohesive_energy_plotting, s=6, c=colors[1], zorder=1, label=f"Cohesive energy versus energy cutoff ({info_suffix})")
    plt.plot(encut_plotting, cohesive_energy_plotting, c=colors[1], lw=1.5)

    plt.tight_layout()

def plot_cohesive_energy_encut(encut_list):
    """
    Generalized function to plot cohesive energy versus ENCUT (energy cutoff) for multiple datasets.

    Parameters:
    - encut_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, encut_boundary, color_family].
    """

    # Check if input is a single data set (either single list or a list with one sublist)
    if is_nested_list(encut_list) is False:
        return plot_cohesive_energy_encut_single(encut_list)
    elif isinstance(encut_list[0], list) and len(encut_list) == 1:
        return plot_cohesive_energy_encut_single(*encut_list)
    else:
        pass

    # Check if encut_list is a 2D list structure for multiple datasets
    if not all(isinstance(data, list) and len(data) == 4 for data in encut_list):
        print("Error: Each item in encut_list must be a list with [info_suffix, source_data, encut_boundary, color_family].")
        return

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
    all_encut_set = set()  # Collect unique ENCUT values across all datasets

    # Iterate over each dataset in encut_list to gather ENCUT values
    for data in encut_list:
        info_suffix, source_data, encut_boundary, color_family = data
        data_dict_list = read_energy_parameters(source_data)
        encut_values = [d.get("energy cutoff (ENCUT)") for d in data_dict_list]

        # Apply boundary to filter relevant ENCUT values for each dataset
        encut_start = min(encut_values) if encut_boundary is None or encut_boundary[0] is None else float(encut_boundary[0])
        encut_end = max(encut_values) if encut_boundary is None or encut_boundary[1] is None else float(encut_boundary[1])
        filtered_encut = [enc for enc in encut_values if encut_start <= enc <= encut_end]

        # Add filtered ENCUT values to global set
        all_encut_set.update(filtered_encut)

    # Sort the global ENCUT values for unified x-axis
    all_encut_sorted = sorted(all_encut_set)

    # Iterate over each dataset to align with global ENCUT and plot
    for data in encut_list:
        info_suffix, source_data, encut_boundary, color_family = data

        # Color for the current dataset
        colors = color_sampling(color_family)

        # Data input
        data_dict_list = read_energy_parameters(source_data)
        cohesive_energy = [d.get("cohesive energy") for d in data_dict_list]
        encut_values = [d.get("energy cutoff (ENCUT)") for d in data_dict_list]

        # Apply boundary values for the current dataset
        encut_start = min(encut_values) if encut_boundary is None or encut_boundary[0] is None else float(encut_boundary[0])
        encut_end = max(encut_values) if encut_boundary is None or encut_boundary[1] is None else float(encut_boundary[1])

        # Filter data within the specified boundary for the current dataset
        encut_filtered = [enc for enc in encut_values if encut_start <= enc <= encut_end]
        cohesive_energy_filtered = [cohesive_energy[idx] for idx, enc in enumerate(encut_values) if encut_start <= enc <= encut_end]

        # Align cohesive energies with the global sorted ENCUT values
        cohesive_energy_aligned = [cohesive_energy_filtered[encut_filtered.index(enc)] if enc in encut_filtered else np.nan for enc in all_encut_sorted]

        # Plotting with color and unique label
        plt.plot(all_encut_sorted, cohesive_energy_aligned, c=colors[1], lw=1.5)
        plt.scatter(all_encut_sorted, cohesive_energy_aligned, s=6, c=colors[1], zorder=1)

        # Add legend entry with custom handle
        legend_handle = mlines.Line2D([], [], color=colors[1], marker='o', markersize=6, linestyle='-', 
                                      label=f"Cohesive energy versus energy cutoff {info_suffix}")
        legend_handles.append(legend_handle)

    # Set labels and legend for multi-dataset
    plt.xlabel("Energy cutoff (eV)")
    plt.ylabel("Cohesive energy (eV/atom)")
    plt.title("Cohesive energy versus energy cutoff")
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

def plot_cohesive_energy_lattice_single(*args_list):
    help_info = (
        "Usage: plot_cohesive_energy_lattice_single(args_list)\n"
        "args_list: A list containing [info_suffix, source_data, lattice_boundary, color_family, num_samples].\n"
        "Example: plot_cohesive_energy_lattice_single(['Material Info', 'source_data_path', (start, end), 'green', 11])\n"
    )

    # Check if the user requested help
    if not args_list or args_list[0] in ["HELP", "Help", "help"]:
        print(help_info)
        return

    # Unpack args_list
    info_suffix, source_data, lattice_boundary, color_family, num_samples = args_list[0]

    # Figure settings
    fig_setting = canvas_setting()
    fig, ax = plt.subplots(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Color selection
    colors = color_sampling(color_family)

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
    lattice_sorted, cohesive_energy_sorted = zip(*sorted_data)

    # Define boundaries for lattice constants
    if lattice_boundary in [None, ""]:
        lattice_start = min(lattice_sorted)
        lattice_end = max(lattice_sorted)
    else:
        lattice_start = float(lattice_boundary[0]) if lattice_boundary[0] else min(lattice_sorted)
        lattice_end = float(lattice_boundary[1]) if lattice_boundary[1] else max(lattice_sorted)

    # Filter data within the specified boundary
    lattice_filtered = []
    cohesive_energy_filtered = []
    for lattice, energy in zip(lattice_sorted, cohesive_energy_sorted):
        if lattice_start <= lattice <= lattice_end:
            lattice_filtered.append(lattice)
            cohesive_energy_filtered.append(energy)

    if not lattice_filtered:
        print("No data within the specified lattice boundary for dataset.")
        return

    # Estimate EOS parameters and fitted energy values using all data
    eos_params, resampled_lattice, fitted_energy = fit_birch_murnaghan(lattice_filtered, cohesive_energy_filtered, sample_count=100)

    # Plot the fitted EOS curve
    ax.plot(resampled_lattice, fitted_energy, color=colors[1], lw=1.5, label=f"Fitted EOS Curve {info_suffix}")

    # Select scatter sample points based on approximately equal intervals in x-axis values
    if num_samples is None or num_samples >= len(lattice_filtered):
        scatter_lattice = lattice_filtered
        scatter_energy = cohesive_energy_filtered
    else:
        # Define equally spaced x-axis values within the lattice boundary
        x_samples = np.linspace(lattice_start, lattice_end, num_samples)
        scatter_lattice = []
        scatter_energy = []
        for x in x_samples:
            # Find the data point closest to the x sample
            idx = (np.abs(np.array(lattice_filtered) - x)).argmin()
            scatter_lattice.append(lattice_filtered[idx])
            scatter_energy.append(cohesive_energy_filtered[idx])

    # Remove duplicate points (if any)
    unique_points = set()
    scatter_lattice_unique = []
    scatter_energy_unique = []
    for x, y in zip(scatter_lattice, scatter_energy):
        if x not in unique_points:
            unique_points.add(x)
            scatter_lattice_unique.append(x)
            scatter_energy_unique.append(y)

    # Scatter sample data points
    ax.scatter(scatter_lattice_unique, scatter_energy_unique, s=48, fc="#FFFFFF", ec=colors[1],
               label=f"Sampled data {info_suffix}", zorder=2)

    # Find and mark the minimum energy point from the filtered data
    min_energy_idx = np.argmin(cohesive_energy_filtered)
    ax.scatter(lattice_filtered[min_energy_idx], cohesive_energy_filtered[min_energy_idx], s=48,
               fc=colors[2], ec=colors[2], label=f"Minimum Energy {info_suffix}", zorder=3)

    # Set labels, title, and legend
    ax.set_xlabel(r"Lattice constant (Å)")
    ax.set_ylabel(r"Cohesive energy (eV/atom)")
    ax.set_title(f"Cohesive energy versus lattice constant {info_suffix}")
    ax.legend()
    plt.tight_layout()

def plot_cohesive_energy_lattice(lattice_list):
    """
    Generalized function to plot cohesive energy versus lattice constant for multiple datasets.

    Parameters:
    - lattice_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, lattice_boundary, color_family, num_samples].
    """
    # Single dataset case
    if not isinstance(lattice_list[0], list):
        return plot_cohesive_energy_lattice_single(lattice_list)
    elif len(lattice_list) == 1:
        return plot_cohesive_energy_lattice_single(*lattice_list)

    # Multi-dataset case: Verify each inner list has the correct format
    if not all(isinstance(data, list) and len(data) == 5 for data in lattice_list):
        print("Error: Each item in lattice_list must be a list with [info_suffix, source_data, lattice_boundary, color_family, num_samples].")
        return

    fig_setting = canvas_setting()
    fig, ax = plt.subplots(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Configure scientific notation for y-axis
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)

    legend_handles = []

    for data in lattice_list:
        info_suffix, source_data, lattice_boundary, color_family, num_samples = data
        colors = color_sampling(color_family)

        # Data input
        data_dict_list = read_energy_parameters(source_data)
        cohesive_energy_values = [d.get("cohesive energy") for d in data_dict_list]
        lattice_constants = [d.get("lattice constant") for d in data_dict_list]

        # Filter out None values
        filtered_data = [(lattice, energy) for lattice, energy in zip(lattice_constants, cohesive_energy_values)
                         if lattice is not None and energy is not None]

        if not filtered_data:
            print(f"No valid data found for plotting for dataset {info_suffix}.")
            continue

        # Unzip filtered data and sort
        lattice_constants, cohesive_energy_values = zip(*filtered_data)
        sorted_data = sorted(zip(lattice_constants, cohesive_energy_values), key=lambda x: x[0])
        lattice_sorted, cohesive_energy_sorted = zip(*sorted_data)

        # Define boundaries for lattice constants
        if lattice_boundary in [None, ""]:
            lattice_start = min(lattice_sorted)
            lattice_end = max(lattice_sorted)
        else:
            lattice_start = float(lattice_boundary[0]) if lattice_boundary[0] else min(lattice_sorted)
            lattice_end = float(lattice_boundary[1]) if lattice_boundary[1] else max(lattice_sorted)

        # Filter data within the specified boundary
        lattice_filtered = []
        cohesive_filtered = []
        for lattice, energy in zip(lattice_sorted, cohesive_energy_sorted):
            if lattice_start <= lattice <= lattice_end:
                lattice_filtered.append(lattice)
                cohesive_filtered.append(energy)

        if not lattice_filtered:
            print(f"No data within the specified lattice boundary for dataset {info_suffix}.")
            continue

        # Estimate EOS parameters and fitted energy values using all data
        eos_params, resampled_lattice, fitted_energy = fit_birch_murnaghan(lattice_filtered, cohesive_filtered, sample_count=100)

        # Plot the fitted EOS curve
        ax.plot(resampled_lattice, fitted_energy, color=colors[1], lw=1.5, label=f"Fitted EOS Curve {info_suffix}")

        # Select scatter sample points based on approximately equal intervals in x-axis values
        if num_samples is None or num_samples >= len(lattice_filtered):
            scatter_lattice = lattice_filtered
            scatter_energy = cohesive_filtered
        else:
            # Define equally spaced x-axis values within the lattice boundary
            x_samples = np.linspace(lattice_start, lattice_end, num_samples)
            scatter_lattice = []
            scatter_energy = []
            for x in x_samples:
                # Find the data point closest to the x sample
                idx = (np.abs(np.array(lattice_filtered) - x)).argmin()
                scatter_lattice.append(lattice_filtered[idx])
                scatter_energy.append(cohesive_filtered[idx])

        # Remove duplicate points (if any)
        unique_points = set()
        scatter_lattice_unique = []
        scatter_energy_unique = []
        for x, y in zip(scatter_lattice, scatter_energy):
            if x not in unique_points:
                unique_points.add(x)
                scatter_lattice_unique.append(x)
                scatter_energy_unique.append(y)

        # Scatter sample data points
        ax.scatter(scatter_lattice_unique, scatter_energy_unique, s=48, fc="#FFFFFF", ec=colors[1],
                   label=f"Sampled data {info_suffix}", zorder=2)

        # Find and mark the minimum cohesive energy point from the filtered data
        min_energy_idx = np.argmin(cohesive_filtered)
        ax.scatter(lattice_filtered[min_energy_idx], cohesive_filtered[min_energy_idx], s=48,
                   fc=colors[2], ec=colors[2], label=f"Minimum Energy {info_suffix}", zorder=3)

        # Add legend entry
        legend_handle = mlines.Line2D([], [], color=colors[1], marker='o', markersize=6, linestyle='-',
                                      label=f"Dataset {info_suffix}")
        legend_handles.append(legend_handle)

    # Set labels and legend for multi-dataset
    ax.set_xlabel("Lattice constant (Å)")
    ax.set_ylabel("Cohesive energy (eV/atom)")
    ax.set_title("Cohesive energy versus lattice constant")
    ax.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

def plot_cohesive_energy_kpoints_encut_single(kpoints_list, encut_list):
    # Unpack lists to retrieve the new info_suffix (materials information)
    info_suffix_kpoints, kpoints_source_data, kpoints_boundary, kpoints_color_family = kpoints_list
    info_suffix_encut, encut_source_data, encut_boundary, encut_color_family = encut_list

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
    kpoints_start = int(kpoints_boundary[0]) if kpoints_boundary[0] is not None else min(total_kpoints)
    kpoints_end = int(kpoints_boundary[1]) if kpoints_boundary[1] is not None else max(total_kpoints)

    kpoints_indices = [
        index for index, val in enumerate(total_kpoints) if kpoints_start <= val <= kpoints_end
    ]
    cohesive_energy_kpoints_plot = [cohesive_energy_kpoints[index] for index in kpoints_indices]
    kpoints_labels_plot = [kpoints_labels[index] for index in kpoints_indices]

    # Set ENCUT boundary
    encut_start = float(encut_boundary[0]) if encut_boundary[0] is not None else min(encut_values_sorted)
    encut_end = float(encut_boundary[1]) if encut_boundary[1] is not None else max(encut_values_sorted)

    encut_indices = [
        index for index, val in enumerate(encut_values_sorted) if encut_start <= val <= encut_end
    ]
    encut_values_plot = [encut_values_sorted[index] for index in encut_indices]
    cohesive_energy_encut_plot = [cohesive_energy_encut_sorted[index] for index in encut_indices]

    # Plot K-points data with fixed spacing on x-axis
    ax_kpoints.plot(range(len(cohesive_energy_kpoints_plot)), cohesive_energy_kpoints_plot, c=kpoints_colors[1], lw=1.5)
    ax_kpoints.scatter(range(len(cohesive_energy_kpoints_plot)), cohesive_energy_kpoints_plot, s=6, c=kpoints_colors[1], zorder=1)
    ax_kpoints.set_xlabel("K-points configuration", color=kpoints_colors[0])
    ax_kpoints.set_ylabel("Cohesive energy (eV/atom)")
    ax_kpoints.set_xticks(range(len(kpoints_labels_plot)))
    ax_kpoints.set_xticklabels(kpoints_labels_plot, rotation=45, ha="right", color=kpoints_colors[0])

    # Plot ENCUT data on the top x-axis
    ax_encut.plot(encut_values_plot, cohesive_energy_encut_plot, c=encut_colors[1], lw=1.5)
    ax_encut.scatter(encut_values_plot, cohesive_energy_encut_plot, s=6, c=encut_colors[1], zorder=1)
    ax_encut.set_xlabel("Energy cutoff (eV)", color=encut_colors[0])
    ax_encut.xaxis.set_label_position("top")
    ax_encut.xaxis.tick_top()
    for encut_label in ax_encut.get_xticklabels():
        encut_label.set_color(encut_colors[0])

    # Create unified legend
    kpoints_legend = mlines.Line2D([], [], color=kpoints_colors[1], marker='o', markersize=6, linestyle='-', 
                                   label=f"Cohesive energy versus K-points {info_suffix_kpoints}")
    encut_legend = mlines.Line2D([], [], color=encut_colors[1], marker='o', markersize=6, linestyle='-', 
                                 label=f"Cohesive energy versus energy cutoff {info_suffix_encut}")
    plt.legend(handles=[kpoints_legend, encut_legend], loc="best")

    plt.title("Cohesive energy versus K-points and energy cutoff")
    plt.tight_layout()

def plot_cohesive_energy_kpoints_encut(kpoints_list_source, encut_list_source):
    """
    Generalized function to plot cohesive energy versus K-points and ENCUT (energy cutoff) for multiple datasets.

    Parameters:
    - kpoints_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, kpoints_boundary, color_family] for k-points data.
    - encut_list: A list of lists, where each inner list contains:
      [info_suffix, source_data, encut_boundary, color_family] for ENCUT data.
    """

    # Ensure kpoints_list and encut_list are always nested lists
    kpoints_list = [kpoints_list_source] if not is_nested_list(kpoints_list_source) else kpoints_list_source
    encut_list = [encut_list_source] if not is_nested_list(encut_list_source) else encut_list_source

    # Downgrade system to handle single data cases
    kpoints_list_downgrad, encut_list_downgrad = [], []
    if not is_nested_list(kpoints_list_source):
        kpoints_list_downgrad = kpoints_list_source
    elif len(kpoints_list_source) == 1:
        kpoints_list_downgrad = kpoints_list_source[0]
    
    if not is_nested_list(encut_list_source):
        encut_list_downgrad = encut_list_source
    elif len(encut_list_source) == 1:
        encut_list_downgrad = encut_list_source[0]
    
    if encut_list_downgrad and kpoints_list_downgrad:
        return plot_cohesive_energy_kpoints_encut_single(kpoints_list_downgrad, encut_list_downgrad)

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

    # Collect all filtered K-points for global alignment
    global_kpoints = set()

    # Process K-points datasets for cohesive energy
    all_kpoints_labels = []
    for kpoints_data_item in kpoints_list:
        info_suffix, kpoints_source_data, kpoints_boundary, kpoints_color_family = kpoints_data_item
        kpoints_colors = color_sampling(kpoints_color_family)

        # K-points data processing for cohesive energy
        kpoints_data = read_energy_parameters(kpoints_source_data)
        cohesive_energy_kpoints = [d.get("cohesive energy") for d in kpoints_data]
        total_kpoints = [d.get("total kpoints") for d in kpoints_data]
        directs_kpoints = [d.get("kpoints mesh") for d in kpoints_data]
        kpoints_labels = {kp: f"({coord[0]}, {coord[1]}, {coord[2]})" for kp, coord in zip(total_kpoints, directs_kpoints)}

        # Apply boundary to filter K-points and cohesive energy values
        kpoints_start = kpoints_boundary[0] if kpoints_boundary[0] is not None else min(total_kpoints)
        kpoints_end = kpoints_boundary[1] if kpoints_boundary[1] is not None else max(total_kpoints)
        filtered_data = [
            (kp, en, kpoints_labels[kp]) for kp, en in zip(total_kpoints, cohesive_energy_kpoints)
            if kpoints_start <= kp <= kpoints_end
        ]

        # Sort filtered data by total_kpoints and unpack
        filtered_data.sort(key=lambda x: x[0])
        filtered_kpoints, filtered_energy, filtered_labels = zip(*filtered_data)

        # Update global_kpoints for consistent alignment across datasets
        global_kpoints.update(filtered_kpoints)

        # Plot K-points data
        ax_kpoints.plot(range(len(filtered_kpoints)), filtered_energy, c=kpoints_colors[1], lw=1.5)
        ax_kpoints.scatter(range(len(filtered_kpoints)), filtered_energy, s=6, c=kpoints_colors[1], zorder=1)

        # Add to legend handles
        legend_handles.append(mlines.Line2D([], [], color=kpoints_colors[1], marker='o', markersize=6, linestyle='-', label=f"Cohesive energy versus K-points {info_suffix}"))
        all_kpoints_labels.extend(filtered_labels)

    # Set x-axis labels for K-points, ensuring unique and ordered labels
    all_kpoints_labels = sorted(set(all_kpoints_labels), key=all_kpoints_labels.index)
    ax_kpoints.set_xticks(range(len(all_kpoints_labels)))
    ax_kpoints.set_xticklabels(all_kpoints_labels, rotation=45, ha="right")
    ax_kpoints.set_xlabel("K-points configuration")
    ax_kpoints.set_ylabel("Cohesive energy (eV/atom)")

    # Process ENCUT datasets for cohesive energy
    all_encut_values = []
    for encut_data_item in encut_list:
        info_suffix, encut_source_data, encut_boundary, encut_color_family = encut_data_item
        encut_colors = color_sampling(encut_color_family)

        # ENCUT data processing for cohesive energy
        encut_data = read_energy_parameters(encut_source_data)
        cohesive_energy_encut = [d.get("cohesive energy") for d in encut_data]
        encut_values = [d.get("energy cutoff (ENCUT)") for d in encut_data]
        encut_filtered_data = [(enc, en) for enc, en in zip(encut_values, cohesive_energy_encut) if enc is not None and en is not None]
        encut_sorted_data = sorted(encut_filtered_data, key=lambda x: x[0])
        encut_values_sorted, cohesive_energy_encut_sorted = zip(*encut_sorted_data)

        # Set ENCUT boundary
        encut_start = float(encut_boundary[0]) if encut_boundary[0] is not None else min(encut_values_sorted)
        encut_end = float(encut_boundary[1]) if encut_boundary[1] is not None else max(encut_values_sorted)

        # Filter sorted data within boundary
        encut_indices = [
            index for index, val in enumerate(encut_values_sorted) if encut_start <= val <= encut_end
        ]
        encut_values_plot = [encut_values_sorted[index] for index in encut_indices]
        cohesive_energy_encut_plot = [cohesive_energy_encut_sorted[index] for index in encut_indices]
        all_encut_values.extend(encut_values_plot)

        # Plot ENCUT data on the top x-axis
        ax_encut.plot(encut_values_plot, cohesive_energy_encut_plot, c=encut_colors[1], lw=1.5)
        ax_encut.scatter(encut_values_plot, cohesive_energy_encut_plot, s=6, c=encut_colors[1], zorder=1)

        # Add to legend handles
        legend_handles.append(mlines.Line2D([], [], color=encut_colors[1], marker='o', markersize=6, linestyle='-', label=f"Cohesive energy versus Energy cutoff {info_suffix}"))

    # Set top x-axis labels for ENCUT, using MaxNLocator for fewer ticks
    ax_encut.set_xlabel("Energy cutoff (eV)")
    ax_encut.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
    ax_encut.xaxis.set_label_position("top")
    ax_encut.xaxis.tick_top()

    # Create unified legend
    plt.legend(handles=legend_handles, loc="best")
    plt.title("Cohesive energy versus K-points and energy cutoff")
    plt.tight_layout()

# Scheduling functions

def plot_energy_parameters(parameters, *args):
    if isinstance(parameters, str):
        param = parameters.lower()
        args_list = args[0] if len(args) == 1 and isinstance(args[0], list) else list(args)
        if param in ["kpoint", "kpoints", "k-point", "k-points"]:
            return plot_energy_kpoints(args_list)
        elif param in ["encut", "energy cutoff", "energy_cutoff", "energy-cutoff"]:
            return plot_energy_encut(args_list)
        elif param in ["lattice", "lattice constant"]:
            return plot_energy_lattice(args_list)
        else:
            print("Parameter type not recognized or incorrect arguments. To be continued.")
    elif isinstance(parameters, (tuple, list)):
        # If the list or tuple length is 1, treat it as a single string case
        if len(parameters) == 1:
            return plot_energy_parameters(parameters[0], *args)
        # Handle combination of two parameters
        elif len(parameters) == 2:
            first_param, second_param = parameters
            # Standardize parameter names for easy comparison
            first_param = first_param.lower()
            second_param = second_param.lower()
            if first_param in ["kpoints", "k-point", "kpoints configuration"] and second_param in ["encut", "energy cutoff", "energy_cutoff", "energy-cutoff"]:
                return plot_energy_kpoints_encut(*args)
            elif first_param in ["encut", "energy cutoff", "energy_cutoff", "energy-cutoff"] and second_param in ["kpoints", "k-point", "kpoints configuration"]:
                print("Reordering parameters to [\"kpoints\", \"encut\"] and plotting.")
                # Reorder parameters to call plot_energy_kpoints_encut
                return plot_energy_kpoints_encut(args[1], args[0])
            else: print("Combination of parameters not recognized.")
        else: print("Error: Only one or two parameters are allowed for plot types.")
    else: print("Invalid input type for parameters.")

def plot_energy_parameter(*args):
    # Alias for single-parameter usage
    return plot_energy_parameters(*args)

def plot_cohesive_energy_parameters(parameters, *args):
    if isinstance(parameters, str):
        param = parameters.lower()
        args_list = args[0] if len(args) == 1 and isinstance(args[0], list) else list(args)
        if param in ["kpoint", "kpoints", "k-point", "k-points"]:
            return plot_cohesive_energy_kpoints(args_list)
        elif param in ["encut", "energy cutoff", "energy_cutoff", "energy-cutoff"]:
            return plot_cohesive_energy_encut(args_list)
        else:
            print("Parameter type not recognized or incorrect arguments. To be continued.")
    elif isinstance(parameters, (tuple, list)):
        # If the list or tuple length is 1, treat it as a single string case
        if len(parameters) == 1:
            return plot_cohesive_energy_parameters(parameters[0], *args)
        # Handle combination of two parameters
        elif len(parameters) == 2:
            first_param, second_param = parameters
            # Standardize parameter names for easy comparison
            first_param = first_param.lower()
            second_param = second_param.lower()
            if first_param in ["kpoints", "k-point", "kpoints configuration"] and second_param in ["encut", "energy cutoff", "energy_cutoff", "energy-cutoff"]:
                return plot_cohesive_energy_kpoints_encut(*args)
            elif first_param in ["encut", "energy cutoff", "energy_cutoff", "energy-cutoff"] and second_param in ["kpoints", "k-point", "kpoints configuration"]:
                print("Reordering parameters to [\"kpoints\", \"encut\"] and plotting.")
                # Reorder parameters to call plot_cohesive_energy_kpoints_encut
                return plot_cohesive_energy_kpoints_encut(args[1], args[0])
            else:
                print("Combination of parameters not recognized.")
        else:
            print("Error: Only one or two parameters are allowed for plot types.")
    else:
        print("Invalid input type for parameters.")

def plot_cohesive_energy_parameter(*args):
    # Alias for single-parameter usage
    return plot_cohesive_energy_parameters(*args)
