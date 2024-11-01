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
from vmatplot.algorithms import is_nested_list
from vmatplot.output_settings import canvas_setting, color_sampling

def summarize_parameters(directory=".", lattice_boundary=None):
    result_file = "parameters_energy.dat"
    result_file_path = os.path.join(directory, result_file)

    if directory == "help":
        print("Please use this function on the parent directory of the project's main folder.")
        return []

    if not os.path.exists(directory):
        os.makedirs(directory)

    dirs_to_walk = check_vasprun(directory)
    results = []

    lattice_within_start = True
    lattice_within_end = True

    for work_dir in dirs_to_walk:
        xml_path = os.path.join(work_dir, "vasprun.xml")
        kpoints_path = os.path.join(work_dir, "KPOINTS")
        outcar_path = os.path.join(work_dir, "OUTCAR")

        if os.path.isfile(xml_path) and os.path.isfile(kpoints_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Extract total atom count
                atom_count_tag = root.find(".//atominfo/atoms")
                total_atom_count = int(atom_count_tag.text) if atom_count_tag is not None else None

                # Extract total energy
                energy = float(root.findall(".//calculation/energy/i[@name='e_fr_energy']")[-1].text)

                # Extract lattice constant
                basis_vectors = root.findall(".//calculation/structure/crystal/varray[@name='basis']")[-1]
                a_vector = basis_vectors[0].text.split()
                lattice_constant = (float(a_vector[0])**2 + float(a_vector[1])**2 + float(a_vector[2])**2)**0.5

                # Extract SYMPREC for symmetry precision
                symprec_tag = root.find(".//parameters/separator/i[@name='SYMPREC']")
                symprec = float(symprec_tag.text) if symprec_tag is not None else None

                # Extract kpoints information from KPOINTS file
                with open(kpoints_path, "r", encoding="utf-8") as kpoints_file:
                    lines = kpoints_file.readlines()
                    for index, line in enumerate(lines):
                        if any(keyword in line.lower() for keyword in ["gamma", "monkhorst", "monkhorst-pack", "explicit", "line-mode"]):
                            kpoints_index = index + 1
                            break
                    else: raise ValueError("Kpoints type keyword not found in KPOINTS file.")

                    kpoints_values = lines[kpoints_index].split()
                    x_kpoints, y_kpoints, z_kpoints = int(kpoints_values[0]), int(kpoints_values[1]), int(kpoints_values[2])
                    tot_kpoints = x_kpoints * y_kpoints * z_kpoints
                    kpoints_sum = x_kpoints + y_kpoints + z_kpoints  # For sorting
                # Tolerance check for lattice boundary
                TOLERANCE = 1e-6
                if lattice_boundary is not None:
                    lattice_start, lattice_end = lattice_boundary
                    lattice_within_start = lattice_start in [None, ""] or lattice_constant >= lattice_start - TOLERANCE
                    lattice_within_end = lattice_end in [None, ""] or lattice_constant <= lattice_end + TOLERANCE
                # Initialize elapsed time variable
                elapsed_time = None
                if os.path.isfile(outcar_path):
                    with open(outcar_path, "r", encoding="utf-8") as outcar_file:
                        for line in outcar_file:
                            if "Elapsed time (sec):" in line:
                                elapsed_time = float(line.split(":")[-1].strip())
                                break
                if lattice_within_start and lattice_within_end:
                    # Collect all required parameters in order
                    params = {
                        "total atom count": total_atom_count,
                        "total energy": energy,
                        "total kpoints": tot_kpoints,
                        "kpoints mesh": (x_kpoints, y_kpoints, z_kpoints),
                        "lattice constant": lattice_constant,
                        "symmetry precision (SYMPREC)": symprec,
                        "energy cutoff (ENCUT)": None,
                        "k-point spacing (KSPACING)": None,
                        "volume": None,
                        "time step (POTIM)": None,
                        "mixing parameter (AMIX)": None,
                        "mixing parameter (BMIX)": None,
                        "electronic convergence (EDIFF)": None,
                        "force convergence (EDIFFG)": None,
                        "elapsed time (sec)": elapsed_time
                    }
                    # Extract additional parameters from <incar> section
                    for param in root.findall(".//incar/i"):
                        name = param.get("name")
                        if name == "ENCUT":
                            params["energy cutoff (ENCUT)"] = float(param.text)
                        elif name == "KSPACING":
                            params["k-point spacing (KSPACING)"] = float(param.text)
                        elif name == "POTIM":
                            params["time step (POTIM)"] = float(param.text)
                        elif name == "AMIX":
                            params["mixing parameter (AMIX)"] = float(param.text)
                        elif name == "BMIX":
                            params["mixing parameter (BMIX)"] = float(param.text)
                        elif name == "EDIFF":
                            params["electronic convergence (EDIFF)"] = float(param.text)
                        elif name == "EDIFFG":
                            params["force convergence (EDIFFG)"] = float(param.text)
                    # Extract volume
                    volume_tag = root.find(".//structure[@name='finalpos']/crystal/volume")
                    if volume_tag is not None:
                        params["volume"] = float(volume_tag.text)
                    # Append results as dictionary
                    results.append(params)
            except (ET.ParseError, ValueError, IndexError) as e:
                print(f"Error parsing files in {work_dir}: {e}")
        else: print(f"vasprun.xml or KPOINTS is not found in {work_dir}.")

    # Sort results by elapsed time (sec)
    results.sort(key=lambda x: x["elapsed time (sec)"])

    # Write the sorted results to the output file
    try:
        with open(result_file_path, "w", encoding="utf-8") as f:
            # Write headers based on keys in the first result dictionary
            headers = "\t".join(results[0].keys())
            f.write(headers + "\n")
            for result in results:
                f.write("\t".join(str(result[key]) if result[key] is not None else 'None' for key in result) + "\n")
    except IOError as e:
        print(f"Error writing to file at {result_file_path}:", e)

def read_energy_parameters(data_path):
    help_info = "Usage: read_energy_parameters(data_path)\n" + \
                "data_path: Path to the data file containing energy and various VASP parameters.\n"

    # Check if the user asked for help
    if data_path == "help":
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

def plot_energy_kpoints_single(*args_list):
    help_info = "Usage: plot_energy_kpoints(args_list)\n" + \
                "args_list: A list containing [info_suffix, source_data, kpoints_boundary, color_family].\n" + \
                "Example: plot_energy_kpoints(['Material Info', 'source_data_path', (start, end), 'blue'])\n"

    if not args_list or args_list[0] == "help":
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
    total_kpoints = [d["total kpoints"] for d in data_dict_list]
    energy = [d["total energy"] for d in data_dict_list]
    sep_kpoints = [d["kpoints mesh"] for d in data_dict_list]

    # Sort data based on total_kpoints to maintain order
    sorted_data = sorted(zip(total_kpoints, energy, sep_kpoints), key=lambda x: x[0])
    total_kpoints_sorted, energy_sorted, sep_kpoints_sorted = zip(*sorted_data)

    # Set title with info_suffix
    plt.title(f"Energy versus K-points {info_suffix}")
    plt.xlabel("K-points configuration")
    plt.ylabel(r"Energy (eV)")

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
        total_kpoints = [d["total kpoints"] for d in data_dict_list]
        kpoints_config = [d["kpoints mesh"] for d in data_dict_list]  # Extract kpoints configurations in (X, Y, Z) format

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
        energy = [d["total energy"] for d in data_dict_list]
        total_kpoints = [d["total kpoints"] for d in data_dict_list]

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

    if not args_list or args_list[0] == "help":
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
    plt.title(f"Energy versus Energy Cutoff {info_suffix}")
    plt.xlabel("Energy cutoff (eV)")
    plt.ylabel(r"Energy (eV)")

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

def plot_energy_parameters(parameters, *args):
    if isinstance(parameters, str):
        param = parameters.lower()
        args_list = args[0] if len(args) == 1 and isinstance(args[0], list) else list(args)
        if param in ["kpoint", "kpoints", "k-point", "k-points"]:
            return plot_energy_kpoints(args_list)
        elif param in ["encut", "energy cutoff", "energy_cutoff", "energy-cutoff"]:
            return plot_energy_encut(args_list)
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

# Alias for single-parameter usage
def plot_energy_parameter(*args):
    return plot_energy_parameters(*args)

# def plot_cohesive_energy_single(info_suffix, cohesive_list):