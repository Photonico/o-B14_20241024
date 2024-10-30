#### Convergence test
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0914

import xml.etree.ElementTree as ET
import os

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import ScalarFormatter
from vmatplot.commons import check_vasprun
from vmatplot.output import canvas_setting, color_sampling

def identify_parameters(directory="."):
    """Extract energy, total kpoints, calculated kpoints, kpoints (x, y, z), lattice constant, SYMPREC, ENCUT, KSPACING, VOLUME, POTIM, AMIX, BMIX, EDIFF, EDIFFG values."""

    if directory == "help":
        print("Please use this function on the project directory.")
        return "Help provided."

    vasprun_path = os.path.join(directory, "vasprun.xml")
    kpoints_path = os.path.join(directory, "KPOINTS")

    if not os.path.exists(vasprun_path):
        return "vasprun.xml file not found in the specified directory."
    if not os.path.exists(kpoints_path):
        return "KPOINTS file not found in the specified directory."

    try:
        tree = ET.parse(vasprun_path)
        root = tree.getroot()

        parameters = {
            "total energy": None,
            "total kpoints": None,
            "calculated kpoints": None,
            "kpoints grid": (None, None, None),
            "lattice constant": None,
            "symmetry precision (SYMPREC)": None,
            "energy cutoff (ENCUT)": None,
            "k-point spacing (KSPACING)": None,
            "volume": None,
            "time step (POTIM)": None,
            "mixing parameter (AMIX)": None,
            "mixing parameter (BMIX)": None,
            "electronic convergence (EDIFF)": None,
            "force convergence (EDIFFG)": None
        }

        # Extract final total energy
        energy_tag = root.find(".//calculation/energy/i[@name='e_fr_energy']")
        if energy_tag is not None:
            parameters["total energy"] = float(energy_tag.text)

        # Extract lattice constant (assuming it's the length of the "a" vector from the final structure)
        basis_vectors = root.findall(".//calculation/structure/crystal/varray[@name='basis']")[-1]
        a_vector = basis_vectors[0].text.split()
        parameters["lattice constant"] = (float(a_vector[0])**2 + float(a_vector[1])**2 + float(a_vector[2])**2)**0.5

        # Extract SYMPREC from the parameters
        symprec_tag = root.find(".//parameters/separator/i[@name='SYMPREC']")
        if symprec_tag is not None:
            parameters["symmetry precision (SYMPREC)"] = float(symprec_tag.text)

        # Extract INCAR parameters from the `<incar>` tag directly under the root
        incar_parameters = {
            "ENCUT": "energy cutoff (ENCUT)",
            "KSPACING": "k-point spacing (KSPACING)",
            "POTIM": "time step (POTIM)",
            "AMIX": "mixing parameter (AMIX)",
            "BMIX": "mixing parameter (BMIX)",
            "EDIFF": "electronic convergence (EDIFF)",
            "EDIFFG": "force convergence (EDIFFG)"
        }

        for param in root.findall(".//incar/i"):
            name = param.get("name")
            if name in incar_parameters:
                parameters[incar_parameters[name]] = float(param.text)

        # Extract volume from the structure information
        volume_tag = root.find(".//structure[@name='finalpos']/crystal/volume")
        if volume_tag is not None:
            parameters["volume"] = float(volume_tag.text)

        # Extract k-points information from KPOINTS file
        with open(kpoints_path, "r", encoding="utf-8") as kpoints_file:
            lines = kpoints_file.readlines()
            for index, line in enumerate(lines):
                if any(keyword in line.lower() for keyword in ["gamma", "monkhorst", "monkhorst-pack", "explicit", "line-mode"]):
                    kpoints_index = index + 1
                    break
            else:
                raise ValueError("Kpoints type keyword not found in KPOINTS file.")

            kpoints_values = lines[kpoints_index].split()
            x_kpoints, y_kpoints, z_kpoints = int(kpoints_values[0]), int(kpoints_values[1]), int(kpoints_values[2])
            tot_kpoints = x_kpoints * y_kpoints * z_kpoints
            parameters["total kpoints"] = tot_kpoints
            parameters["kpoints grid"] = (x_kpoints, y_kpoints, z_kpoints)

        # Calculate calculated kpoints (reduced due to symmetry)
        cal_kpoints_tag = root.find(".//kpoints/varray[@name='kpointlist']")
        if cal_kpoints_tag is not None:
            parameters["calculated kpoints"] = len(cal_kpoints_tag.findall("v"))

        # Check for missing values
        missing_params = [k for k, v in parameters.items() if v is None]
        if missing_params:
            print(f"Warning: Some parameters could not be found in vasprun.xml or KPOINTS file: {', '.join(missing_params)}")

        return parameters

    except (ET.ParseError, ValueError, IndexError) as e:
        print(f"Error parsing files in {directory}: {e}")
        return None

def summarize_energy_parameters(directory=".", lattice_boundary=None):
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

        if os.path.isfile(xml_path) and os.path.isfile(kpoints_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

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
                    else:
                        raise ValueError("Kpoints type keyword not found in KPOINTS file.")

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

                if lattice_within_start and lattice_within_end:
                    # Collect all required parameters in order
                    params = {
                        "total energy": energy,
                        "total kpoints": tot_kpoints,
                        "kpoints grid": (x_kpoints, y_kpoints, z_kpoints),
                        "lattice constant": lattice_constant,
                        "symmetry precision (SYMPREC)": symprec,
                        "energy cutoff (ENCUT)": None,
                        "k-point spacing (KSPACING)": None,
                        "volume": None,
                        "time step (POTIM)": None,
                        "mixing parameter (AMIX)": None,
                        "mixing parameter (BMIX)": None,
                        "electronic convergence (EDIFF)": None,
                        "force convergence (EDIFFG)": None
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

                    # Append results in the specified order
                    results.append((
                        params["total energy"], params["total kpoints"], kpoints_sum, params["kpoints grid"], 
                        params["lattice constant"], params["symmetry precision (SYMPREC)"], params["energy cutoff (ENCUT)"],
                        params["k-point spacing (KSPACING)"], params["volume"], params["time step (POTIM)"], 
                        params["mixing parameter (AMIX)"], params["mixing parameter (BMIX)"], 
                        params["electronic convergence (EDIFF)"], params["force convergence (EDIFFG)"]
                    ))

            except (ET.ParseError, ValueError, IndexError) as e:
                print(f"Error parsing files in {work_dir}: {e}")

        else:
            print(f"vasprun.xml or KPOINTS is not found in {work_dir}.")

    # Sort results by the sum of Kpoints
    results.sort(key=lambda x: x[2])  # Sort by kpoints_sum

    # Write the sorted results to the output file
    try:
        with open(result_file_path, "w", encoding="utf-8") as f:
            f.write("Total Energy\tTotal Kpoints\tCalculated Kpoints\tKpoints(X Y Z)\tLattice Constant\tSYMPREC\tENCUT\tKSPACING\tVOLUME\tPOTIM\tAMIX\tBMIX\tEDIFF\tEDIFFG\n")
            for result in results:
                f.write("\t".join(map(str, result)) + "\n")
    except IOError as e:
        print(f"Error writing to file at {result_file_path}:", e)

def read_energy_parameters(data_path):
    help_info = "Usage: read_energy_parameters(data_path)\n" + \
                "data_path: Path to the data file containing energy and various VASP parameters.\n"

    # Check if the user asked for help
    if data_path == "help":
        print(help_info)
        return

    # Initialize the returns for each parameter
    energy, total_kpoints, cal_kpoints_sum, directs_kpoints, lattice = [], [], [], [], []
    symprec, encut, kspacing, volume, potim, amix, bmix, ediff, ediffg = [], [], [], [], [], [], [], [], []

    # Read data file and parse each line to extract parameters
    with open(data_path, "r", encoding="utf-8") as data_file:
        lines = data_file.readlines()[1:]  # Skip the header line
        for line in lines:
            split_line = line.strip().split('\t')
            energy.append(float(split_line[0]))
            total_kpoints.append(int(split_line[1]))
            cal_kpoints_sum.append(int(split_line[2]))

            # Extract x, y, z from the tuple format (x, y, z)
            x, y, z = map(int, split_line[3][1:-1].split(','))
            directs_kpoints.append((x, y, z))

            lattice.append(float(split_line[4]))
            symprec.append(float(split_line[5]) if split_line[5] != 'None' else None)
            encut.append(float(split_line[6]) if split_line[6] != 'None' else None)
            kspacing.append(float(split_line[7]) if split_line[7] != 'None' else None)
            volume.append(float(split_line[8]) if split_line[8] != 'None' else None)
            potim.append(float(split_line[9]) if split_line[9] != 'None' else None)
            amix.append(float(split_line[10]) if split_line[10] != 'None' else None)
            bmix.append(float(split_line[11]) if split_line[11] != 'None' else None)
            ediff.append(float(split_line[12]) if split_line[12] != 'None' else None)
            ediffg.append(float(split_line[13]) if split_line[13] != 'None' else None)

    return energy, total_kpoints, cal_kpoints_sum, directs_kpoints, lattice, symprec, encut, kspacing, volume, potim, amix, bmix, ediff, ediffg

def plot_energy_kpoints(matter, source_data=None, kpoints_boundary=None, color_family="blue"):
    help_info = "Usage: plot_energy_kpoints(matter, source_data, kpoints_boundary, color_family)\n" + \
                "source_data: Path to the data file containing energy, kpoints, and lattice values.\n"

    # Check if the user asked for help
    if matter == "help" and source_data is None:
        print(help_info)
        return

    # Figure Settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Apply ScalarFormatter with scientific notation limits
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))      # Set scientific notation limits
    plt.gca().yaxis.set_major_formatter(formatter)

    # Color calling
    colors = color_sampling(color_family)

    # Data input
    data_input = read_energy_parameters(source_data)
    energy = data_input[0]        # First column (energy)
    tot_kpoints = data_input[1]   # Total kpoints
    cal_kpoints = data_input[2]
    sep_kpoints = data_input[3]   # List of kpoints configurations like [(1, 1, 1), (2, 3, 2), ...]
    lattice = data_input[4]       # Lattice constant

    # Calculate kpoints sum and sort based on it
    kpoints_sum = [sum(kpoint) for kpoint in sep_kpoints]
    sorted_data = sorted(zip(kpoints_sum, energy, sep_kpoints), key=lambda x: x[0])
    kpoints_sum_sorted, energy_sorted, sep_kpoints_sorted = zip(*sorted_data)

    # Figure title
    plt.title(f"Energy versus K-points {matter}")
    plt.xlabel("K-points Configuration")
    plt.ylabel(r"Energy (eV)")

    # Ensure boundary values are integers
    if kpoints_boundary in [None, ""]:
        kpoints_start = min(kpoints_sum_sorted)
        kpoints_end = max(kpoints_sum_sorted)
    else:
        kpoints_start = int(kpoints_boundary[0]) if isinstance(kpoints_boundary[0], (int, str)) and kpoints_boundary[0] != "" else min(kpoints_sum_sorted)
        kpoints_end = int(kpoints_boundary[1]) if isinstance(kpoints_boundary[1], (int, str)) and kpoints_boundary[1] != "" else max(kpoints_sum_sorted)

    # Find indices for the specified boundary in sorted data
    start_index = next((i for i, val in enumerate(kpoints_sum_sorted) if val >= kpoints_start), 0)
    end_index = next((i for i, val in enumerate(kpoints_sum_sorted) if val > kpoints_end), len(kpoints_sum_sorted)) - 1

    # Prepare kpoints configurations and energy values for plotting in sorted order
    kpoints_plotting = sep_kpoints_sorted[start_index:end_index + 1]
    energy_plotting = energy_sorted[start_index:end_index + 1]

    # Plotting
    plt.scatter(range(len(kpoints_plotting)), energy_plotting, size=6, c=colors[1], zorder=1)
    plt.plot(range(len(kpoints_plotting)), energy_plotting, c=colors[1], lw=1.5)

    # Set custom tick labels for x-axis to show kpoints configurations
    kpoints_labels = [f"({x[0]}, {x[1]}, {x[2]})" for x in kpoints_plotting]
    plt.xticks(ticks=range(len(kpoints_labels)), labels=kpoints_labels, rotation=45, ha="right")

    plt.tight_layout()

def plot_energy_encut(matter, source_data=None, encut_boundary=None, color_family="blue"):
    help_info = "Usage: plot_energy_encut(matter, source_data, encut_boundary, color_family)\n" + \
                "source_data: Path to the data file containing energy and ENCUT values.\n"

    # Check if the user asked for help
    if matter == "help" and source_data is None:
        print(help_info)
        return

    # Figure Settings
    fig_setting = canvas_setting()
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Apply ScalarFormatter with scientific notation limits
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-3, 3))      # Set scientific notation limits
    plt.gca().yaxis.set_major_formatter(formatter)

    # Color calling
    colors = color_sampling(color_family)

    # Data input
    data_input = read_energy_parameters(source_data)
    energy = data_input[0]          # First column (energy)
    encut_values = data_input[6]    # ENCUT values (seventh column)

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

    # Figure title
    plt.title(f"Energy versus energy cutoff {matter}")
    plt.xlabel("Energy cutoff (eV)")
    plt.ylabel(r"Energy (eV)")

    # Set boundaries for ENCUT
    if encut_boundary in [None, ""]:
        encut_start = min(encut_sorted)
        encut_end = max(encut_sorted)
    else:
        encut_start = float(encut_boundary[0]) if isinstance(encut_boundary[0], (int, float, str)) and encut_boundary[0] != "" else min(encut_sorted)
        encut_end = float(encut_boundary[1]) if isinstance(encut_boundary[1], (int, float, str)) and encut_boundary[1] != "" else max(encut_sorted)

    # Find indices for the specified boundary in sorted data
    start_index = next((i for i, val in enumerate(encut_sorted) if val >= encut_start), 0)
    end_index = next((i for i, val in enumerate(encut_sorted) if val > encut_end), len(encut_sorted)) - 1

    # Prepare ENCUT values and energy values for plotting within specified boundary
    encut_plotting = encut_sorted[start_index:end_index + 1]
    energy_plotting = energy_sorted[start_index:end_index + 1]

    # Plotting
    plt.scatter(encut_plotting, energy_plotting, size=6, c=colors[1], zorder=1)
    plt.plot(encut_plotting, energy_plotting, c=colors[1], lw=1.5)

    plt.tight_layout()
