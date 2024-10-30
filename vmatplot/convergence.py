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
    """Find folders with vasprun.xml and extract energy, total kpoints, calculated kpoints, kpoints (x, y, z), lattice constant, SYMPREC, energy cutoff, k-point spacing, volume, time step, mixing parameter (AMIX), mixing parameter (BMIX), electronic convergence (EDIFF), and force convergence (EDIFFG) values."""

    if directory == "help":
        print("Please use this function on the project directory.")
        return "Help provided."

    vasprun_path = os.path.join(directory, "vasprun.xml")
    kpoints_path = os.path.join(directory, "KPOINTS")

    if not os.path.exists(vasprun_path):
        return "vasprun.xml file not found in the specified directory."
    if not os.path.exists(kpoints_path):
        return "KPOINTS file not found in the specified directory."

    # Parse the vasprun.xml file
    try:
        tree = ET.parse(vasprun_path)
        root = tree.getroot()

        # Initialize a dictionary to store the parameters
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

        # Extract SYMPREC parameter from INCAR settings in vasprun.xml
        symprec_tag = root.find(".//parameters/separator/i[@name='SYMPREC']")
        if symprec_tag is not None:
            parameters["symmetry precision (SYMPREC)"] = float(symprec_tag.text)

        # Extract INCAR parameters like ENCUT, KSPACING, POTIM, AMIX, BMIX, EDIFF, EDIFFG
        for param in root.findall(".//parameters/separator/i"):
            name = param.get("name")
            if name == "ENCUT":
                parameters["energy cutoff (ENCUT)"] = float(param.text)
            elif name == "KSPACING":
                parameters["k-point spacing (KSPACING)"] = float(param.text)
            elif name == "POTIM":
                parameters["time step (POTIM)"] = float(param.text)
            elif name == "AMIX":
                parameters["mixing parameter (AMIX)"] = float(param.text)
            elif name == "BMIX":
                parameters["mixing parameter (BMIX)"] = float(param.text)
            elif name == "EDIFF":
                parameters["electronic convergence (EDIFF)"] = float(param.text)
            elif name == "EDIFFG":
                parameters["force convergence (EDIFFG)"] = float(param.text)

        # Extract volume from the structure information
        volume_tag = root.find(".//structure[@name='finalpos']/crystal/volume")
        if volume_tag is not None:
            parameters["volume"] = float(volume_tag.text)

        # Extract k-points information
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

        # Calculate calculated kpoints (reduced due to symmetry) from vasprun.xml
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

                # Extract kpoints information
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

                    # Extract additional parameters from vasprun.xml
                    for param in root.findall(".//parameters/separator/i"):
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

def identify_kpoints_type(directory="."):
    """Find folders with KPOINTS and print its type."""
    # Key words
    automatic = "Automatic k-point grid"
    explicit = "Explicit k-points listed"
    linear = "Linear mode"

    # Check if the user asked for help
    if directory == "help":
        print("Please use this function on the project directory.")
        return "Help provided."

    kpoints_path = os.path.join(directory, "KPOINTS")
    if not os.path.exists(kpoints_path):
        return "KPOINTS file not found in the specified directory."

    with open(kpoints_path, "r", encoding="utf-8") as kpoints_file:
        lines = kpoints_file.readlines()
        if len(lines) < 3:
            return "Invalid file format, unable to identify"
        second_line = lines[1].strip()
        third_line = lines[2].strip()
        if second_line == "0":
            if "Gamma" in third_line or "Monkhorst" in third_line or "Monkhorst-pack" in third_line:
                return automatic
            else: return "Invalid file format, unable to identify"
        elif second_line.isdigit():
            if "Explicit" in third_line:
                return explicit
            elif "Line-mode" in third_line:
                return linear
            else: return "Invalid file format, unable to identify"
        else: return "Invalid file format, unable to identify"

def read_kpoints_energy(data_path):
    help_info = "Usage: read_kpoints_energy(data_path)\n" + \
                "data_path: Path to the data file containing lattice and energy values.\n"

    # Check if the user asked for help
    if data_path == "help":
        print(help_info)
        return

    # Initialize the returns
    energy, kpoints, cal_kpoints, directs_kpoints, lattice = [], [], [], [], []

    with open(data_path, "r", encoding="utf-8") as data_file:
        lines = data_file.readlines()[1:]
        for line in lines:
            split_line = line.strip().split('\t')
            energy.append(float(split_line[0]))
            kpoints.append(int(split_line[1]))
            cal_kpoints.append(int(split_line[2]))
            # Extract x, y, z from the tuple format (x, y, z)
            x, y, z = map(int, split_line[3][1:-1].split(','))
            directs_kpoints.append((x, y, z))
            lattice.append(float(split_line[4]))

    return energy, kpoints, cal_kpoints, directs_kpoints, lattice

def plot_energy_kpoints(matter, source_data=None, kpoints_boundary=None, color_family="blue"):
    help_info = "Usage: plot_kpoints_energy(matter, source_data, kpoints_boundary, color_family)\n" + \
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

    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    plt.gca().yaxis.set_major_formatter(formatter)

    # Color calling
    colors = color_sampling(color_family)

    # Data input
    data_input = read_kpoints_energy(source_data)
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
    plt.xlabel("Kpoints Configuration")
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
    plt.scatter(range(len(kpoints_plotting)), energy_plotting, s=5, c=colors[1], zorder=1)
    plt.plot(range(len(kpoints_plotting)), energy_plotting, c=colors[1], lw=1.5)

    # Set custom tick labels for x-axis to show kpoints configurations
    kpoints_labels = [f"({x[0]}, {x[1]}, {x[2]})" for x in kpoints_plotting]
    plt.xticks(ticks=range(len(kpoints_labels)), labels=kpoints_labels, rotation=45, ha="right")

    plt.tight_layout()
