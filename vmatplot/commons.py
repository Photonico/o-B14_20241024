#### Common codes
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914

import xml.etree.ElementTree as ET
import os
import math
import numpy as np

from typing import Tuple, Union

def vector_length(vec):
    return math.sqrt(sum(x**2 for x in vec))

def get_or_default(value, default):
    """Return the value if it's not None or an empty string, otherwise return the default."""
    return default if value in [None, ""] else value

def extract_part(ind_values, dep_values, start_value=None, end_value=None):
    # Ensure ind_values and dep_values are numpy arrays
    ind_values = np.asarray(ind_values)
    dep_values = np.asarray(dep_values)
    # Condition handling:
    # If both start_value and end_value are None, return the original data without processing.
    if start_value is None and end_value is None:
        return (ind_values, dep_values)
    # If only start_value is provided (not None), slice from start_value to the end.
    elif start_value is not None and end_value is None:
        condition = ind_values >= start_value
    # If only end_value is provided (not None), slice from the beginning to end_value.
    elif start_value is None and end_value is not None:
        condition = ind_values <= end_value
    # If both start_value and end_value are provided, slice between start_value and end_value.
    else:
        condition = (ind_values >= start_value) & (ind_values <= end_value)
    # Filtering data based on the condition
    ind_values_filtered = ind_values[condition]
    dep_values_filtered = dep_values[condition]
    return (ind_values_filtered, dep_values_filtered)

def check_vasprun(directory="."):
    """Find folders with complete vasprun.xml and print incomplete ones."""
    # Check if the user asked for help
    if directory == "help":
        print("Please use this function on the parent directory of the project's main folder.")
        return []
    complete_folders = []
    # Traverse all folders under "directory"
    for dirpath, _, filenames in os.walk(directory):
        if "vasprun.xml" in filenames:
            xml_path = os.path.join(dirpath, "vasprun.xml")

            # Check if vasprun.xml is complete
            try:
                with open(xml_path, "r", encoding="utf-8") as xml_file:
                    # Check the last few lines for the closing tag
                    last_lines = xml_file.readlines()[-10:] # read the last 10 lines
                    for line in last_lines:
                        if "</modeling>" in line or "</vasp>" in line:
                            complete_folders.append(dirpath)
                            break
                    else:
                        print(f"vasprun.xml in {dirpath} is incomplete.")
            except IOError as e:    # Change from Exception to IOError
                print(f"Error reading {xml_path}: {e}")
    return complete_folders

def extract_fermi(directory):
    xml_file = os.path.join(directory, "vasprun.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    ## Extract eigen, occupancy number
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        for dos in root.findall("./calculation/dos"):
            comment = dos.get("comment")
            if comment == "kpoints_opt":
                for i in dos.findall("i"):
                    if "name" in i.attrib:
                        if i.attrib["name"] == "efermi":
                            fermi_energy = float(i.text)
                            return fermi_energy
    # GGA-PBE algorithms
    elif os.path.exists(kpoints_file_path):
        for i in root.iter("i"):
            if "name" in i.attrib:
                if i.attrib["name"] == "efermi":
                    fermi_energy = float(i.text)
                    return fermi_energy
    raise ValueError("Fermi energy not found in vasprun.xml")

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

def identify_parameters(directory="."):
    help_info = (
        "Usage Guide for identify_parameters:\n\n"
        "This function extracts important VASP calculation parameters from output files in a specified directory.\n\n"
        "Parameters:\n"
        "  directory : str, optional\n"
        "    Path to the directory containing VASP output files (default is current directory).\n"
        "    Set to 'help' to display this usage guide.\n\n"
        "Returned Values:\n"
        "  A dictionary with the extracted parameters if successful.\n"
        "  Returns None if required files are missing, an error occurs, or 'help' is passed.\n\n"
        "Extracted Parameters:\n"
        "  - Total atom count: Number of atoms in the system.\n"
        "  - Total energy: Total energy of the system.\n"
        "  - Fermi energy: Fermi level.\n"
        "  - Total kpoints: Number of k-points defined in the KPOINTS file.\n"
        "  - Calculated kpoints: Reduced k-points count after symmetry reduction.\n"
        "  - Kpoints mesh: (x, y, z) k-points grid dimensions.\n"
        "  - Lattice constant: Length of the 'a' lattice vector.\n"
        "  - SYMPREC: Symmetry precision parameter.\n"
        "  - ENCUT: Energy cutoff (from INCAR).\n"
        "  - KSPACING: K-point spacing for automatic mesh.\n"
        "  - VOLUME: Final crystal volume.\n"
        "  - POTIM: Time step for ionic motion.\n"
        "  - AMIX: Mixing parameter for electronic density.\n"
        "  - BMIX: Mixing parameter for charge density.\n"
        "  - EDIFF: Electronic convergence criterion.\n"
        "  - EDIFFG: Force convergence criterion.\n"
        "  - Elapsed time (sec): Total simulation time (from OUTCAR).\n"
        "  - Scaling: Scaling factor from the second line of CONTCAR.\n"
        "  - a1: primitive basis a_1\n"
        "  - a2: primitive basis a_2\n"
        "  - a3: primitive basis a_3\n"
        "Required Files:\n"
        "  - vasprun.xml\n"
        "  - KPOINTS\n"
        "  - OUTCAR (optional, for elapsed time)\n\n"
        "Example Usage:\n"
        "  identify_parameters()  # Use current directory\n"
        "  identify_parameters('/path/to/vasp_project')  # Specify a custom directory\n"
        "  identify_parameters('help')  # Display this usage guide\n"
    )

    if directory.lower() in ["help"]:
        print(help_info)
        return None

    vasprun_path = os.path.join(directory, "vasprun.xml")
    kpoints_path = os.path.join(directory, "KPOINTS")
    outcar_path = os.path.join(directory, "OUTCAR")
    contcar_path = os.path.join(directory, "CONTCAR")

    # Check file existence
    if not os.path.exists(vasprun_path) or not os.path.exists(kpoints_path) or not os.path.exists(contcar_path):
        print(f"Required files not found in {directory}. Skipping this directory.")
        return None

    try:
        # Initialize the parameters dictionary
        parameters = {
            "total atom count": None,
            "total energy": None,
            "fermi energy": None,
            "total kpoints": None,
            "calculated kpoints": None,
            "kpoints mesh": (None, None, None),
            "lattice constant": None,
            "symmetry precision (SYMPREC)": None,
            "energy cutoff (ENCUT)": None,
            "k-point spacing (KSPACING)": None,
            "volume": None,
            "time step (POTIM)": None,
            "mixing parameter (AMIX)": None,
            "mixing parameter (BMIX)": None,
            "electronic convergence (EDIFF)": None,
            "force convergence (EDIFFG)": None,
            "elapsed time (sec)": None,
            "Scaling": None,
            "a1": None,
            "a2": None,
            "a3": None,
        }

        # Extract Scaling factor and primitive bases from CONTCAR (second line)
        try:
            with open(contcar_path, "r", encoding="utf-8") as contcar_file:
                lines = contcar_file.readlines()
                scaling = float(lines[1].strip())
                parameters["Scaling"] = scaling

                a1_vector = [float(val) for val in lines[2].split()]
                a2_vector = [float(val) for val in lines[3].split()]
                a3_vector = [float(val) for val in lines[4].split()]

                a1_length = scaling * vector_length(a1_vector)
                a2_length = scaling * vector_length(a2_vector)
                a3_length = scaling * vector_length(a3_vector)

                parameters["a1"] = a1_length
                parameters["a2"] = a2_length
                parameters["a3"] = a3_length

        except (IndexError, ValueError) as e:
            print(f"Error reading Scaling from CONTCAR in {directory}: {e}")

        # Parse XML data from vasprun.xml
        tree = ET.parse(vasprun_path)
        root = tree.getroot()

        # Extract total atom count
        atom_count_tag = root.find(".//atominfo/atoms")
        if atom_count_tag is not None:
            parameters["total atom count"] = int(atom_count_tag.text)

        # Extract total energy
        energy_tag = root.find(".//calculation/energy/i[@name='e_fr_energy']")
        if energy_tag is not None:
            parameters["total energy"] = float(energy_tag.text)

        # Extract Fermi energy
        fermi_energy_tag = root.find(".//dos/i[@name='efermi']")
        if fermi_energy_tag is not None:
            parameters["fermi energy"] = float(fermi_energy_tag.text)

        # Extract lattice constant (using 'a' vector from final structure)
        basis_vectors = root.findall(".//calculation/structure/crystal/varray[@name='basis']")[-1]
        a_vector = basis_vectors[0].text.split()
        parameters["lattice constant"] = (float(a_vector[0])**2 + float(a_vector[1])**2 + float(a_vector[2])**2) ** 0.5

        # Extract SYMPREC for symmetry precision
        symprec_tag = root.find(".//parameters/separator/i[@name='SYMPREC']")
        if symprec_tag is not None:
            parameters["symmetry precision (SYMPREC)"] = float(symprec_tag.text)

        # Extract INCAR parameters from the <incar> section
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

        # Extract volume
        volume_tag = root.find(".//structure[@name='finalpos']/crystal/volume")
        if volume_tag is not None:
            parameters["volume"] = float(volume_tag.text)

        # Extract k-points from KPOINTS file
        with open(kpoints_path, "r", encoding="utf-8") as kpoints_file:
            lines = kpoints_file.readlines()
            for index, line in enumerate(lines):
                if any(keyword in line.lower() for keyword in ["gamma", "monkhorst", "monkhorst-pack", "explicit", "line-mode"]):
                    kpoints_index = index + 1
                    break
            else:
                print(f"Kpoints type keyword not found in KPOINTS file at {directory}.")
                return None

            kpoints_values = lines[kpoints_index].split()
            x_kpoints, y_kpoints, z_kpoints = int(kpoints_values[0]), int(kpoints_values[1]), int(kpoints_values[2])
            parameters["total kpoints"] = x_kpoints * y_kpoints * z_kpoints
            parameters["kpoints mesh"] = (x_kpoints, y_kpoints, z_kpoints)

        # Extract calculated kpoints (reduced due to symmetry)
        cal_kpoints_tag = root.find(".//kpoints/varray[@name='kpointlist']")
        if cal_kpoints_tag is not None:
            parameters["calculated kpoints"] = len(cal_kpoints_tag.findall("v"))

        # Extract elapsed time from OUTCAR file
        if os.path.isfile(outcar_path):
            with open(outcar_path, "r", encoding="utf-8") as outcar_file:
                for line in outcar_file:
                    if "Elapsed time (sec):" in line:
                        parameters["elapsed time (sec)"] = float(line.split(":")[-1].strip())
                        break

        return parameters

    except (ET.ParseError, ValueError, IndexError) as e:
        print(f"Error parsing files in {directory}: {e}")
        return None

def get_atoms_count(directory):
    """
    Extracts the total number of atoms from a VASP vasprun.xml file.

    Args:
    directory (str): The directory path that contains the VASP vasprun.xml file.

    Returns:
    int: The total number of atoms in the calculation.
    """
    # Construct the path to the vasprun.xml file and parse it
    xml_file = os.path.join(directory, "vasprun.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find the atominfo section and extract the total number of atoms
    atominfo_section = root.find(".//atominfo/atoms")
    if atominfo_section is not None:
        return int(atominfo_section.text)
    else:
        print("Atominfo section not found in the XML file.")
        return None

def get_elements(directory_path):
    ## Construct the full path to the vasprun.xml file
    file_path = os.path.join(directory_path, "vasprun.xml")
    # Check if the vasprun.xml file exists in the given directory
    if not os.path.isfile(file_path):
        print(f"Error: The file vasprun.xml does not exist in the directory {directory_path}.")
        return

    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Initialize an empty dictionary to store the element-ion pairs
    element_ions = {}

    # Use XPath to locate the <rc><c> tags under the path "atominfo/array[@name="atoms"]/set"
    for i, atom in enumerate(root.findall(".//atominfo//array[@name='atoms']//set//rc"), start=1):
        element = atom.find("c").text.strip()
        if element in element_ions:
            # Update the maximum index for the element
            element_ions[element][1] = i
        else:
            # Add a new entry for the element, with the minimum and maximum index being the same
            element_ions[element] = [i, i]

    # Convert the lists to tuples
    for element in element_ions:
        element_ions[element] = tuple(element_ions[element])

    return element_ions

import os
import xml.etree.ElementTree as ET

def get_elements_coordinates(directory_path):
    # Construct the full path to the vasprun.xml file
    file_path = os.path.join(directory_path, "vasprun.xml")
    # Check if the vasprun.xml file exists in the given directory
    if not os.path.isfile(file_path):
        print(f"Error: The file vasprun.xml does not exist in the directory {directory_path}.")
        return

    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Initialize a list to store detailed ion data
    ions_data = []

    # Use XPath to locate the <rc><c> tags under the path "atominfo/array[@name='atoms']/set"
    element_list = root.findall(".//atominfo//array[@name='atoms']//set//rc")
    coordinate_sets = root.findall(".//structure[@name='finalpos']//varray[@name='positions']//v")

    if not element_list or not coordinate_sets:
        print("Error: Unable to find atomic elements or coordinates in vasprun.xml.")
        return

    # Loop through elements and their coordinates
    for idx, (atom, coord) in enumerate(zip(element_list, coordinate_sets), start=1):
        element = atom.find("c").text.strip()  # Extract the element type
        coord_values = tuple(map(float, coord.text.split()))  # Extract the coordinates
        ions_data.append({
            "index": idx,
            "element": element,
            "coordinates": coord_values
        })

    return ions_data

def check_range_type(data: Union[Tuple, Tuple[Tuple, ...], int, float]) -> str:
    """
    Determine the type of the provided range data.
    Args:
        data (Union[Tuple, Tuple[Tuple, ...], int, float]): The range data to check.
        It can be a simple number, a tuple with one or more elements, or nested tuples.
    Returns:
        str: A description of whether the input is a "Simple end",
             "Double ends", "Simple range", "Simple range with a rate", or "Double ranges".
    """
    # Check if the input is a single number (not a tuple)
    if isinstance(data, (int, float)):
        return "Simple end"
    # Check if the input is a tuple with one element
    elif isinstance(data, tuple) and len(data) == 1:
        if isinstance(data[0], tuple):
            return "Simple range"
        else:
            return "Simple end"
    # Check if the input is a tuple with two elements
    elif isinstance(data, tuple) and len(data) == 2:
        if isinstance(data[0], tuple) and isinstance(data[1], tuple):
            return "Double ranges"
        elif isinstance(data[0], tuple) and isinstance(data[1], (int, float)):
            return "Simple range with a rate"
        elif isinstance(data[0], (int, float)) and isinstance(data[1], (int, float)):
            return "Double ends"
    return "Unknown type"

def process_boundary(boundary, default=(None, None)):
    # Enhanced to handle single values as well as tuples/lists
    # If boundary is None or empty, return the default
    if not boundary:
        return default
    # If boundary is a single value (not a container), treat it as the end value
    if isinstance(boundary, (int, float)):
        return (None, boundary)
    # If boundary is a container with a single item, unpack it
    if isinstance(boundary, (list, tuple)) and len(boundary) == 1:
        return (None, boundary[0])
    # If boundary is a container with two items, return them as start and end
    elif isinstance(boundary, (list, tuple)) and len(boundary) == 2:
        return (boundary[0], boundary[1])
    # In case boundary doesn't match any expected pattern, return default
    else:
        return default

def process_boundaries_rescaling(boundary):
    boundary_type = check_range_type(boundary)
    scale_flag = False
    source_start, source_end, scaled_start, scaled_end = None, None, None, None

    if boundary_type == "Simple end":
        source_range = (None, boundary)
        source_start, source_end = process_boundary(source_range)
    elif boundary_type == "Simple range":
        source_range = boundary[0]
        source_start, source_end = process_boundary(source_range)
    elif boundary_type == "Double ends":
        scale_flag = True
        source_range = (None, boundary[0])
        scaled_range = (None, boundary[1])
        source_start, source_end = process_boundary(source_range)
        scaled_start, scaled_end = process_boundary(scaled_range)
    elif boundary_type == "Simple range with a rate":
        scale_flag = True
        source_range = boundary[0]
        scaled_range = tuple(bounds * boundary[1] for bounds in boundary[0])
        source_start, source_end = process_boundary(source_range)
        scaled_start, scaled_end = process_boundary(scaled_range)
    elif boundary_type == "Double ranges":
        scale_flag = True
        source_range = boundary[0]
        scaled_range = boundary[1]
        source_start, source_end = process_boundary(source_range)
        scaled_start, scaled_end = process_boundary(scaled_range)
    return scale_flag, source_start, source_end, scaled_start, scaled_end
