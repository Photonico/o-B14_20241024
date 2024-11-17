#### Bandstructure
# pylint: disable = C0103, C0114, C0116, C0301, C0302, C0321, R0913, R0914, R0915, W0612, W0105

import xml.etree.ElementTree as ET
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from vmatplot.algorithms import transpose_matrix
from vmatplot.commons import extract_fermi

def extract_bandgap_outcar(directory="."):
    """
    Extract the bandgap, LUMO, and HOMO values from the OUTCAR file and return as a dictionary.

    Parameters:
        directory (str): Path to the directory containing the VASP output files (default is current directory).

    Returns:
        dict: A dictionary containing:
            - "bandgap": The bandgap value in eV.
            - "HOMO index": The index of the HOMO band.
            - "HOMO energy": The energy of the HOMO band in eV.
            - "LUMO index": The index of the LUMO band.
            - "LUMO energy": The energy of the LUMO band in eV.
        str: Error message if required data is missing or cannot be processed.
    """
    outcar_path = os.path.join(directory, "OUTCAR")

    # Check if OUTCAR exists
    if not os.path.exists(outcar_path):
        return "Error: OUTCAR not found in the specified directory."

    try:
        with open(outcar_path, "r") as file:
            lines = file.readlines()

        # Extract NELECT and NKPTS
        nelect = None
        nkpts = None
        for line in lines:
            if "NELECT" in line:
                nelect = float(line.split()[2]) / 2  # HOMO band index
            elif "NKPTS" in line:
                nkpts = int(line.split()[3])  # Total k-points

        if nelect is None or nkpts is None:
            return "Error: Could not extract NELECT or NKPTS from OUTCAR."

        # Calculate HOMO and LUMO band indices
        homo_band = int(nelect)
        lumo_band = homo_band + 1

        # Extract HOMO and LUMO energies
        homo_energies = []
        lumo_energies = []
        for line in lines:
            if f"{homo_band:5d}" in line:  # Strictly match HOMO band
                try:
                    homo_energies.append(float(line.split()[1]))
                except (ValueError, IndexError):
                    pass
            elif f"{lumo_band:5d}" in line:  # Strictly match LUMO band
                try:
                    lumo_energies.append(float(line.split()[1]))
                except (ValueError, IndexError):
                    pass

        if not homo_energies or not lumo_energies:
            return "Error: Could not extract HOMO or LUMO energies from OUTCAR."

        # Sort HOMO energies and take the last (maximum)
        homo_energy = sorted(homo_energies)[-1]

        # Sort LUMO energies and take the first (minimum)
        lumo_energy = sorted(lumo_energies)[0]

        # Calculate bandgap
        bandgap = lumo_energy - homo_energy

        return {
            "bandgap": bandgap,
            "HOMO index": homo_band,
            "HOMO energy": homo_energy,
            "HOMO": homo_energy,
            "LUMO index": lumo_band,
            "LUMO energy": lumo_energy,
            "LUMO": lumo_energy,
        }

    except Exception as e:
        return f"Error: {str(e)}"

def extract_bandgap_OUTCAR(*args):
    return extract_bandgap_outcar(*args)

def extract_high_sym(directory):
    """
    Extracts the high symmetry lines from the KPOINTS file in a VASP calculation directory.
    """
    # Open and read the KPOINTS file
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    if os.path.exists(kpoints_opt_path):
        kpoints_file = kpoints_opt_path
    elif os.path.exists(kpoints_file_path):
        kpoints_file = kpoints_file_path
    with open(kpoints_file, "r", encoding="utf-8") as file:
        KPOINTS = file.readlines()
    # Check if the KPOINTS file is in line-mode
    if KPOINTS[2][0] not in ("l", "L"):
        raise ValueError(f"Expected 'L' on the third line of KPOINTS file, got: {KPOINTS[2]}")
    # Initialize a set to store unique high symmetry points
    high_symmetry_points = set()
    # Read the high symmetry points from the KPOINTS file
    for i in range(4, len(KPOINTS)):
        tokens = KPOINTS[i].strip().split()
        if tokens and tokens[-1].isalpha():
            high_symmetry_points.add(tokens[-1])
    # The set of high symmetry points
    sets = high_symmetry_points
    return list(sets)

def extract_highsym(directory):
    """
    Extracts the list of k-point coordinates from the vasprun.xml file of a VASP calculation.

    Args:
    directory (str): The directory path that contains the VASP vasprun.xml file.
    
    Returns:
    list: A list of k-point coordinates where each k-point is represented as a list of its coordinates.
    
    The function parses the vasprun.xml file to find the k-point coordinates used in the calculation.
    It looks for the 'varray' XML element with the attribute name set to 'kpointlist', which contains
    the k-point data. Each k-point is then extracted and appended to a list, which is returned.
    """
    # Construct the full path to the vasprun.xml file
    xml_file = os.path.join(directory, "vasprun.xml")
    # Parse the XML file
    tree = ET.parse(xml_file)
    # Get the root of the XML tree
    root = tree.getroot()
    # Initialize a list to store the k-point coordinates
    kpoints = []
    # Find all 'v' elements within 'varray' elements that have a 'name' attribute equal to 'kpointlist'
    # These elements contain the k-point coordinates
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        for kpoint in root.findall("./calculation/eigenvalues_kpoints_opt[@comment='kpoints_opt']/kpoints/varray[@name='kpointlist']/v"):
            # Split the text content of the 'v' element to get the individual coordinate strings
            # Convert each coordinate string to a float and create a list of coordinates
            coords = [float(x) for x in kpoint.text.split()]
            # Append the list of coordinates to the kpoints list
            kpoints.append(coords)
    # GGA-PBE algorithms
    elif os.path.exists(kpoints_file_path):
        for kpoint in root.findall(".//varray[@name='kpointlist']/v"):
            # Split the text content of the 'v' element to get the individual coordinate strings
            # Convert each coordinate string to a float and create a list of coordinates
            coords = [float(x) for x in kpoint.text.split()]
            # Append the list of coordinates to the kpoints list
            kpoints.append(coords)
    # Return the list of k-point coordinates
    return kpoints

def extract_high_symlines(directory):
    """
    Extracts the high symmetry lines from the KPOINTS file in a VASP calculation directory.

    Args:
    directory (str): The directory path that contains the VASP KPOINTS file.
    
    Returns:
    tuple: A tuple containing the kpoints format, number of high symmetry lines, 
           a set of high symmetry points, and a list of limit points for each line.
    
    This function opens the KPOINTS file and reads the high symmetry lines specified within it.
    It checks for the expected format and extracts the high symmetry points and their limits.
    """
    # Open and read the KPOINTS file
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    if os.path.exists(kpoints_opt_path):
        kpoints_file = kpoints_opt_path
    elif os.path.exists(kpoints_file_path):
        kpoints_file = kpoints_file_path

    with open(kpoints_file, "r", encoding="utf-8") as file:
        KPOINTS = file.readlines()
    # Check if the KPOINTS file is in line-mode
    if KPOINTS[2][0] not in ("l", "L"):
        raise ValueError(f"Expected 'L' on the third line of KPOINTS file, got: {KPOINTS[2]}")
    # Determine the format of the kpoints (cartesian or reciprocal)
    kpoints_format = "cartesian" if KPOINTS[3][0] in ["c", "C"] else "reciprocal"
    # Initialize a set to store unique high symmetry points
    high_symmetry_points = set()
    # Read the high symmetry points from the KPOINTS file
    for i in range(4, len(KPOINTS)):
        tokens = KPOINTS[i].strip().split()
        if tokens and tokens[-1].isalpha():
            high_symmetry_points.add(tokens[-1])
    # The number of unique high symmetry lines
    lines = len(high_symmetry_points)
    # The set of high symmetry points
    sets = high_symmetry_points
    # Extract non-empty lines from the KPOINTS file
    non_empty_lines = [line.split() for line in KPOINTS[4:] if line.strip()]
    # Extract the start and end points for each high symmetry line
    limits = []
    for i in range(0, len(non_empty_lines), 2):
        start = non_empty_lines[i]
        end = non_empty_lines[i+1]
        limits.append([start, end])
    # Return the kpoints format, number of lines, set of high symmetry points, and their limits
    return kpoints_format, lines, list(sets), limits

def extract_kpoints_eigenval(directory):
    """
    Extracts k-point coordinates from a VASP EIGENVAL file.

    Args:
    directory (str): The directory path that contains the VASP EIGENVAL file.

    Returns:
    numpy.ndarray: An array of k-point coordinates.

    The function reads the EIGENVAL file, which contains the eigenvalues for each k-point and band.
    It extracts the k-point coordinates from this file and returns them as a NumPy array.
    """
    # Open the EIGENVAL file
    with open(os.path.join(directory, "EIGENVAL"), "r", encoding="utf-8") as file:
        lines = file.readlines()
    # Initialize the list for k-points
    kpoints_list = []
    # Get the total number of bands and k-points from the sixth line of the file
    try:
        num_bands = int(lines[5].split()[2])
        num_kpoints = int(lines[5].split()[1])
    except IndexError as exc:
        # If the expected format is not found, raise an error
        raise ValueError("The EIGENVAL file does not have the expected format.") from exc
    # Calculate the number of lines in each k-point block (including the k-point line itself)
    block_size = num_bands + 1
    # Iterate over the EIGENVAL file to extract k-point coordinates
    # The k-point blocks start from line 7 (index 6) and are spaced by the block size
    for i in range(6, 6 + num_kpoints * block_size, block_size):
        # Extract the k-point coordinates from the first line of each block
        kpoint_line = lines[i].strip().split()
        # Check if there are enough elements in the line to represent a k-point
        if len(kpoint_line) < 4:
            continue  # Skip lines that don't have enough elements
        # Take the first three values as k-point coordinates (ignoring the weight)
        kpoint_coords = [float(kpoint_line[j]) for j in range(3)]
        kpoints_list.append(kpoint_coords)
    # Convert the k-point list to a NumPy array for efficiency
    kpoints_array = np.array(kpoints_list)
    return kpoints_array

def extract_weight(directory):
    xml_file = os.path.join(directory, "vasprun.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")

    weight_list = []
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        for weight in root.findall(".//eigenvalues_kpoints_opt[@comment='kpoints_opt']/kpoints/varray[@name='weights']/v"): # <varray name="weights" >
            weight_list.append(float(weight.text))
    # GGA-PBE algorithms
    elif os.path.exists(kpoints_file_path):
        for weight in root.findall(".//varray[@name='weights']/v"): # <varray name="weights" >
            weight_list.append(float(weight.text))
    return weight_list

def extract_kpoints_count(directory):
    xml_file = os.path.join(directory, "vasprun.xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    # Find the kpoints varray
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        kpoints_varray = root.find("./calculation/eigenvalues_kpoints_opt[@comment='kpoints_opt']/kpoints/varray[@name='kpointlist']")
    # GGA-PBE algorithms
    elif os.path.exists(kpoints_file_path):
        kpoints_varray = root.find(".//kpoints/varray[@name='kpointlist']")
    # Check if the varray exists
    if kpoints_varray is not None:
        # The number of kpoints is the number of <v> tags within the varray
        num_kpoints = len(kpoints_varray.findall("./v"))
        return num_kpoints
    else:
        print("The kpointlist section does not exist in the provided XML file.")
        return None

def extract_bands_count(directory):
    eigen_lines = extract_eigenvalues_bands_nonpolarized(directory)
    return len(eigen_lines)

def kpoints_coordinate(directory):
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    if os.path.exists(kpoints_opt_path):
        kpoints_file = kpoints_opt_path
    elif os.path.exists(kpoints_file_path):
        kpoints_file = kpoints_file_path

    high_symmetry_points = {}
    with open(kpoints_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
        # Assume high symmetry points start from the fifth line in the KPOINTS file
        for line in lines[4:]:      # High symmetry points coordinates usually start from line 5
            if line.strip():        # Ignore empty lines
                parts = line.split()
                if len(parts) == 4: # A line with coordinates should have four parts
                    # Assume the coordinates and label are separated by spaces, with the label being the last part
                    coords = tuple(map(float, parts[:3]))   # Convert the first three parts to float coordinates
                    label = parts[3]                        # The last part is the label of the high symmetry point
                    high_symmetry_points[label] = coords
    return high_symmetry_points

def kpoints_index(directory):
    # Retrieve the coordinates of the high symmetry points
    high_symmetry_points = kpoints_coordinate(directory)
    # Retrieve the list of kpoints
    kpoints_list = extract_highsym(directory)
    # Initialize a dictionary to store the indices of the high symmetry points
    high_symmetry_indices = {}
    # For each high symmetry point, find the closest kpoint
    for label, coord in high_symmetry_points.items():
        # Initialize a minimum distance to a very large number so any actual distance will be smaller
        min_distance = float("inf")
        min_index = None
        # Iterate over the kpoint list to find the kpoint closest to the current high symmetry point coordinates
        for index, kpoint in enumerate(kpoints_list):
            # Calculate the Euclidean distance
            distance = sum((c - k) ** 2 for c, k in zip(coord, kpoint)) ** 0.5
            # If this distance is the smallest so far, update the minimum distance and index
            if distance < min_distance:
                min_distance = distance
                min_index = index
        # Store the index of the closest kpoint
        high_symmetry_indices[label] = min_index
    return high_symmetry_indices

def kpoints_path(directory):
    """
    This function calculates the path distances for high symmetry points in the Brillouin zone.
    
    Args:
    directory (str): The directory path that contains the VASP output files.
    
    Returns:
    dict: A dictionary mapping high symmetry points to their cumulative path distances.
    
    The function works by first finding the indices of the high symmetry points in the k-point list.
    Then, it calculates the cumulative path distance for each k-point. Finally, it creates a dictionary
    that maps each high symmetry point label to its corresponding path distance.
    """
    # Get the indices of high symmetry points in the k-point list
    high_symmetry_indices = kpoints_index(directory)
    # Calculate the cumulative path distances for the k-points
    path = extract_kpath(directory)
    # Initialize a dictionary to store the high symmetry points and their path distances
    high_symmetry_paths = {}
    # Iterate over the high symmetry points and their indices
    for label, index in high_symmetry_indices.items():
        # Map the high symmetry point label to its path distance
        high_symmetry_paths[label] = path[index]
    # Return the dictionary of high symmetry points and their path distances
    return high_symmetry_paths

def high_symmetry_coordinates(directory):
    """
    This function extracts the coordinates of high symmetry points from the KPOINTS file.

    Args:
    directory (str): The directory path that contains the VASP KPOINTS file.
    
    Returns:
    list: A list of coordinates for the high symmetry points in the Brillouin zone.
    """
    # Retrieve the coordinates of the high symmetry points
    high_symmetry_points = kpoints_coordinate(directory)
    # Extract the coordinates from the dictionary and store them in a list
    coordinates_list = list(high_symmetry_points.values())
    return coordinates_list

def high_symmetry_path(directory):
    """
    This function extracts the x-axis values (cumulative path distances) for the high symmetry points
    in a bandstructure plot.

    Args:
    directory (str): The directory path that contains the VASP output files.
    
    Returns:
    list: A list of x-axis values for the high symmetry points in the bandstructure plot.
    """
    # Get the indices of high symmetry points in the k-point list
    high_symmetry_indices = kpoints_index(directory)
    # Calculate the cumulative path distances for the k-points
    path = extract_kpath(directory)
    # Initialize a list to store the x-axis values for the high symmetry points
    high_sym_path = []
    # Iterate over the high symmetry points and their indices
    for index in high_symmetry_indices.values():
        # Append the corresponding path distance to the list
        high_sym_path.append(path[index])
    # Return the list of x-axis values
    return high_sym_path

def clean_kpoints(kpoints_list, tol=1e-10):
    kpoints_list[np.isclose(kpoints_list, 0, atol=tol)] = 0
    return kpoints_list

def extract_eigenvalues_kpoints(directory, spin_label):
    """
    Extracts the eigenvalues for each k-point from a VASP vasprun.xml file considering spin polarization.

    Args:
        directory (str): The directory path that contains the VASP vasprun.xml file.
        spin_label (str): The spin channel label ('spin1' or 'spin2').

    Returns:
        list of lists: A matrix where each sublist contains the eigenvalues for a specific k-point and spin channel.

    This function parses the vasprun.xml file to extract the electronic energy levels (eigenvalues)
    at each k-point in the reciprocal lattice for the material being studied, considering the specified spin channel.
    These eigenvalues are crucial for analyzing the material's electronic structure, such as plotting band structures.
    """
    xml_file = os.path.join(directory, "vasprun.xml")
    kpoints_file_path = os.path.join(directory, "KPOINTS")
    kpoints_opt_path = os.path.join(directory, "KPOINTS_OPT")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # Initialize a list to store the eigenvalues for each k-point
    eigenvalues_matrix = []
    # Find the eigenvalues section in the XML tree
    # HSE06 algorithms
    if os.path.exists(kpoints_opt_path):
        eigenvalues_section = root.find("./calculation/eigenvalues_kpoints_opt[@comment='kpoints_opt']/eigenvalues")
    # GGA-PBE algorithms
    elif os.path.exists(kpoints_file_path):
        eigenvalues_section = root.find("./calculation/eigenvalues")
    if eigenvalues_section is not None:
        # Find all k-point <set> elements within the eigenvalues section
        # kpoint_sets = eigenvalues_section.findall(".//set/set/set")
        kpoint_sets = eigenvalues_section.findall(f".//set/set[@comment='{spin_label}']/set")
        if kpoint_sets:
            # Iterate over each k-point set to extract eigenvalues
            for kpoint_set in kpoint_sets:
                kpoint_eigenvalues = []
                # Iterate over each band's eigenvalue within the current k-point set
                for r in kpoint_set.findall("./r"):
                    # The energy eigenvalue is the first number in the <r> tag's text
                    energy = float(r.text.split()[0])
                    kpoint_eigenvalues.append(energy)
                # Append the list of eigenvalues for this k-point to the matrix
                eigenvalues_matrix.append(kpoint_eigenvalues)
        else:
            # Handle the case where no k-point <set> elements are found
            print("No k-point <set> elements found in the eigenvalues section.")
    else:
        # Handle the case where the eigenvalues section is missing
        print("Eigenvalues section not found in the XML file.")
    # Return the matrix of eigenvalues
    return eigenvalues_matrix

def extract_eigenvalues_kpoints_nonpolarized(directory):
    return extract_eigenvalues_kpoints(directory, "spin 1")

def extract_eigenvalues_kpoints_spinUp(directory):
    return extract_eigenvalues_kpoints(directory, "spin 1")

def extract_eigenvalues_kpoints_spinDown(directory):
    return extract_eigenvalues_kpoints(directory, "spin 2")

def extract_eigenvalues_bands(directory, spin_label):
    """
    Extracts and transposes the eigenvalues for each band from a VASP calculation.

    This function is designed to work with data from VASP (Vienna Ab initio Simulation Package) calculations. 
    It extracts the eigenvalues associated with each k-point for a given spin orientation (either 'spin1' or 'spin2'), 
    and then transposes the matrix so that each row represents a band and each column represents a k-point.

    Args:
        directory (str): The directory path that contains the VASP vasprun.xml file.
        spin_label (str): The spin label ('spin 1' or 'spin 2') for which the eigenvalues are to be extracted. 
                          'spin 1' typically refers to spin-up and 'spin 2' to spin-down in spin-polarized calculations.

    Returns:
        list of lists: A transposed matrix of eigenvalues where each row represents a band and each column represents a k-point. 
                       This format is useful for plotting band structures and analyzing the electronic properties of materials.

    Example:
        # Extract eigenvalues for 'spin 1' (spin-up) orientation
        directory = "/path/to/vasp/output"
        spin_label = "spin1"
        bands_matrix = extract_eigenvalues_bands(directory, spin_label)
        # 'bands_matrix' now contains the eigenvalues with bands as rows and k-points as columns
    """
    # Extract the eigenvalues for each k-point
    eigenvalues_matrix = extract_eigenvalues_kpoints(directory, spin_label)
    # Transpose the matrix so that bands are rows and k-points are columns
    transposed_eigenvalues_matrix = transpose_matrix(eigenvalues_matrix)
    # Return the transposed matrix of eigenvalues
    return transposed_eigenvalues_matrix

def extract_eigenvalues_bands_nonpolarized(directory):
    return extract_eigenvalues_bands(directory, "spin 1")

def extract_eigenvalues_bands_spinUp(directory):
    return extract_eigenvalues_bands(directory, "spin 1")

def extract_eigenvalues_bands_spinDown(directory):
    return extract_eigenvalues_bands(directory, "spin 2")

global_tolerance = 0.8

def extract_eigenvalues_conductionBands(directory, spin_label):
    eigenvalues_matrix = extract_eigenvalues_bands(directory, spin_label)
    conduction_bands = []
    TOLERANCE = global_tolerance
    current_LUMO = extract_bandgap_outcar(directory)[2]
    current_HOMO = extract_bandgap_outcar(directory)[1]
    for eigenvalues_bands in eigenvalues_matrix:
        if np.min(eigenvalues_bands) >= current_LUMO-TOLERANCE:
            conduction_bands.append(eigenvalues_bands)
    return conduction_bands

def extract_eigenvalues_valenceBands(directory, spin_label):
    eigenvalues_matrix = extract_eigenvalues_bands(directory, spin_label)
    valence_bands = []
    TOLERANCE = global_tolerance
    current_LUMO = extract_bandgap_outcar(directory)[2]
    current_HOMO = extract_bandgap_outcar(directory)[1]
    for eigenvalues_bands in eigenvalues_matrix:
        if np.max(eigenvalues_bands) <= current_HOMO+TOLERANCE:
            valence_bands.append(eigenvalues_bands)
    return valence_bands

def extract_eigenvalues_conductionBands_nonpolarized(directory):
    return extract_eigenvalues_conductionBands(directory, "spin 1")

def extract_eigenvalues_valenceBands_nonpolarized(directory):
    return extract_eigenvalues_valenceBands(directory, "spin 1")

def extract_eigenvalues_conductionBands_spinUp(directory):
    return extract_eigenvalues_conductionBands(directory, "spin 1")

def extract_eigenvalues_valenceBands_spinUp(directory):
    return extract_eigenvalues_valenceBands(directory, "spin 1")

def extract_eigenvalues_conductionBands_spinDown(directory):
    return extract_eigenvalues_conductionBands(directory, "spin 2")

def extract_eigenvalues_valenceBands_spinDown(directory):
    return extract_eigenvalues_valenceBands(directory, "spin 2")

def extract_high_sym_intersections(directory, spin_label):
    """
    Extracts the eigenvalues at high symmetry points for each band in a VASP bandstructure calculation,
    using the path value (x) in the bandstructure plot and the corresponding eigenvalue (y).

    Args:
        directory (str): The directory path that contains the VASP output files.
        spin_label (str): The spin label ('spin1' or 'spin2') for which the eigenvalues are to be extracted.

    Returns:
        dict: A dictionary where the keys are high symmetry points (e.g., 'Gamma', 'K', 'M'), and the values 
              are lists of tuples, where each tuple represents (path, eigenvalue) coordinates of intersection 
              points at that high symmetry point. 'path' is the x-coordinate, and 'eigenvalue' is the y-coordinate.
    """
    # Extract the eigenvalues (bands as rows, k-points as columns)
    eigenvalues_bands = extract_eigenvalues_bands(directory, spin_label)

    # Get the path distances for each k-point along the bandstructure path
    path = extract_kpath(directory)

    # Extract the high symmetry points and their positions in the bandstructure
    high_symmetry_indices = kpoints_index(directory)  # Get the indices of high symmetry points
    high_symmetry_labels = extract_high_sym(directory)  # Get high symmetry labels

    # Initialize a dictionary to store the intersection points
    intersections = {}

    # Iterate over each high symmetry point and its corresponding index
    for label, index in high_symmetry_indices.items():
        # Initialize a list to store (path, eigenvalue) coordinates at this high symmetry point
        intersections[label] = []
        for band in eigenvalues_bands:
            # x is the path value (cumulative distance along the bandstructure path)
            x = path[index]  # The x-axis value is the path (cumulative distance) at the high symmetry point
            # y is the eigenvalue at that point
            y = band[index]  # The y-axis value is the eigenvalue at that k-point
            # Append the (path, eigenvalue) coordinates to the list for this high symmetry point
            intersections[label].append((x, y))

    return intersections

def extract_high_sym_intersections_with_fermi(directory, spin_label):
    """
    Extracts the eigenvalues at high symmetry points for each band in a VASP bandstructure calculation,
    using the path value (x) in the bandstructure plot and the corresponding eigenvalue (y) minus the Fermi level.

    Args:
        directory (str): The directory path that contains the VASP output files.
        spin_label (str): The spin label ('spin1' or 'spin2') for which the eigenvalues are to be extracted.

    Returns:
        dict: A dictionary where the keys are high symmetry points (e.g., 'Gamma', 'K', 'M'), and the values 
              are lists of tuples, where each tuple represents (path, eigenvalue - Fermi) coordinates of intersection 
              points at that high symmetry point. 'path' is the x-coordinate, and 'eigenvalue' is the y-coordinate minus Fermi.
    """
    # Extract the eigenvalues (bands as rows, k-points as columns)
    eigenvalues_bands = extract_eigenvalues_bands(directory, spin_label)

    # Get the Fermi energy
    fermi_energy = extract_fermi(directory)

    # Get the path distances for each k-point along the bandstructure path
    path = extract_kpath(directory)

    # Extract the high symmetry points and their positions in the bandstructure
    high_symmetry_indices = kpoints_index(directory)    # Get the indices of high symmetry points
    high_symmetry_labels = extract_high_sym(directory)  # Get high symmetry labels

    # Initialize a dictionary to store the intersection points
    intersections = {}

    # Iterate over each high symmetry point and its corresponding index
    for label, index in high_symmetry_indices.items():
        # Initialize a list to store (path, eigenvalue - Fermi) coordinates at this high symmetry point
        intersections[label] = []
        for band in eigenvalues_bands:
            # x is the path value (cumulative distance along the bandstructure path)
            x = path[index]  # The x-axis value is the path (cumulative distance) at the high symmetry point
            # y is the eigenvalue at that point, minus the Fermi energy
            y = band[index] - fermi_energy  # The y-axis value is the eigenvalue at that k-point minus the Fermi level
            # Append the (path, eigenvalue - Fermi) coordinates to the list for this high symmetry point
            intersections[label].append((x, y))

    return intersections

def extract_high_sym_valence_intersections(directory, spin_label):
    """
    Extracts the valence band eigenvalues (y < 0) at high symmetry points for each band in a VASP 
    bandstructure calculation, using the path value (x) in the bandstructure plot and the corresponding 
    eigenvalue (y) minus the Fermi level.

    Args:
        directory (str): The directory path that contains the VASP output files.
        spin_label (str): The spin label ('spin1' or 'spin2') for which the eigenvalues are to be extracted.

    Returns:
        dict: A dictionary where the keys are high symmetry points (e.g., 'Gamma', 'K', 'M'), and the values 
              are lists of tuples, where each tuple represents (path, eigenvalue - Fermi) coordinates of valence 
              band intersection points (where eigenvalue - Fermi < 0).
    """
    intersections_with_fermi = extract_high_sym_intersections_with_fermi(directory, spin_label)
    valence_intersections = {}

    # Filter for valence band intersections (y < 0)
    for label, points in intersections_with_fermi.items():
        valence_intersections[label] = [point for point in points if point[1] < 0]

    return valence_intersections

def extract_high_sym_conduction_intersections(directory, spin_label):
    """
    Extracts the conduction band eigenvalues (y > 0) at high symmetry points for each band in a VASP 
    bandstructure calculation, using the path value (x) in the bandstructure plot and the corresponding 
    eigenvalue (y) minus the Fermi level.

    Args:
        directory (str): The directory path that contains the VASP output files.
        spin_label (str): The spin label ('spin1' or 'spin2') for which the eigenvalues are to be extracted.

    Returns:
        dict: A dictionary where the keys are high symmetry points (e.g., 'Gamma', 'K', 'M'), and the values 
              are lists of tuples, where each tuple represents (path, eigenvalue - Fermi) coordinates of conduction 
              band intersection points (where eigenvalue - Fermi > 0).
    """
    intersections_with_fermi = extract_high_sym_intersections_with_fermi(directory, spin_label)
    conduction_intersections = {}

    # Filter for conduction band intersections (y > 0)
    for label, points in intersections_with_fermi.items():
        conduction_intersections[label] = [point for point in points if point[1] > 0]

    return conduction_intersections

def extract_high_sym_min_conduction_intersections(directory, spin_label):
    """
    Extracts the lowest conduction band eigenvalue (minimum y > 0) at high symmetry points for each band 
    in a VASP bandstructure calculation, using the path value (x) in the bandstructure plot and the corresponding 
    eigenvalue (y) minus the Fermi level.

    Args:
        directory (str): The directory path that contains the VASP output files.
        spin_label (str): The spin label ('spin1' or 'spin2') for which the eigenvalues are to be extracted.

    Returns:
        dict: A dictionary where the keys are high symmetry points (e.g., 'Gamma', 'K', 'M'), and the values 
              are tuples representing (path, min_eigenvalue - Fermi) for the lowest conduction band intersection 
              point (where min_eigenvalue - Fermi > 0).
    """
    intersections_with_fermi = extract_high_sym_conduction_intersections(directory, spin_label)
    min_conduction_intersections = {}

    # Find the minimum conduction band eigenvalue (y > 0) for each high symmetry point
    for label, points in intersections_with_fermi.items():
        if points:  # If there are conduction intersections
            min_point = min(points, key=lambda point: point[1])  # Get the point with the minimum y
            min_conduction_intersections[label] = min_point

    return min_conduction_intersections

def extract_high_sym_max_valence_intersections(directory, spin_label):
    """
    Extracts the highest valence band eigenvalue (maximum y < 0) at high symmetry points for each band 
    in a VASP bandstructure calculation, using the path value (x) in the bandstructure plot and the corresponding 
    eigenvalue (y) minus the Fermi level.

    Args:
        directory (str): The directory path that contains the VASP output files.
        spin_label (str): The spin label ('spin1' or 'spin2') for which the eigenvalues are to be extracted.

    Returns:
        dict: A dictionary where the keys are high symmetry points (e.g., 'Gamma', 'K', 'M'), and the values 
              are tuples representing (path, max_eigenvalue - Fermi) for the highest valence band intersection 
              point (where max_eigenvalue - Fermi < 0).
    """
    intersections_with_fermi = extract_high_sym_valence_intersections(directory, spin_label)
    max_valence_intersections = {}

    # Find the maximum valence band eigenvalue (y < 0) for each high symmetry point
    for label, points in intersections_with_fermi.items():
        if points:  # If there are valence intersections
            max_point = max(points, key=lambda point: point[1])  # Get the point with the maximum y
            max_valence_intersections[label] = max_point

    return max_valence_intersections
