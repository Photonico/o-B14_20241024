#### Common codes
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914

import xml.etree.ElementTree as ET
import os

import numpy as np

from typing import Tuple, Union

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
    """
    Extracts total atom count, total energy, Fermi energy, total kpoints, calculated kpoints, 
    kpoints mesh (x, y, z), lattice constant, SYMPREC, ENCUT, KSPACING, VOLUME, POTIM, AMIX, 
    BMIX, EDIFF, EDIFFG values, and elapsed time from VASP output files.
    """

    if directory == "help":
        print("Please use this function on the project directory.")
        return "Help provided."

    vasprun_path = os.path.join(directory, "vasprun.xml")
    kpoints_path = os.path.join(directory, "KPOINTS")
    outcar_path = os.path.join(directory, "OUTCAR")

    # Check file existence
    if not os.path.exists(vasprun_path):
        return "vasprun.xml file not found in the specified directory."
    if not os.path.exists(kpoints_path):
        return "KPOINTS file not found in the specified directory."

    try:
        # Parse XML data from vasprun.xml
        tree = ET.parse(vasprun_path)
        root = tree.getroot()

        parameters = {
            "total atom count": None,
            "total energy": None,
            "fermi energy": None,  # New entry for Fermi energy
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
            "elapsed time (sec)": None
        }

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
                raise ValueError("Kpoints type keyword not found in KPOINTS file.")

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
