#### Bandstructure
# pylint: disable = C0103, C0114, C0116, C0301, C0302, C0321, R0913, R0914, R0915, W0612, W0105

import xml.etree.ElementTree as ET
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# from vmatplot.commons import identify_parameters
from vmatplot.algorithms import transpose_matrix

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
