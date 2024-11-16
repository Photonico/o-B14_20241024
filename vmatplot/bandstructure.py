#### Bandstructure
# pylint: disable = C0103, C0114, C0116, C0301, C0302, C0321, R0913, R0914, R0915, W0612, W0105

import xml.etree.ElementTree as ET
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# from vmatplot.commons import identify_parameters

def extract_bandgap(directory="."):
    """
    Extract the bandgap (in eV) from a VASP band structure calculation directory.
    
    Parameters:
        directory (str): Path to the directory containing VASP output files (default is current directory).
    
    Returns:
        float: Bandgap value in eV if successful.
        str: Error message if required data is missing or cannot be processed.
    """
    vasprun_path = os.path.join(directory, "vasprun.xml")

    # Check if vasprun.xml exists
    if not os.path.exists(vasprun_path):
        return "Error: vasprun.xml not found in the specified directory."

    try:
        # Parse XML data from vasprun.xml
        tree = ET.parse(vasprun_path)
        root = tree.getroot()

        # Find the Fermi energy
        fermi_energy_tag = root.find(".//dos/i[@name='efermi']")
        if fermi_energy_tag is None:
            return "Error: Fermi energy not found in vasprun.xml."
        fermi_energy = float(fermi_energy_tag.text)

        # Extract band energies for each k-point
        eigenvalues = []
        for eigenvalue_set in root.findall(".//calculation/eigenvalues/array/set/set/set"):
            eigenvalues.append([
                float(entry.text.split()[0])  # Only take the first value (energy)
                for entry in eigenvalue_set.findall("r")
            ])

        if not eigenvalues:
            return "Error: Band eigenvalues not found in vasprun.xml."

        # Determine the highest valence band and lowest conduction band
        valence_band_max = float("-inf")
        conduction_band_min = float("inf")

        for band in eigenvalues:
            for energy in band:
                if energy <= fermi_energy:  # Valence band
                    valence_band_max = max(valence_band_max, energy)
                else:  # Conduction band
                    conduction_band_min = min(conduction_band_min, energy)

        # Calculate the bandgap
        if conduction_band_min > valence_band_max:
            bandgap = conduction_band_min - valence_band_max
            return bandgap
        else:
            return "Error: No bandgap found (metallic system)."

    except ET.ParseError:
        return "Error: Failed to parse vasprun.xml."
    except Exception as e:
        return f"Error: {str(e)}"
