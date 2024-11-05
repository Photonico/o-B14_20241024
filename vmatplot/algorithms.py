#### Algorithms codes
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914

import numpy as np

def is_nested_list(input_list): return isinstance(input_list, list) and isinstance(input_list[0], list)

def birch_murnaghan_equation_of_state(parameters, volume):
    # Birch-Murnaghan equation of state
    equilibrium_energy, bulk_modulus, bulk_modulus_derivative, equilibrium_volume = parameters
    compression_ratio = (volume / equilibrium_volume) ** (1.0 / 3.0)
    energy = (
        equilibrium_energy 
        + 9.0 * bulk_modulus * equilibrium_volume / 16.0 
        * (compression_ratio**2 - 1.0)**2 
        * (6.0 + bulk_modulus_derivative * (compression_ratio**2 - 1.0) - 4.0 * compression_ratio**2)
    )
    return energy

def polynomially_fit_curve(lattice_list, energy_list = None, degree = None, sample_count = None):
    help_info = "Usage: polynomially_fit_curve(lattice_list, energy_list, degree, sample_count)\n" + \
                "sample_count here means the sampling numbers.\n"
    # Check if the user asked for help
    if lattice_list in ["help"]:
        print(help_info)
        return
    # Ensure the other parameters are provided
    if energy_list is None or degree is None or sample_count is None:
        raise ValueError("Missing required parameters. Use 'help' for more information.")

    # Apply polynomial fitting to the data
    poly = np.polyfit(lattice_list, energy_list, degree)
    # Generate a polynomial function from the fitted parameters
    fitted = np.poly1d(poly)

    # Generate new x and y data using the polynomial function
    fitted_lattice = np.linspace(min(lattice_list), max(lattice_list), num=sample_count, endpoint=True)
    fitted_free_energy = fitted(fitted_lattice)
    return fitted_lattice, fitted_free_energy
