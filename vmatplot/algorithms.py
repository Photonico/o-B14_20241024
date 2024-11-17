#### Algorithms codes
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914

import numpy as np

from scipy.optimize import leastsq

def get_matrix_shape(matrix):
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    return (rows, cols)

def transpose_matrix(matrix):
    return [list(row) for row in zip(*matrix)]

def compute_average(data_lines):
    """Define the function to compute the average of the last value in each line."""
    total = 0
    for line in data_lines:
        values = line.split()       # Split the line into individual values
        total += float(values[-1])  # Add the last value to the total
        print(line)
    return total / len(data_lines)  # Return the average

def is_nested_list(input_list): return isinstance(input_list, list) and isinstance(input_list[0], list)

def birch_murnaghan_equation_of_state(parameters, volume):
    """
    Birch-Murnaghan equation of state for energy calculation.
    
    Parameters:
    - parameters: list containing [equilibrium_energy, bulk_modulus, bulk_modulus_derivative, equilibrium_volume]
    - volume: Volume to calculate the energy for.
    
    Returns:
    - energy: Calculated energy for the given volume.
    """
    equilibrium_energy, bulk_modulus, bulk_modulus_derivative, equilibrium_volume = parameters
    compression_ratio = (volume / equilibrium_volume) ** (1.0 / 3.0)
    energy = (
        equilibrium_energy 
        + 9.0 * bulk_modulus * equilibrium_volume / 16.0 
        * (compression_ratio**2 - 1.0)**2 
        * (6.0 + bulk_modulus_derivative * (compression_ratio**2 - 1.0) - 4.0 * compression_ratio**2)
    )
    return energy

def objective_function(params, energy_values, volumes):
    """
    Objective function for least squares fitting using Birch-Murnaghan equation of state.
    
    Parameters:
    - params: list containing initial guess for the parameters [equilibrium_energy, bulk_modulus, bulk_modulus_derivative, equilibrium_volume]
    - energy_values: Observed energy values.
    - volumes: Volume values corresponding to observed energy values.
    
    Returns:
    - Difference between observed and calculated energy values for fitting.
    """
    calculated_energies = [birch_murnaghan_equation_of_state(params, v) for v in volumes]
    return energy_values - np.array(calculated_energies)

def fit_birch_murnaghan(lattice_values, energy_values, sample_count=12000):
    """
    Fit energy vs. volume data using the Birch-Murnaghan equation of state.
    
    Parameters:
    - lattice_values: List of lattice parameter values.
    - energy_values: List of corresponding energy values.
    - sample_count: Number of points to sample for resampled lattice output.
    
    Returns:
    - params: Fitted parameters [equilibrium_energy, bulk_modulus, bulk_modulus_derivative, equilibrium_volume].
    - resampled_lattice: Resampled lattice values for plotting.
    - fitted_energy: Fitted energy values corresponding to resampled lattice.
    """
    # Convert input lists to numpy arrays
    lattice_values = np.array(lattice_values)
    energy_values = np.array(energy_values)

    # Resample lattice_values for output
    resampled_lattice = np.linspace(min(lattice_values), max(lattice_values), num=sample_count, endpoint=True)
    volumes = lattice_values**3  # Assuming volume is the cube of the lattice parameter
    resampled_volumes = resampled_lattice**3

    # Initial guess for parameters: [equilibrium_energy, bulk_modulus, bulk_modulus_derivative, equilibrium_volume]
    E0 = min(energy_values)
    V0 = volumes[np.argmin(energy_values)]
    B0 = 0.1  # Initial guess for bulk modulus
    Bp = 4.0  # Initial guess for bulk modulus derivative
    initial_params = [E0, B0, Bp, V0]

    # Least squares fitting
    params, _ = leastsq(objective_function, initial_params, args=(energy_values, volumes))

    # Compute corresponding fitted energy values for resampled lattice
    fitted_energy = [birch_murnaghan_equation_of_state(params, v) for v in resampled_volumes]

    return params, resampled_lattice, fitted_energy

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
    fitted_energy = fitted(fitted_lattice)
    return fitted_lattice, fitted_energy
