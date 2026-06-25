# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914

# Necessary packages invoking
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from vmatplot.commons import process_boundary
from vmatplot.output_settings import canvas_setting, color_sampling

import matplotlib as mpl

mpl.rcParams["lines.solid_capstyle"] = "round"
mpl.rcParams["lines.dash_capstyle"]  = "round"
mpl.rcParams["lines.solid_joinstyle"] = "round"
mpl.rcParams["lines.dash_joinstyle"]  = "round"

def _resolve_oszicar_path(target_directory=".", filename="OSZICAR"):
    """
    Resolve OSZICAR path from either a directory or a direct file path.
    """
    if os.path.isdir(target_directory):
        oszicar_path = os.path.join(target_directory, filename)
    else:
        oszicar_path = target_directory

    if not os.path.isfile(oszicar_path):
        raise FileNotFoundError(f"OSZICAR was not found: {oszicar_path}")

    return oszicar_path


def _extract_value_after_key(tokens, key):
    """
    Extract the numerical value after a key such as T=, E=, F=, or E0=.
    """
    key_token = f"{key}="

    for index, token in enumerate(tokens):
        if token == key_token and index + 1 < len(tokens):
            return float(tokens[index + 1])

        if token.startswith(key_token) and len(token) > len(key_token):
            return float(token.split("=", 1)[1])

    raise ValueError(f"Cannot find key {key_token} in OSZICAR line.")


def extract_aimd_oszicar(target_directory=".", filename="OSZICAR",
                         energy_key="F", time_step=1.0, time_shift=0.0):
    """
    Extract AIMD step, time, energy, and temperature from OSZICAR.

    Parameters:
        target_directory: Directory containing OSZICAR, or direct OSZICAR path.
        filename: OSZICAR file name.
        energy_key: Energy tag to extract, such as "F", "E", or "E0".
                    Default "F" follows the original senior code.
        time_step: Time interval per AIMD step in fs.
        time_shift: Additional shift applied to the time axis in fs.

    Returns:
        dict: AIMD data containing step, time, energy, and temperature.
    """
    oszicar_path = _resolve_oszicar_path(target_directory, filename=filename)

    step_list = []
    time_list = []
    energy_list = []
    temperature_list = []

    with open(oszicar_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if "T=" not in line:
                continue

            tokens = line.split()

            try:
                current_step = int(tokens[0])
                current_temperature = _extract_value_after_key(tokens, "T")
                current_energy = _extract_value_after_key(tokens, energy_key)

                step_list.append(current_step)
                time_list.append(current_step * time_step + time_shift)
                energy_list.append(current_energy)
                temperature_list.append(current_temperature)

            except (ValueError, IndexError):
                continue

    if len(step_list) == 0:
        raise ValueError(f"No AIMD ionic-step data were found in {oszicar_path}.")

    return {
        "step": np.array(step_list, dtype=int),
        "time": np.array(time_list, dtype=float),
        "energy": np.array(energy_list, dtype=float),
        "temperature": np.array(temperature_list, dtype=float),
        "energy_key": energy_key,
        "oszicar": oszicar_path,
    }


def summarize_aimd_oszicar(target_directory=".", filename="OSZICAR",
                           energy_key="F", time_step=1.0, time_shift=0.0):
    """
    Summarize AIMD energy drift and temperature fluctuation.
    """
    aimd_data = extract_aimd_oszicar(target_directory, filename=filename,
                                     energy_key=energy_key,
                                     time_step=time_step,
                                     time_shift=time_shift)

    energy = aimd_data["energy"]
    temperature = aimd_data["temperature"]

    summary = {
        "steps": len(aimd_data["step"]),
        "initial energy": energy[0],
        "final energy": energy[-1],
        "energy drift": energy[-1] - energy[0],
        "mean temperature": np.mean(temperature),
        "temperature std": np.std(temperature),
        "minimum temperature": np.min(temperature),
        "maximum temperature": np.max(temperature),
    }

    return summary


def _apply_axis_boundary(ax, axis, boundary):
    """
    Apply x or y boundary using the vmatplot boundary convention.
    """
    boundary_start, boundary_end = process_boundary(boundary)

    if axis == "x":
        if boundary_start is not None or boundary_end is not None:
            ax.set_xlim(boundary_start, boundary_end)

    elif axis == "y":
        if boundary_start is not None or boundary_end is not None:
            ax.set_ylim(boundary_start, boundary_end)


def plot_aimd(title=None, target_directory=".", filename="OSZICAR",
              energy_key="F", time_step=1.0, time_shift=0.0,
              x_boundary=None, energy_boundary=None, temperature_boundary=None,
              energy_color="Blue", temperature_color="Red",
              line_style="solid", line_weight=1.5, line_alpha=1.0,
              legend_loc=False, grid=False):
    """
    Plot AIMD energy and temperature evolution from OSZICAR.

    Parameters:
        title: Figure title.
        target_directory: Directory containing OSZICAR, or direct OSZICAR path.
        filename: OSZICAR file name.
        energy_key: Energy tag to extract, such as "F", "E", or "E0".
        time_step: Time interval per AIMD step in fs.
        time_shift: Additional shift applied to the time axis in fs.
        x_boundary: X-axis boundary in fs.
        energy_boundary: Y-axis boundary for energy.
        temperature_boundary: Y-axis boundary for temperature.
        energy_color: Color family for energy curve.
        temperature_color: Color family for temperature curve.
        line_style: Line style.
        line_weight: Line width.
        line_alpha: Line alpha.
        legend_loc: Legend location. Set False to disable legend.
        grid: Whether to show grid.

    Returns:
        tuple: fig, axes, aimd_data.
    """
    help_info = """
    Usage: plot_aimd
        arg[0]: title;
        arg[1]: target directory or OSZICAR path;
        energy_key: "F", "E", or "E0";
        time_step: AIMD time step in fs;
        x_boundary: x-axis boundary in fs;
        energy_boundary: y-axis boundary for energy;
        temperature_boundary: y-axis boundary for temperature.
    """
    if title in ["help", "Help"]:
        print(help_info)
        return None

    # Data extracting
    aimd_data = extract_aimd_oszicar(target_directory, filename=filename,
                                     energy_key=energy_key,
                                     time_step=time_step,
                                     time_shift=time_shift)

    time = aimd_data["time"]
    energy = aimd_data["energy"]
    temperature = aimd_data["temperature"]

    # Figure settings
    fig_setting = canvas_setting(10, 6)
    params = fig_setting[2]
    plt.rcParams.update(params)

    fig, axes = plt.subplots(2, 1, figsize=fig_setting[0], dpi=fig_setting[1], sharex=True)
    ax_energy, ax_temperature = axes

    # Colors calling
    energy_colors = color_sampling(energy_color)
    temperature_colors = color_sampling(temperature_color)
    annotate_color = color_sampling("Grey")

    # Title
    if title not in [None, False, ""]:
        fig.suptitle(f"{title}", fontsize=fig_setting[3][0], y=1.00)

    # Energy plotting
    ax_energy.plot(time, energy, color=energy_colors[1], linestyle=line_style,
                   lw=line_weight, alpha=line_alpha, label=f"{energy_key} energy", zorder=4)
    ax_energy.set_ylabel("Energy (eV)")
    ax_energy.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    ax_energy.tick_params(labelbottom=False)

    # Temperature plotting
    ax_temperature.plot(time, temperature, color=temperature_colors[1], linestyle=line_style,
                        lw=line_weight, alpha=line_alpha, label="Temperature", zorder=4)
    ax_temperature.set_xlabel("Time (fs)")
    ax_temperature.set_ylabel("Temperature (K)")
    ax_temperature.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

    # Axis range
    _apply_axis_boundary(ax_energy, "x", x_boundary)
    _apply_axis_boundary(ax_temperature, "x", x_boundary)
    _apply_axis_boundary(ax_energy, "y", energy_boundary)
    _apply_axis_boundary(ax_temperature, "y", temperature_boundary)

    # Minor ticks
    ax_energy.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_energy.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax_temperature.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_temperature.yaxis.set_minor_locator(AutoMinorLocator(5))

    # Grid
    if grid is True:
        ax_energy.grid(color=annotate_color[1], linestyle="--", alpha=0.35, zorder=0)
        ax_temperature.grid(color=annotate_color[1], linestyle="--", alpha=0.35, zorder=0)

    # Legend
    if legend_loc not in [None, False]:
        current_legend_loc = "best" if legend_loc is True else legend_loc

        legend_energy = ax_energy.legend(loc=current_legend_loc, frameon=True, fancybox=True,
                                         shadow=False, facecolor="white",
                                         edgecolor=annotate_color[1], framealpha=0.9)
        legend_energy.get_frame().set_linewidth(1.0)

        legend_temperature = ax_temperature.legend(loc=current_legend_loc, frameon=True, fancybox=True,
                                                   shadow=False, facecolor="white",
                                                   edgecolor=annotate_color[1], framealpha=0.9)
        legend_temperature.get_frame().set_linewidth(1.0)

    plt.tight_layout()

    return fig, axes, aimd_data