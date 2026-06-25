#### AIMD
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, R0915

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
    Resolve OSZICAR path from either a calculation directory or a direct file path.
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

    for token_index, token in enumerate(tokens):
        if token == key_token and token_index + 1 < len(tokens):
            return float(tokens[token_index + 1])

        if token.startswith(key_token) and len(token) > len(key_token):
            return float(token.split("=", 1)[1])

    raise ValueError(f"Cannot find key {key_token} in OSZICAR line.")


def _select_color(color_input, color_index=1):
    """
    Select color from vmatplot color family, direct hex color, or matplotlib color name.
    """
    if color_input is None:
        return None

    if isinstance(color_input, str) and color_input.startswith("#"):
        return color_input

    try:
        color_list = color_sampling(color_input)
        if isinstance(color_list, (list, tuple)) and len(color_list) > color_index:
            return color_list[color_index]
        if isinstance(color_list, (list, tuple)) and len(color_list) > 0:
            return color_list[-1]
    except Exception:
        pass

    return color_input


def _apply_axis_boundary(ax, axis_name, boundary):
    """
    Apply x or y axis boundary while allowing None as one side.
    """
    if boundary is None:
        return

    boundary_low, boundary_high = process_boundary(boundary)

    if axis_name == "x":
        current_low, current_high = ax.get_xlim()

        if boundary_low is None:
            boundary_low = current_low
        if boundary_high is None:
            boundary_high = current_high

        ax.set_xlim(boundary_low, boundary_high)

    elif axis_name == "y":
        current_low, current_high = ax.get_ylim()

        if boundary_low is None:
            boundary_low = current_low
        if boundary_high is None:
            boundary_high = current_high

        ax.set_ylim(boundary_low, boundary_high)


def extract_aimd_oszicar(target_directory=".", filename="OSZICAR",
                         energy_key="F", time_step=1.0, time_shift=0.0):
    """
    Extract AIMD step, time, energy, and temperature from OSZICAR.

    Parameters:
        target_directory: Directory containing OSZICAR, or direct OSZICAR path.
        filename: OSZICAR file name.
        energy_key: Energy tag to extract, such as "F", "E", or "E0".
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
                current_step = int(float(tokens[0]))
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


def read_aimd_oszicar(*args, **kwargs):
    """
    Alias of extract_aimd_oszicar.
    """
    return extract_aimd_oszicar(*args, **kwargs)


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
        "mean energy": np.mean(energy),
        "energy std": np.std(energy),
        "minimum energy": np.min(energy),
        "maximum energy": np.max(energy),
        "mean temperature": np.mean(temperature),
        "temperature std": np.std(temperature),
        "minimum temperature": np.min(temperature),
        "maximum temperature": np.max(temperature),
    }

    return summary


def plot_aimd(title=None, target_directory=".", filename="OSZICAR",
              energy_key="F", time_step=1.0, time_shift=0.0,
              x_boundary=None, energy_boundary=None, temperature_boundary=None,
              energy_color="Blue", temperature_color="Red",
              energy_line_weight=1.5, temperature_line_weight=1.5,
              energy_line_style="solid", temperature_line_style="solid",
              energy_line_alpha=1.0, temperature_line_alpha=1.0,
              energy_label=None, temperature_label=None,
              legend_loc=False, grid=False, figure_size=(10, 6)):
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
        energy_color: Color family, hex color, or matplotlib color for energy curve.
        temperature_color: Color family, hex color, or matplotlib color for temperature curve.
        energy_line_weight: Line width of energy curve.
        temperature_line_weight: Line width of temperature curve.
        energy_line_style: Line style of energy curve.
        temperature_line_style: Line style of temperature curve.
        energy_line_alpha: Line alpha of energy curve.
        temperature_line_alpha: Line alpha of temperature curve.
        energy_label: Legend label of energy curve.
        temperature_label: Legend label of temperature curve.
        legend_loc: Legend location. Set False to disable legend.
        grid: Whether to show grid.
        figure_size: Figure size, such as (10, 6).

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
        temperature_boundary: y-axis boundary for temperature;
        energy_color: color for energy curve;
        temperature_color: color for temperature curve;
        energy_line_weight: line width for energy curve;
        temperature_line_weight: line width for temperature curve.
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
    fig_setting = canvas_setting(figure_size[0], figure_size[1])
    params = fig_setting[2]
    plt.rcParams.update(params)

    fig, axes = plt.subplots(2, 1, figsize=fig_setting[0], dpi=fig_setting[1], sharex=True)
    ax_energy, ax_temperature = axes

    # Colors calling
    energy_plot_color = _select_color(energy_color)
    temperature_plot_color = _select_color(temperature_color)
    annotate_color = color_sampling("Grey")

    # Label setting
    if energy_label is None:
        energy_label = f"{energy_key} energy"
    if temperature_label is None:
        temperature_label = "Temperature"

    # Title
    if title not in [None, False, ""]:
        fig.suptitle(f"{title}", fontsize=fig_setting[3][0], y=1.00)

    # Energy plotting
    ax_energy.plot(time, energy,
                   color=energy_plot_color,
                   linestyle=energy_line_style,
                   lw=energy_line_weight,
                   alpha=energy_line_alpha,
                   label=energy_label,
                   zorder=4)

    ax_energy.set_ylabel("Energy (eV)")
    ax_energy.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    ax_energy.tick_params(labelbottom=False)

    # Temperature plotting
    ax_temperature.plot(time, temperature,
                        color=temperature_plot_color,
                        linestyle=temperature_line_style,
                        lw=temperature_line_weight,
                        alpha=temperature_line_alpha,
                        label=temperature_label,
                        zorder=4)

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
        current_legend_loc = fig_setting[4] if legend_loc is True else legend_loc

        legend_energy = ax_energy.legend(loc=current_legend_loc,
                                         frameon=True,
                                         fancybox=True,
                                         shadow=False,
                                         facecolor="white",
                                         edgecolor=annotate_color[1],
                                         framealpha=0.9)
        legend_energy.get_frame().set_linewidth(1.0)

        legend_temperature = ax_temperature.legend(loc=current_legend_loc,
                                                   frameon=True,
                                                   fancybox=True,
                                                   shadow=False,
                                                   facecolor="white",
                                                   edgecolor=annotate_color[1],
                                                   framealpha=0.9)
        legend_temperature.get_frame().set_linewidth(1.0)

    if title not in [None, False, ""]:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()

    # return fig, axes, aimd_data
