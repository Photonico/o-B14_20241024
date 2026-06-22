#### linear optical properties
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914, W0612

import numpy as np
import matplotlib.pyplot as plt

from vmatplot.dielectric_function import dielectric_systems_list
from vmatplot.commons import process_boundary_alt, extract_part
from vmatplot.output_settings import canvas_setting, color_sampling
from vmatplot.algorithms import energy_to_wavelength, energy_to_frequency, wavelength_to_energy

from matplotlib.colors import ListedColormap

import matplotlib as mpl

mpl.rcParams["lines.solid_capstyle"] = "round"
mpl.rcParams["lines.dash_capstyle"]  = "round"
mpl.rcParams["lines.solid_joinstyle"] = "round"
mpl.rcParams["lines.dash_joinstyle"]  = "round"

## References

# <https://vaspkit.com/tutorials.html#linear-optical-properties>

## Constants
c_ms = 2.99792458e8     # Speed of light in vacuum in meters per second
c_nm = c_ms * 1e9       # Speed of light in vacuum nanometers per second
hbar = 4.135667662e-15

## Theoretical formulas

# 1 absorption coefficient
def comp_absorption_coefficient(frequency,density_energy_real,density_energy_imag):
    coe = (np.sqrt(2)*frequency/c_nm)*(np.sqrt(np.sqrt(np.square(density_energy_real)+np.square(density_energy_imag))-density_energy_real))
    return coe

# 2 refractive index
def comp_refractive_index(density_energy_real,density_energy_imag):
    index = np.sqrt((np.sqrt(np.square(density_energy_real)+np.square(density_energy_imag))+density_energy_real)/2)
    return index

# 3 extinction coefficient
def comp_extinction_coefficient(density_energy_real,density_energy_imag):
    coe = np.sqrt((np.sqrt(np.square(density_energy_real)+np.square(density_energy_imag))-density_energy_real)/2)
    return coe

# 4 reflectivity
def comp_reflectivity(density_energy_real,density_energy_imag):
    n = comp_refractive_index(density_energy_real,density_energy_imag)
    k = comp_extinction_coefficient(density_energy_real,density_energy_imag)
    R = (np.square(n-1)+np.square(k))/(np.square(n+1)+np.square(k))
    return R

# 5 energy loss spectrum
def comp_energy_loss_spectrum(density_energy_real,density_energy_imag):
    spectrum = (density_energy_imag)/(np.square(density_energy_real)+np.square(density_energy_imag))
    return spectrum

## Plotting
# systems list
def lop_systems(*args):
    return dielectric_systems_list(*args)

def identify_linear_optical_functions(incoming = None):
    help_info = "Please use one of the following terminologies as a string-type variable:\n" + \
                "\t absorption coefficient, refractive index, extinction coefficient, reflectivity, energy-loss\n"
    linear_title, linear_flag, compfunc_name, plotfunc_name = None, None, None, None
    if incoming.lower() in ["absorption coefficient","absorption"]:
        linear_title = "Absorption coefficient"
        linear_flag = "absorption"
        compfunc_name = "comp_absorption_coefficient"
    elif incoming.lower() in ["refractive index","refractive"]:
        linear_title = "Refractive index"
        linear_flag = "refractive"
        compfunc_name = "comp_refractive_index"
    elif incoming.lower() in ["extinction coefficient", "extinction"]:
        linear_title = "Extinction coefficient"
        linear_flag = "extinction"
        compfunc_name = "comp_extinction_coefficient"
    elif incoming.lower() in ["reflectivity"]:
        linear_title = "Reflectivity"
        linear_flag = "reflectivity"
        compfunc_name = "comp_reflectivity"
    elif incoming.lower() in ["energy-loss function", "energy-loss spectrum", "energy-loss"]:
        linear_title = "Energy-loss spectrum"
        linear_flag = "energy-loss"
        compfunc_name = "comp_energy_loss_spectrum"
    else:
        print(help_info)
        return None
    return {"title":linear_title, "flag":linear_flag, "calculation function":compfunc_name}

# current linear optical propertie
def current_lop(lop_flag, *args):
    formula_flag = identify_linear_optical_functions(lop_flag)["flag"]
    if formula_flag == "absorption":
        return comp_absorption_coefficient(*args)
    elif formula_flag == "refractive":
        return comp_refractive_index(*args)
    elif formula_flag == "extinction":
        return comp_extinction_coefficient(*args)
    elif formula_flag == "reflectivity":
        return comp_reflectivity(*args)
    elif formula_flag == "energy-loss":
        return comp_energy_loss_spectrum(*args)

def determine_formula_flag(plotting_function_name):
    if plotting_function_name == "plot_absorption_coefficient":
        formula_flag = "absorption"
    elif plotting_function_name == "plot_refractive_index":
        formula_flag = "refractive"
    elif plotting_function_name == "plot_extinction_coefficient":
        formula_flag = "extinction"
    elif plotting_function_name == "plot_reflectivity":
        formula_flag = "reflectivity"
    elif plotting_function_name == "plot_energy_loss_spectrum":
        formula_flag = "energy-loss"
    return formula_flag

def lop_plotting_help():
    help_info = "Usage: plot_linear_optical_property \n" +\
                "Demonstrate linear optical properties by each component \n" +\
                "\t suptitle: the suptitle; \n" +\
                "\t systems_list: dielectric function data list; \n" +\
                "\t components: select components in a list ({'xx'<default>, 'yy', 'zz', 'xy', 'yx', 'yz', 'zy', 'zx', 'xz'}); \n" +\
                "\t expansion: select one variable to expansion (rescale<auto>, properties, systems); \n" +\
                "\t layout: subfigures layout (horizontal<default>, vertical); \n" +\
                "\t unit: x-axis unit (eV<default>, nm); \n" +\
                "\t photon_boundary: x-axis range <optional>; \n" +\
                "\t value_boundary: a-axis range <optional>; \n" +\
                "\t figure_size: figure size <optional>. \n"
    return help_info

def plot_linear_optical_property_backup(suptitle, systems=None, properties=None, components="xx",
                                 layout="horizontal", expansion_label=True,
                                 unit="eV", photon_boundary=(None, None), value_boundary=(None, None), 
                                 spectrum_flag=None, figure_size=(None, None)):
    ## Support information
    if suptitle.lower() in ["help", "support"]:
        help_info = lop_plotting_help()
        print(help_info)
        return
    # expansion label
    if expansion_label == False:
        return plot_merged_linear_optical_property(suptitle, systems, properties, components, layout, unit, photon_boundary, value_boundary, spectrum_flag, figure_size)
    else: pass

    # properties labels and determination
    multi_prop_flag = None
    if isinstance(properties, str):
        multi_prop_flag = False
        formula_flag = identify_linear_optical_functions(properties)["flag"]
        formula_title = identify_linear_optical_functions(properties)["title"]
    elif isinstance(properties, list):
        formula_flags, formula_titles =[],[]
        for formula_input in properties:
            formula_flags.append(identify_linear_optical_functions(formula_input)["flag"])
            formula_titles.append(identify_linear_optical_functions(formula_input)["title"])
        if len(properties) == 1:
            multi_prop_flag = False
            formula_flag = formula_flags[0]
            formula_title = formula_titles[0]
        elif len(properties) > 1:
            multi_prop_flag = True
    if multi_prop_flag is True:
        print("We currently do not support multiple linear optical properties in one figure")
        return None
    # multi sysatem determination
    multi_system_flag = False
    if len(systems) == 1:
        multi_system_flag = False
    else: multi_system_flag = True
    # multi component determination
    multi_comp_flag = None
    comp_labels, comp_aliases = [], []
    if isinstance(components, str):
        multi_comp_flag = False
        comp_labels.append(components)
        comp_aliases.append(f"{components}-component")
    elif isinstance(components, dict):
        if len(components) == 1:
            multi_comp_flag = False
            comp_labels.append(list(components.keys())[0])
            comp_aliases.append(list(components.values())[0])
        elif len(components) > 1:
            multi_comp_flag = True
            comp_labels.append(list(components.keys()))
            comp_aliases.append(list(components.values()))
    elif isinstance(components, list):
        if len(components) == 1:
            multi_comp_flag = False
        elif len(components) > 1:
            multi_comp_flag = True
        for comp_unit in components:
            if isinstance(comp_unit, dict):
                comp_labels.append(list(comp_unit.keys())[0])
                comp_aliases.append(list(comp_unit.values())[0])
            elif isinstance(comp_unit, str):
                comp_labels.append(comp_unit)
                comp_aliases.append(f"{comp_unit}-component")
    if multi_comp_flag is False:
        comp_label = comp_labels[0]
        comp_aliase = comp_aliases[0]

    ## boundaries processing
    photon_start, photon_end = process_boundary_alt(photon_boundary)

    ## identify x-axis unit
    var_label = "wavelength" if unit and unit.lower() == "nm" else "energy"
    xaxis_str = "Photon wavelength (nm)" if var_label == "wavelength" else "Photon energy (eV)"

    ## systems information
    dataset = dielectric_systems_list(systems)

    ## figure settings
    layout_flag = "horizontal" if layout.lower() not in ["vertical", "ver","v"] else "vertical"
    if multi_comp_flag is False:
        return plot_merged_linear_optical_property(suptitle, systems, properties, components, layout, unit, photon_boundary, value_boundary, spectrum_flag, figure_size)

    elif multi_comp_flag is True:
        ## figure settings
        if len(components) == 2:
            if layout_flag == "horizontal":
                fig_setting = canvas_setting(16, 6) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(1, 2, figsize=fig_setting[0], dpi=fig_setting[1])
                axs = axs.reshape(1, 2)
                axes_element = [axs[0, i] for i in range(2)]
            else:
                fig_setting = canvas_setting(8, 12) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(2, 1, figsize=fig_setting[0], dpi=fig_setting[1])
                axs = axs.reshape(2, 1)
                axes_element = [axs[i, 0] for i in range(2)]
        elif len(components) in [3, 5, 7]:
            if layout_flag == "horizontal":
                fig_setting = canvas_setting(8*len(components), 6) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(1, len(components), figsize=fig_setting[0], dpi=fig_setting[1])
                axes_element = [axs[i] for i in range(len(components))]
            else:
                fig_setting = canvas_setting(10, 6*len(components)) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(len(components), 1, figsize=fig_setting[0], dpi=fig_setting[1])
                axes_element = [axs[i] for i in range(len(components))]
        elif len(components) in [4, 6, 8]:
            if layout_flag == "horizontal":
                fig_setting = canvas_setting(8*len(components)/2, 12) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(2, int(len(components)/2), figsize=fig_setting[0], dpi=fig_setting[1])
                axes_element = [axs[i, j] for i in range(2) for j in range(int(len(components)/2))]
            else:
                fig_setting = canvas_setting(16, 6*len(components)/2+1) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(int(len(components)/2), 2, figsize=fig_setting[0], dpi=fig_setting[1])
                axes_element = [axs[i, j] for j in range(2) for i in range(int(len(components)/2))]
        elif len(components) == 9:
            if layout_flag == "horizontal":
                fig_setting = canvas_setting(24, 18) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(3, 3, figsize=fig_setting[0], dpi=fig_setting[1])
                axes_element = [axs[i, j] for i in range(3) for j in range(3)]
            else:
                fig_setting = canvas_setting(24, 18) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(3, 3, figsize=fig_setting[0], dpi=fig_setting[1])
                axes_element = [axs[i, j] for j in range(3) for i in range(3)]

        fig.suptitle(f"{suptitle}", fontsize=fig_setting[3][0])
        for subplot_index in range(len(components)):
            ax = axes_element[subplot_index]
            ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

            # components index and subtitles
            component_index = subplot_index
            ax.set_title(comp_aliases[component_index])

            # current component key and label
            current_component = comp_labels[component_index].lower()
            data_key_real = f"density_{current_component}_real"
            data_key_imag = f"density_{current_component}_imag"

            # curve plotting
            for _, data in enumerate(dataset):
                supercell_thickness, system_thickness = data[6]
                # print(data[6])
                d_ratio = supercell_thickness/system_thickness
                energy_real, density_energy_real_source = extract_part(data[1]["density_energy_real"], data[1][data_key_real], photon_start, photon_end)
                energy_imag, density_energy_imag_source = extract_part(data[1]["density_energy_imag"], data[1][data_key_imag], photon_start, photon_end)
                density_energy_real = density_energy_real_source * d_ratio - d_ratio + 1
                density_energy_imag= density_energy_imag_source * d_ratio
                frequency_real = energy_to_frequency(energy_real)
                wavelength_real = energy_to_wavelength(energy_real)
                if formula_flag == "absorption":
                    variables = current_lop(formula_flag,frequency_real,density_energy_real,density_energy_imag)
                else:
                    variables = current_lop(formula_flag,density_energy_real,density_energy_imag)
                if var_label == "energy":
                    ax.plot(energy_real, variables, color=color_sampling(data[2])[1], ls=data[3], alpha=data[5], lw=data[4], label=data[0])
                elif var_label == "wavelength":
                    wavelength_real, wavelength_variables = extract_part(energy_to_wavelength(data[1]["density_energy_real"]), data[1][data_key_real], photon_start, photon_end)
                    ax.plot(wavelength_real, wavelength_variables, color=color_sampling(data[2])[1], ls=data[3], alpha=data[5], lw=data[4], label=data[0])

            # y boundary
            y_min_source, y_max_source = ax.get_ylim()
            # print(y_min_source, y_max_source)
            y_low, y_hig = process_boundary_alt(value_boundary)
            y_sup = y_max_source if y_hig is None else min(y_hig, y_max_source)
            y_inf = y_min_source if y_low is None else y_low
            ax.set_ylim(y_inf, y_sup)

            # Spectrum
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(xmin, xmax)
            wl_vis = np.linspace(380, 750, 1000)    # 380 nm (violet) to 750 nm (red)
            ev_vis = wavelength_to_energy(wl_vis)   # 1.65 eV (red) to 3.26 eV (violet)
            cmap = plt.get_cmap("nipy_spectral")
            if spectrum_flag == True:
                if var_label == "energy":
                    colors = cmap(np.linspace(0, 1, 1000))
                    idx_sort = np.argsort(ev_vis)
                    colors_sorted = colors[idx_sort]
                    energy_cmap = ListedColormap(colors_sorted)
                    ev_min, ev_max = np.min(ev_vis), np.max(ev_vis)
                    grad = np.linspace(0, 1, 1000).reshape(1, -1)
                    grad = np.vstack([grad] * 10)
                    alpha_vals = np.sin(np.linspace(0, np.pi, 1000)) * 2.0
                    alpha_vals = np.clip(alpha_vals, 0, 0.325)
                    alpha_grad = np.tile(alpha_vals, (10, 1))
                    ymin, ymax = ax.get_ylim()
                    extent = [ev_min, ev_max, ymin, ymax]
                    ax.imshow(grad, aspect="auto", extent=extent, cmap=energy_cmap, alpha=alpha_grad*0.6, zorder=-12)
                else:
                    grad = np.linspace(0, 1, 1000).reshape(1, -1)
                    grad = np.vstack([grad] * 10)
                    alpha_vals = np.sin(np.linspace(0, np.pi, 1000)) * 0.4
                    alpha_vals = np.clip(alpha_vals, 0, 1)
                    alpha_grad = np.tile(alpha_vals, (10, 1))
                    ymin, ymax = ax.get_ylim()
                    extent = [380, 750, ymin, ymax]
                    ax.imshow(grad, aspect="auto", extent=extent, cmap=cmap, alpha=alpha_grad*0.6, zorder=-12)
            else: pass

            # axis labels
            ax.legend(loc="best")
            if layout_flag == "horizontal" and len(components) == 2:
                ax.set_xlabel(xaxis_str)
                if subplot_index == 0:
                    ax.set_ylabel(f"{formula_title}")
            elif layout_flag == "vertical" and len(components) == 2:
                ax.set_ylabel(f"{formula_title}")
                if subplot_index == len(components)-1:
                    ax.set_xlabel(xaxis_str)
            elif layout_flag == "horizontal" and len(components) == 9:
                if subplot_index >= len(components)-3:
                    ax.set_xlabel(xaxis_str)
                if subplot_index%3==0:
                    ax.set_ylabel(f"{formula_title}")
            elif layout_flag == "vertical" and len(components) == 9:
                if subplot_index%3==2:
                    ax.set_xlabel(xaxis_str)
                if subplot_index in [0,1,2]:
                    ax.set_ylabel(f"{formula_title}")
            elif layout_flag == "horizontal" and len(components)%2==0:
                if subplot_index >= len(components)-(len(components)/2):
                    ax.set_xlabel(xaxis_str)
                if subplot_index%(len(components)/2)==0:
                    ax.set_ylabel(f"{formula_title}")
            elif layout_flag == "vertical" and len(components)%2==0:
                if (subplot_index+1) % len(components)==0:
                    ax.set_xlabel(xaxis_str)
                if subplot_index < len(components):
                    ax.set_ylabel(f"{formula_title}")
            elif layout_flag == "horizontal" and len(components) in [3,5,7]:
                ax.set_xlabel(xaxis_str)
                if subplot_index == 0:
                    ax.set_ylabel(f"{formula_title}")
            elif layout_flag == "vertical" and len(components) in [3,5,7]:
                if subplot_index == len(components)-1:
                    ax.set_xlabel(xaxis_str)
                ax.set_ylabel(f"{formula_title}")
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)
            plt.tight_layout()

def plot_linear_optical_property(suptitle, systems=None, properties=None, components="xx",
                                 layout="horizontal", expansion_label=True,
                                 unit="eV", photon_boundary=(None, None), value_boundary=(None, None), 
                                 spectrum_flag=None, figure_size=(None, None)):
    ## Support information
    if suptitle.lower() in ["help", "support"]:
        help_info = lop_plotting_help()
        print(help_info)
        return
    # expansion label
    if expansion_label == False:
        return plot_merged_linear_optical_property(suptitle, systems, properties, components, layout, unit, photon_boundary, value_boundary, spectrum_flag, figure_size)
    else: pass

    # properties labels and determination
    multi_prop_flag = None
    if isinstance(properties, str):
        multi_prop_flag = False
        formula_flag = identify_linear_optical_functions(properties)["flag"]
        formula_title = identify_linear_optical_functions(properties)["title"]
    elif isinstance(properties, list):
        formula_flags, formula_titles =[],[]
        for formula_input in properties:
            formula_flags.append(identify_linear_optical_functions(formula_input)["flag"])
            formula_titles.append(identify_linear_optical_functions(formula_input)["title"])
        if len(properties) == 1:
            multi_prop_flag = False
            formula_flag = formula_flags[0]
            formula_title = formula_titles[0]
        elif len(properties) > 1:
            multi_prop_flag = True
    if multi_prop_flag is True:
        print("We currently do not support multiple linear optical properties in one figure")
        return None
    # multi sysatem determination
    multi_system_flag = False
    if len(systems) == 1:
        multi_system_flag = False
    else: multi_system_flag = True
    # multi component determination
    multi_comp_flag = None
    comp_labels, comp_aliases = [], []
    if isinstance(components, str):
        multi_comp_flag = False
        comp_labels.append(components)
        comp_aliases.append(f"{components}-component")
    elif isinstance(components, dict):
        if len(components) == 1:
            multi_comp_flag = False
            comp_labels.append(list(components.keys())[0])
            comp_aliases.append(list(components.values())[0])
        elif len(components) > 1:
            multi_comp_flag = True
            comp_labels.append(list(components.keys()))
            comp_aliases.append(list(components.values()))
    elif isinstance(components, list):
        if len(components) == 1:
            multi_comp_flag = False
        elif len(components) > 1:
            multi_comp_flag = True
        for comp_unit in components:
            if isinstance(comp_unit, dict):
                comp_labels.append(list(comp_unit.keys())[0])
                comp_aliases.append(list(comp_unit.values())[0])
            elif isinstance(comp_unit, str):
                comp_labels.append(comp_unit)
                comp_aliases.append(f"{comp_unit}-component")
    if multi_comp_flag is False:
        comp_label = comp_labels[0]
        comp_aliase = comp_aliases[0]

    ## boundaries processing
    photon_start, photon_end = process_boundary_alt(photon_boundary)

    ## identify x-axis unit
    var_label = "wavelength" if unit and unit.lower() == "nm" else "energy"
    xaxis_str = "Photon wavelength (nm)" if var_label == "wavelength" else "Photon energy (eV)"

    ## systems information
    dataset = dielectric_systems_list(systems)

    ## figure settings
    layout_flag = "horizontal" if layout.lower() not in ["vertical", "ver","v"] else "vertical"
    if multi_comp_flag is False:
        return plot_merged_linear_optical_property(suptitle, systems, properties, components, layout, unit, photon_boundary, value_boundary, spectrum_flag, figure_size)

    elif multi_comp_flag is True:
        ## figure settings
        if len(components) == 2:
            if layout_flag == "horizontal":
                fig_setting = canvas_setting(16, 6) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(1, 2, figsize=fig_setting[0], dpi=fig_setting[1])
                axs = axs.reshape(1, 2)
                nrows, ncols = np.atleast_2d(axs).shape
                axes_element = [axs[0, i] for i in range(2)]
            else:
                fig_setting = canvas_setting(8, 12) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(2, 1, figsize=fig_setting[0], dpi=fig_setting[1])
                axs = axs.reshape(2, 1)
                nrows, ncols = np.atleast_2d(axs).shape
                axes_element = [axs[i, 0] for i in range(2)]
        elif len(components) in [3, 5, 7]:
            if layout_flag == "horizontal":
                fig_setting = canvas_setting(8*len(components), 6) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(1, len(components), figsize=fig_setting[0], dpi=fig_setting[1])
                nrows, ncols = np.atleast_2d(axs).shape
                axes_element = [axs[i] for i in range(len(components))]
            else:
                fig_setting = canvas_setting(10, 6*len(components)) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(len(components), 1, figsize=fig_setting[0], dpi=fig_setting[1])
                nrows, ncols = np.atleast_2d(axs).shape
                axes_element = [axs[i] for i in range(len(components))]
        elif len(components) in [4, 6, 8]:
            if layout_flag == "horizontal":
                fig_setting = canvas_setting(8*len(components)/2, 12) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(2, int(len(components)/2), figsize=fig_setting[0], dpi=fig_setting[1])
                nrows, ncols = np.atleast_2d(axs).shape
                axes_element = [axs[i, j] for i in range(2) for j in range(int(len(components)/2))]
            else:
                fig_setting = canvas_setting(16, 6*len(components)/2+1) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(int(len(components)/2), 2, figsize=fig_setting[0], dpi=fig_setting[1])
                nrows, ncols = np.atleast_2d(axs).shape
                axes_element = [axs[i, j] for j in range(2) for i in range(int(len(components)/2))]
        elif len(components) == 9:
            if layout_flag == "horizontal":
                fig_setting = canvas_setting(24, 18) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(3, 3, figsize=fig_setting[0], dpi=fig_setting[1])
                nrows, ncols = np.atleast_2d(axs).shape
                axes_element = [axs[i, j] for i in range(3) for j in range(3)]
            else:
                fig_setting = canvas_setting(24, 18) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
                params = fig_setting[2]
                plt.rcParams.update(params)
                fig, axs = plt.subplots(3, 3, figsize=fig_setting[0], dpi=fig_setting[1])
                nrows, ncols = np.atleast_2d(axs).shape
                axes_element = [axs[i, j] for j in range(3) for i in range(3)]

        fig.suptitle(f"{suptitle}", fontsize=fig_setting[3][0])
        for subplot_index in range(len(components)):
            ax = axes_element[subplot_index]
            ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

            # components index and subtitles
            component_index = subplot_index
            ax.set_title(comp_aliases[component_index])

            # current component key and label
            current_component = comp_labels[component_index].lower()
            data_key_real = f"density_{current_component}_real"
            data_key_imag = f"density_{current_component}_imag"

            # curve plotting
            for _, data in enumerate(dataset):
                supercell_thickness, system_thickness = data[6]
                # print(data[6])
                d_ratio = supercell_thickness/system_thickness
                energy_real, density_energy_real_source = extract_part(data[1]["density_energy_real"], data[1][data_key_real], photon_start, photon_end)
                energy_imag, density_energy_imag_source = extract_part(data[1]["density_energy_imag"], data[1][data_key_imag], photon_start, photon_end)
                density_energy_real = density_energy_real_source * d_ratio - d_ratio + 1
                density_energy_imag= density_energy_imag_source * d_ratio
                frequency_real = energy_to_frequency(energy_real)
                wavelength_real = energy_to_wavelength(energy_real)
                if formula_flag == "absorption":
                    variables = current_lop(formula_flag,frequency_real,density_energy_real,density_energy_imag)
                else:
                    variables = current_lop(formula_flag,density_energy_real,density_energy_imag)
                if var_label == "energy":
                    ax.plot(energy_real, variables, color=color_sampling(data[2])[1], ls=data[3], alpha=data[5], lw=data[4], label=data[0])
                elif var_label == "wavelength":
                    wavelength_real, wavelength_variables = extract_part(energy_to_wavelength(data[1]["density_energy_real"]), data[1][data_key_real], photon_start, photon_end)
                    ax.plot(wavelength_real, wavelength_variables, color=color_sampling(data[2])[1], ls=data[3], alpha=data[5], lw=data[4], label=data[0])

            # y boundary
            y_min_source, y_max_source = ax.get_ylim()
            # print(y_min_source, y_max_source)
            y_low, y_hig = process_boundary_alt(value_boundary)
            y_sup = y_max_source if y_hig is None else min(y_hig, y_max_source)
            y_inf = y_min_source if y_low is None else y_low
            ax.set_ylim(y_inf, y_sup)

            # Spectrum
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(xmin, xmax)
            wl_vis = np.linspace(380, 750, 1000)    # 380 nm (violet) to 750 nm (red)
            ev_vis = wavelength_to_energy(wl_vis)   # 1.65 eV (red) to 3.26 eV (violet)
            cmap = plt.get_cmap("nipy_spectral")
            if spectrum_flag == True:
                if var_label == "energy":
                    colors = cmap(np.linspace(0, 1, 1000))
                    idx_sort = np.argsort(ev_vis)
                    colors_sorted = colors[idx_sort]
                    energy_cmap = ListedColormap(colors_sorted)
                    ev_min, ev_max = np.min(ev_vis), np.max(ev_vis)
                    grad = np.linspace(0, 1, 1000).reshape(1, -1)
                    grad = np.vstack([grad] * 10)
                    alpha_vals = np.sin(np.linspace(0, np.pi, 1000)) * 0.4
                    alpha_vals = np.clip(alpha_vals, 0, 1)
                    alpha_grad = np.tile(alpha_vals, (10, 1))
                    ymin, ymax = ax.get_ylim()
                    extent = [ev_min, ev_max, ymin, ymax]
                    ax.imshow(grad, aspect="auto", extent=extent, cmap=energy_cmap, alpha=alpha_grad*0.6, zorder=-12)
                else:
                    grad = np.linspace(0, 1, 1000).reshape(1, -1)
                    grad = np.vstack([grad] * 10)
                    alpha_vals = np.sin(np.linspace(0, np.pi, 1000)) * 0.4
                    alpha_vals = np.clip(alpha_vals, 0, 1)
                    alpha_grad = np.tile(alpha_vals, (10, 1))
                    ymin, ymax = ax.get_ylim()
                    extent = [380, 750, ymin, ymax]
                    ax.imshow(grad, aspect="auto", extent=extent, cmap=cmap, alpha=alpha_grad*0.6, zorder=-12)
            else: pass

            # axis labels
            ax.legend(loc="best")
            if layout_flag == "horizontal" and len(components) == 2:
                ax.set_xlabel(xaxis_str, fontsize=14+2*nrows)
                if subplot_index == 0:
                    ax.set_ylabel(f"{formula_title}", fontsize=14+2*ncols)
            elif layout_flag == "vertical" and len(components) == 2:
                ax.set_ylabel(f"{formula_title}", fontsize=14+2*ncols)
                if subplot_index == len(components)-1:
                    ax.set_xlabel(xaxis_str, fontsize=14+2*nrows)
            elif layout_flag == "horizontal" and len(components) == 9:
                if subplot_index >= len(components)-3:
                    ax.set_xlabel(xaxis_str, fontsize=14+2*nrows)
                if subplot_index%3==0:
                    ax.set_ylabel(f"{formula_title}", fontsize=14+2*ncols)
            elif layout_flag == "vertical" and len(components) == 9:
                if subplot_index%3==2:
                    ax.set_xlabel(xaxis_str, fontsize=14+2*nrows)
                if subplot_index in [0,1,2]:
                    ax.set_ylabel(f"{formula_title}", fontsize=14+2*ncols)
            elif layout_flag == "horizontal" and len(components)%2==0:
                if subplot_index >= len(components)-(len(components)/2):
                    ax.set_xlabel(xaxis_str, fontsize=14+2*nrows)
                if subplot_index%(len(components)/2)==0:
                    ax.set_ylabel(f"{formula_title}", fontsize=14+2*ncols)
            elif layout_flag == "vertical" and len(components)%2==0:
                if (subplot_index+1) % len(components)==0:
                    ax.set_xlabel(xaxis_str, fontsize=14+2*nrows)
                if subplot_index < len(components):
                    ax.set_ylabel(f"{formula_title}", fontsize=14+2*ncols)
            elif layout_flag == "horizontal" and len(components) in [3,5,7]:
                ax.set_xlabel(xaxis_str, fontsize=14+2*nrows)
                if subplot_index == 0:
                    ax.set_ylabel(f"{formula_title}", fontsize=14+2*ncols)
            elif layout_flag == "vertical" and len(components) in [3,5,7]:
                if subplot_index == len(components)-1:
                    ax.set_xlabel(xaxis_str, fontsize=14+2*nrows)
                ax.set_ylabel(f"{formula_title}", fontsize=14+2*ncols)
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)
            plt.tight_layout()

def plot_merged_linear_optical_property(suptitle, systems=None, properties=None, components="xx",
                                        layout="horizontal",unit="eV", photon_boundary=(None, None), value_boundary=(None, None), 
                                        spectrum_flag=None, figure_size=(None, None)):
    """
    Plot all systems (and potentially multiple components) in a single Axes.
    Differences from plot_linear_optical_property:
      1 Only one figure/axes (no subplots).
      2 If multiple components are given, iteration order is:
        for comp_idx in comp_labels -> for data_item in dataset
        (component first, then system).
      3 If single component: keep dataset's original linestyle, color_sampling(...)[1].
        If multiple components: ignore original linestyle, use style_cycle + color index.
        style_cycle = ["solid", "dashed", "dashdot", "dotted", "dashdotdotted",
                       "dashed", "dashdot", "dotted", "dashdotdotted"]
    """
    # Predefined style cycle for up to 9 components
    style_cycle = ["solid","dashed","dashdot","dotted","dashdotdotted","dashed","dashdot","dotted","dashdotdotted"]
    # Help info
    if suptitle.lower() in ["help","support"]:
        help_info = lop_plotting_help()
        print(help_info)
        return

    # Optical property
    multi_prop_flag = None
    if isinstance(properties, str):
        multi_prop_flag = False
        prop_dict = identify_linear_optical_functions(properties)
        formula_flag = prop_dict["flag"]
        formula_title = prop_dict["title"]
    elif isinstance(properties, list):
        formula_flags, formula_titles = [], []
        for prop_input in properties:
            prop_dict = identify_linear_optical_functions(prop_input)
            formula_flags.append(prop_dict["flag"])
            formula_titles.append(prop_dict["title"])
        if len(properties) == 1:
            multi_prop_flag = False
            formula_flag = formula_flags[0]
            formula_title = formula_titles[0]
        elif len(properties) > 1:
            multi_prop_flag = True
    if multi_prop_flag is True:
        print("Currently we do not support multiple linear optical properties in one figure.")
        return None

    # Systems
    dataset = dielectric_systems_list(systems)
    multi_system_flag = (len(dataset) > 1)
    # Components
    multi_comp_flag = None
    comp_labels, comp_aliases = [], []
    if isinstance(components, str):
        multi_comp_flag = False
        comp_labels.append(components)
        comp_aliases.append(f"{components}-component")
    elif isinstance(components, dict):
        if len(components) == 1:
            multi_comp_flag = False
            k0 = list(components.keys())[0]
            v0 = list(components.values())[0]
            comp_labels.append(k0)
            comp_aliases.append(v0)
        else:
            multi_comp_flag = True
            for k,v in components.items():
                comp_labels.append(k)
                comp_aliases.append(v)
    elif isinstance(components, list):
        if len(components) == 1: multi_comp_flag = False
        elif len(components) > 1: multi_comp_flag = True
        for comp_unit in components:
            if isinstance(comp_unit, dict):
                dk = list(comp_unit.keys())[0]
                dv = list(comp_unit.values())[0]
                comp_labels.append(dk)
                comp_aliases.append(dv)
            elif isinstance(comp_unit, str):
                comp_labels.append(comp_unit)
                comp_aliases.append(f"{comp_unit}-component")
    if not multi_comp_flag:
        comp_label = comp_labels[0]
        comp_aliase = comp_aliases[0]
    # Figure settings
    layout_flag = "horizontal" if layout.lower() not in ["vertical","ver","v"] else "vertical"
    default_fig_size = (12,6) if layout_flag=="horizontal" else (6,12)
    fig_setting = canvas_setting(default_fig_size[0], default_fig_size[1]) if figure_size==(None,None) else canvas_setting(figure_size[0], figure_size[1])
    plt.figure(figsize=fig_setting[0], dpi=fig_setting[1])
    params = fig_setting[2]
    plt.rcParams.update(params)
    photon_start, photon_end = process_boundary_alt(photon_boundary)
    var_label = "wavelength" if unit and unit.lower()=="nm" else "energy"
    xaxis_str = "Photon wavelength (nm)" if var_label=="wavelength" else "Photon energy (eV)"
    plt.title(f"{suptitle}", fontsize=fig_setting[3][0])
    plt.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)
    # Plot
    if not multi_comp_flag:
        # Single component => use original linestyle + color_sampling(...)[1]
        current_component = comp_label.lower()
        dkey_real = f"density_{current_component}_real"
        dkey_imag = f"density_{current_component}_imag"
        for _, data_item in enumerate(dataset):
            supercell_thickness, system_thickness = data_item[6]
            d_ratio = supercell_thickness/system_thickness
            e_real, den_e_real_source = extract_part(data_item[1]["density_energy_real"], data_item[1][dkey_real], photon_start, photon_end)
            e_imag, den_e_imag_source = extract_part(data_item[1]["density_energy_imag"], data_item[1][dkey_imag], photon_start, photon_end)
            den_e_real = den_e_real_source * d_ratio - d_ratio + 1
            den_e_imag = den_e_imag_source * d_ratio
            freq_real = energy_to_frequency(e_real)
            variables = current_lop(formula_flag, freq_real, den_e_real, den_e_imag) if formula_flag=="absorption" else current_lop(formula_flag, den_e_real, den_e_imag)
            clist = color_sampling(data_item[2])
            if len(clist)>1: line_color = clist[1]
            elif len(clist)>0: line_color = clist[0]
            else: line_color = "blue"
            line_style = data_item[3]
            if var_label=="energy":
                plt.plot(e_real, variables, color=line_color, ls=line_style, alpha=data_item[5], lw=data_item[4], label=f"{data_item[0]}")
            else:
                wl_real, wl_variables = extract_part(energy_to_wavelength(data_item[1]["density_energy_real"]), data_item[1][dkey_real], photon_start, photon_end)
                plt.plot(wl_real, wl_variables, color=line_color, ls=line_style, alpha=data_item[5], lw=data_item[4], label=f"{data_item[0]}")
    else:
        # Multiple components => ignore original linestyle, use style_cycle + color index logic
        for comp_idx, c_label in enumerate(comp_labels):
            c_alias = comp_aliases[comp_idx]
            ckey_lower = c_label.lower()
            dkey_real = f"density_{ckey_lower}_real"
            dkey_imag = f"density_{ckey_lower}_imag"
            for _, data_item in enumerate(dataset):
                supercell_thickness, system_thickness = data_item[6]
                d_ratio = supercell_thickness/system_thickness
                e_real, den_e_real_source = extract_part(data_item[1]["density_energy_real"], data_item[1][dkey_real], photon_start, photon_end)
                e_imag, den_e_imag_source = extract_part(data_item[1]["density_energy_imag"], data_item[1][dkey_imag], photon_start, photon_end)
                den_e_real = den_e_real_source * d_ratio - d_ratio + 1
                den_e_imag = den_e_imag_source * d_ratio
                freq_real = energy_to_frequency(e_real)
                variables = current_lop(formula_flag, freq_real, den_e_real, den_e_imag) if formula_flag=="absorption" else current_lop(formula_flag, den_e_real, den_e_imag)
                if comp_idx<9:
                    line_style = style_cycle[comp_idx]  # up to 9
                    color_idx = 1 if comp_idx<5 else 2  # first 5 => index1, next4 => index2
                else:
                    line_style = "solid"
                    color_idx = 0
                clist = color_sampling(data_item[2])
                line_color = clist[color_idx] if color_idx<len(clist) else "blue"
                if data_item[0] not in ["", None]: line_label = f"{data_item[0]} ({c_alias})"
                else: line_label = f"{data_item[0]} {c_alias}"
                if var_label=="energy":
                    plt.plot(e_real, variables, color=line_color, ls=line_style, alpha=data_item[5], lw=data_item[4], label=line_label)
                else:
                    wl_real, wl_variables = extract_part(energy_to_wavelength(data_item[1]["density_energy_real"]), data_item[1][dkey_real], photon_start, photon_end)
                    plt.plot(wl_real, wl_variables, color=line_color, ls=line_style, alpha=data_item[5], lw=data_item[4], label=line_label)

    # y boundary
    y_min_source, y_max_source = plt.ylim()
    # print(y_min_source, y_max_source)
    y_low, y_hig = process_boundary_alt(value_boundary)
    y_low, y_hig = process_boundary_alt(value_boundary)
    y_sup = y_max_source if y_hig is None else min(y_hig, y_max_source)
    y_inf = y_min_source if y_low is None else y_low
    plt.ylim(y_inf, y_sup)

    # Spectrum
    xmin, xmax = plt.xlim()
    plt.xlim(xmin, xmax)
    wl_vis = np.linspace(380, 750, 1000)                # 380 nm (violet) to 750 nm (red)
    ev_vis = wavelength_to_energy(wl_vis)               # 1.65 eV (red) to 3.26 eV (violet)
    cmap = plt.get_cmap("nipy_spectral")
    if spectrum_flag == True:
        if var_label == "energy":
            cmap = plt.get_cmap("nipy_spectral")
            colors = cmap(np.linspace(0, 1, 1000))
            idx_sort = np.argsort(ev_vis)
            colors_sorted = colors[idx_sort]
            energy_cmap = ListedColormap(colors_sorted)
            ev_min, ev_max = np.min(ev_vis), np.max(ev_vis)
            grad = np.linspace(0, 1, 1000).reshape(1, -1)
            grad = np.vstack([grad] * 10)
            alpha_vals = np.sin(np.linspace(0, np.pi, 1000)) * 0.4
            alpha_vals = np.clip(alpha_vals, 0, 1)
            alpha_grad = alpha_vals.reshape(1, -1)
            alpha_grad = np.vstack([alpha_grad] * 10)
            ymin, ymax = plt.ylim()
            extent = [ev_min, ev_max, ymin, ymax]
            plt.imshow(grad, aspect="auto", extent=extent, cmap=energy_cmap, alpha=alpha_grad*0.6, zorder=-12)
        else:
            grad = np.linspace(0, 1, 1000).reshape(1, -1)
            grad = np.vstack([grad] * 10)
            visible_spectrum_cmap = plt.cm.nipy_spectral
            alpha_vals = np.sin(np.linspace(0, np.pi, 1000)) * 0.4
            alpha_vals = np.clip(alpha_vals, 0, 1)
            alpha_grad = np.tile(alpha_vals, (10, 1))
            ymin, ymax = plt.ylim()
            extent = [380, 750, ymin, ymax]
            plt.imshow(grad, aspect="auto", extent=extent, cmap=visible_spectrum_cmap, alpha=alpha_grad*0.6, zorder=-12)
    else: pass
    # Axis labels
    plt.xlabel(xaxis_str)
    plt.ylabel(f"{formula_title}")
    plt.legend(loc="best")
    plt.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)
    plt.tight_layout()
