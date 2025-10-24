#### Declarations of process functions for Dielectric function
# pylint: disable = C0103, C0114, C0116, C0301, R0914

def plot_dielectric_function(suptitle, systems=None, components=None,
                             layout="horizontal", expansion_label=True,
                             unit=None, x_boundary=(None, None), y_boundary=(None, None),
                             spectrum_flag=None, figure_size = (None,None)):
    ## Help information
    dielectric_help =  plot_dielectric_help()
    if suptitle in ["help", "Help"]:
        print(dielectric_help)
        return None

    ## multi components flag
    if isinstance(components, str) or isinstance(components, dict):
        return plot_dielectric_monocomp(suptitle, systems, components, layout, expansion_label, unit, x_boundary, spectrum_flag, figure_size)
    elif isinstance(components, list) and len(components) == 1:
        return plot_dielectric_monocomp(suptitle, systems, components, layout, expansion_label, unit, x_boundary, spectrum_flag, figure_size)

    ## rescale
    if isinstance(x_boundary, tuple):
        if isinstance(x_boundary[0], tuple) or isinstance(x_boundary[1], tuple):
            return plot_dielectric_function_rescaled(suptitle, systems, components, layout, unit, x_boundary, spectrum_flag, figure_size)
        else: pass
    else: pass

    ## expansion flag
    if isinstance(expansion_label, bool):
        expansion_flag = expansion_label
    elif expansion_label.lower() not in ["true", "yes", "t", "y", "combine"]:
        expansion_flag = False
    else:
        expansion_flag = True

    ## components aliases
    comp_labels, comp_aliases = [], []
    for comp in components:
        if isinstance(comp, dict):
            for key, value in comp.items():
                comp_labels.append(key.lower())
                comp_aliases.append(value)
        else:
            comp_labels.append(comp.lower())
            comp_aliases.append(f"{comp}-component")

    ## figure settings
    folding_flag = None
    allcomps_flag = None
    layout_flag = "horizontal" if layout.lower() not in ["vertical", "ver","v"] else "vertical"
    if expansion_flag is True:
        if layout_flag == "horizontal":
            fig_setting = canvas_setting(8*len(components), 12) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(2, len(components), figsize=fig_setting[0], dpi=fig_setting[1])
            nrows, ncols = np.array(axs).shape if hasattr(axs, "shape") else (1, 1)
            axes_element = [axs[i, j] for j in range(len(components)) for i in range(2)] if len(components) != 1 else [axs[0], axs[1]]
        else:
            fig_setting = canvas_setting(16, 6*len(components)) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(len(components), 2, figsize=fig_setting[0], dpi=fig_setting[1])
            nrows, ncols = np.array(axs).shape if hasattr(axs, "shape") else (1, 1)
            axes_element = [axs[i, j] for i in range(len(components)) for j in range(2)] if len(components) != 1 else [axs[0], axs[1]]
    elif expansion_flag is False and len(components) == 2:
        if layout_flag == "horizontal":
            fig_setting = canvas_setting(16, 6) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(1, 2, figsize=fig_setting[0], dpi=fig_setting[1])
            nrows, ncols = np.array(axs).shape if hasattr(axs, "shape") else (1, 1)
            axs = axs.reshape(1, 2)
            axes_element = [axs[0, i] for i in range(2)]
        else:
            fig_setting = canvas_setting(8, 12) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(2, 1, figsize=fig_setting[0], dpi=fig_setting[1])
            nrows, ncols = np.array(axs).shape if hasattr(axs, "shape") else (1, 1)
            axs = axs.reshape(2, 1)
            axes_element = [axs[i, 0] for i in range(2)]
    elif expansion_flag is False and len(components)%2 == 0:
        folding_flag = True
        if layout_flag == "horizontal":
            fig_setting = canvas_setting(8*len(components)/2, 12) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(2, int(len(components)/2), figsize=fig_setting[0], dpi=fig_setting[1])
            nrows, ncols = np.array(axs).shape if hasattr(axs, "shape") else (1, 1)
            # axes_element = [axs[i, j] for j in range(int(len(components)/2)) for i in range(2)]
            axes_element = [axs[i, j] for i in range(2) for j in range(int(len(components)/2))]
        else:
            fig_setting = canvas_setting(16, 6*len(components)/2+1) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(int(len(components)/2), 2, figsize=fig_setting[0], dpi=fig_setting[1])
            nrows, ncols = np.array(axs).shape if hasattr(axs, "shape") else (1, 1)
            axes_element = [axs[i, j] for j in range(2) for i in range(int(len(components)/2))]
    elif expansion_flag is False and len(components) == 9:
        allcomps_flag = True
        if layout_flag == "horizontal":
            fig_setting = canvas_setting(24, 18) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(3, 3, figsize=fig_setting[0], dpi=fig_setting[1])
            nrows, ncols = np.array(axs).shape if hasattr(axs, "shape") else (1, 1)
            axes_element = [axs[i, j] for i in range(3) for j in range(3)]
        else:
            fig_setting = canvas_setting(24, 18) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(3, 3, figsize=fig_setting[0], dpi=fig_setting[1])
            nrows, ncols = np.array(axs).shape if hasattr(axs, "shape") else (1, 1)
            axes_element = [axs[i, j] for j in range(3) for i in range(3)]
    else:
        if layout_flag == "horizontal":
            fig_setting = canvas_setting(8*len(components), 6) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(1, len(components), figsize=fig_setting[0], dpi=fig_setting[1])
            nrows, ncols = np.array(axs).shape if hasattr(axs, "shape") else (1, 1)
            axes_element = [axs[i] for i in range(len(components))]
        else:
            fig_setting = canvas_setting(10, 6*len(components)) if figure_size == (None, None) else canvas_setting(figure_size[0], figure_size[1])
            params = fig_setting[2]
            plt.rcParams.update(params)
            fig, axs = plt.subplots(len(components), 1, figsize=fig_setting[0], dpi=fig_setting[1])
            nrows, ncols = np.array(axs).shape if hasattr(axs, "shape") else (1, 1)
            axes_element = [axs[i] for i in range(len(components))]

    ## identify x-axis unit
    var_label = "wavelength" if unit and unit.lower() == "nm" else "energy"
    xaxis_label = "Photon wavelength (nm)" if var_label == "wavelength" else "Photon energy (eV)"

    ## systems information
    dataset = dielectric_systems_list(systems)

    ## suptitle
    fig.suptitle(f"{suptitle}\n", fontsize=fig_setting[3][0])

    ## data boundary
    photon_start, photon_end = process_boundary_alt(x_boundary)

    ## data plotting
    # for each subplot
    if expansion_flag is True:
        for subplot_index in range(2*len(components)):
            ax = axes_element[subplot_index]
            ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

            # current component index and label
            component_index = subplot_index // 2
            current_component = comp_labels[component_index].lower()

            data_key = f"density_{current_component}_real" if subplot_index % 2 == 0 else f"density_{current_component}_imag"

            ## subtitles and axis label (self-assertive)
            # subtitles
            ax.set_title([f"Real part for {comp_aliases[component_index]}", f"Imaginary part for {comp_aliases[component_index]}"][subplot_index%2])
            # ylabel
            if layout_flag == "vertical" and subplot_index%2 == 0:
                ax.set_ylabel("Dielectric function",fontsize=)
            elif layout_flag == "horizontal" and subplot_index in range(2):
                ax.set_ylabel("Dielectric function")
            # xlabel
            if layout_flag == "vertical" and subplot_index >= 2*len(components)-2:
                ax.set_xlabel(xaxis_label)
            elif layout_flag == "horizontal" and subplot_index%2 == 1:
                ax.set_xlabel(xaxis_label)

            # initialization
            wavelength_starts, wavelength_ends, energy_starts, energy_ends = [], [], [], []

            # curve plotting: real part
            if subplot_index%2 == 0:
                # for each system
                for _, data in enumerate(dataset):
                    supercell_thickness, system_thickness = data[6]
                    d_ratio = supercell_thickness/system_thickness
                    energy_real, density_energy_real_source = extract_part(data[1]["density_energy_real"], data[1][data_key], photon_start, photon_end)
                    density_energy_real = density_energy_real_source * d_ratio - d_ratio + 1
                    energy_real, density_energy_real = mask_real(energy_real, density_energy_real, None)
                    if var_label == "energy":
                        # ax.plot(energy_real, density_energy_real, color=color_sampling(data[2])[1], ls=data[3], lw=data[4], alpha=data[5], label=f"Real part {data[0]}")
                        ax.plot(energy_real, density_energy_real, color=color_sampling(data[2])[1], ls=data[3], lw=data[4], alpha=data[5], label=f"{data[0]}")
                        # plasmon resonance line for photon energy
                        energy_starts.append(min(energy_real))
                        energy_ends.append(max(energy_real))

                    else:
                        wavelength_real, density_wl_real = extract_part(energy_to_wavelength(data[1]["density_energy_real"]),data[1][data_key], photon_start, photon_end)
                        # ax.plot(wavelength_real, density_wl_real, color=color_sampling(data[2])[1], ls=data[3], lw=data[4], alpha=data[5], label=f"Real part {data[0]}")
                        ax.plot(wavelength_real, density_wl_real, color=color_sampling(data[2])[1], ls=data[3], lw=data[4], alpha=data[5], label=f"{data[0]}")
                        # plasmon resonance line for photon wavelength
                        wavelength_starts.append(min(wavelength_real))
                        wavelength_ends.append(np.max(np.array(wavelength_real)[np.isfinite(wavelength_real)]))
                # plasmon resonance line
                if var_label == "energy":
                    energy_start=min(energy_starts)
                    energy_end=max(energy_ends)
                    ax.plot([energy_start, energy_end],[0,0],color=color_sampling("grey")[1],linestyle="--")
                else:
                    wavelength_start=min(wavelength_starts)
                    wavelength_end=max(wavelength_ends)
                    ax.plot([wavelength_start, wavelength_end],[0,0],color=color_sampling("grey")[1],linestyle="--")

            # curve plotting: imaginary part
            else:
                for _, data in enumerate(dataset):
                    supercell_thickness, system_thickness = data[6]
                    d_ratio = supercell_thickness/system_thickness
                    energy_imag, density_energy_imag_source = extract_part(data[1]["density_energy_imag"], data[1][data_key], photon_start, photon_end)
                    density_energy_imag= density_energy_imag_source * d_ratio
                    energy_imag, density_energy_imag = mask_imag(energy_imag, density_energy_imag, None)
                    if var_label == "energy":
                        # ax.plot(energy_imag, density_energy_imag, color=color_sampling(data[2])[2], ls=data[3], lw=data[4], alpha=data[5], label=f"Imaginary part {data[0]}")
                        ax.plot(energy_imag, density_energy_imag, color=color_sampling(data[2])[2], ls=data[3], lw=data[4], alpha=data[5], label=f"{data[0]}")
                    else:
                        wavelength_imag, density_wl_imag = extract_part(energy_to_wavelength(data[1]["density_energy_imag"]), data[1][data_key], photon_start, photon_end)
                        # ax.plot(wavelength_imag, density_wl_imag, color=color_sampling(data[2])[2], ls=data[3], lw=data[4], alpha=data[5], label=f"Imaginary part {data[0]}")
                        ax.plot(wavelength_imag, density_wl_imag, color=color_sampling(data[2])[2], ls=data[3], lw=data[4], alpha=data[5], label=f"{data[0]}")

            # y boundary
            y_min, y_max = ax.get_ylim()
            # print(y_min, y_max)
            y_low, y_hig = process_boundary_alt(y_boundary)
            y_sup = y_max if y_hig is None else min(y_hig, y_max)
            if subplot_index%2 == 0:
                y_inf = y_min if y_low is None else max(y_low, y_min)
            else: y_inf = -(y_sup*0.05)
            ax.set_ylim(y_inf, y_sup)

            # Spectrum
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(xmin, xmax)
            wl_vis = np.linspace(380, 750, 1000)        # 380 nm (violet) to 750 nm (red)
            ev_vis = wavelength_to_energy(wl_vis)       # 1.65 eV (red) to 3.26 eV (violet)
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

            # Legend
            ax.legend(loc="best")
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)
    else:
        for subplot_index in range(len(components)):
            ax = axes_element[subplot_index]
            ax.tick_params(direction="in", which="both", top=True, right=True, bottom=True, left=True)

            # initialization
            wavelength_starts, wavelength_ends, energy_starts, energy_ends = [], [], [], []

            # current component index and label
            component_index = subplot_index
            current_component = comp_labels[component_index].lower()
            data_key_real = f"density_{current_component}_real"
            data_key_imag = f"density_{current_component}_imag"
            energy_real, density_energy_real, energy_imag, density_energy_imag = mask_real_imag(energy_real, density_energy_real, energy_imag, density_energy_imag, None)
            # curve plotting: real part and imaginary part
            for _, data in enumerate(dataset):
                supercell_thickness, system_thickness = data[6]
                d_ratio = supercell_thickness/system_thickness
                energy_real, density_energy_real_source = extract_part(data[1]["density_energy_real"], data[1][data_key_real], photon_start, photon_end)
                energy_imag, density_energy_imag_source = extract_part(data[1]["density_energy_imag"], data[1][data_key_imag], photon_start, photon_end)
                density_energy_real = density_energy_real_source * d_ratio - d_ratio + 1
                density_energy_imag= density_energy_imag_source * d_ratio
                energy_real, density_energy_real, energy_imag, density_energy_imag = mask_real_imag(energy_real, density_energy_real, energy_imag, density_energy_imag, None)
                if var_label == "energy":
                    ax.plot(energy_real, density_energy_real, color=color_sampling(data[2])[1], ls=data[3], lw=data[4], alpha=data[5], label=f"Real part {data[0]}")
                    ax.plot(energy_imag, density_energy_imag, color=color_sampling(data[2])[1], ls="dashed", lw=data[4], alpha=data[5], label=f"Imaginary part {data[0]}")
                    energy_starts.append(min(energy_real))
                    energy_ends.append(max(energy_real))
                else:
                    wavelength_real, density_wl_real = extract_part(energy_to_wavelength(data[1]["density_energy_real"]), data[1][data_key_real], photon_start, photon_end)
                    wavelength_imag, density_wl_imag = extract_part(energy_to_wavelength(data[1]["density_energy_imag"]), data[1][data_key_imag], photon_start, photon_end)
                    ax.plot(wavelength_real, density_wl_real, color=color_sampling(data[2])[1], ls=data[3], lw=data[4], alpha=data[5], label=f"Real part {data[0]}")
                    ax.plot(wavelength_imag, density_wl_imag, color=color_sampling(data[2])[1], ls="dashed", lw=data[4], alpha=data[5], label=f"Imaginary part {data[0]}")
                    wavelength_starts.append(min(wavelength_real))
                    wavelength_ends.append(np.max(np.array(wavelength_real)[np.isfinite(wavelength_real)]))

            # plasmon resonance line and rescale rate
            if var_label == "energy":
                plasmon_start = min(energy_starts)
                plasmon_end = max(energy_ends)
                ax.plot([plasmon_start, plasmon_end],[0,0], color=color_sampling("grey")[1], linestyle="dashed")
            else:
                plasmon_start=min(wavelength_starts)
                plasmon_end=max(wavelength_ends)
                ax.plot([plasmon_start, plasmon_end],[0,0],color=color_sampling("grey")[1],linestyle="dashed")

            # y boundary
            y_min, y_max = ax.get_ylim()
            # print(y_min, y_max)
            y_low, y_hig = process_boundary_alt(y_boundary)
            y_inf = y_min if y_low is None else max(y_low, y_min)
            y_sup = y_max if y_hig is None else min(y_hig, y_max)
            ax.set_ylim(y_inf, y_sup)

            # Spectrum
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(xmin, xmax)
            wl_vis = np.linspace(380, 750, 1000)        # 380 nm (violet) to 750 nm (red)
            ev_vis = wavelength_to_energy(wl_vis)       # 1.65 eV (red) to 3.26 eV (violet)
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

            # subtitles and axis label (self-assertive): subtitles
            ax.set_title(comp_aliases[component_index])
            if allcomps_flag is True and layout_flag == "horizontal":
                if subplot_index in [0, len(components)/3, 2*len(components)/3]:
                    ax.set_ylabel("Dielectric function",fontsize=50)
                if subplot_index >= 2*len(components)/3:
                    ax.set_xlabel(xaxis_label)
            elif allcomps_flag is True and layout_flag == "vertical":
                if subplot_index < len(components)/3:
                    ax.set_ylabel("Dielectric function")
                if subplot_index in [len(components)/3-1, 2*len(components)/3-1, len(components)-1]:
                    ax.set_xlabel(xaxis_label)
            elif folding_flag is True and layout_flag == "horizontal":
                if subplot_index in [0, len(components)/2]:
                    ax.set_ylabel("Dielectric function")
                if subplot_index >= len(components)/2:
                    ax.set_xlabel(xaxis_label)
            elif folding_flag is True and layout_flag == "vertical":
                if subplot_index < len(components)/2:
                    ax.set_ylabel("Dielectric function")
                if subplot_index in [len(components)/2-1, len(components)-1]:
                    ax.set_xlabel(xaxis_label)
            elif layout_flag == "vertical":
                ax.set_ylabel("Dielectric function")
                if layout_flag == "vertical" and subplot_index == len(components)-1:
                    ax.set_xlabel(xaxis_label)
            else:
                ax.set_xlabel(xaxis_label)
                if subplot_index == 0:
                    ax.set_ylabel("Dielectric function")

            ax.legend(loc="best")
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-3,3), useOffset=False, useMathText=True)

    plt.tight_layout()
