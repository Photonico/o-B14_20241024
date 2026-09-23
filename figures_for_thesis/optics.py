"""Read the original dielectric HDF5 arrays and export the grouped optical figures."""
import h5py
import numpy as np
from style import *

systems = [('bilayer_with_Hydrogen', 'H-terminated bilayer', ORANGE, 30.088116646 / 11.208620),
           ('bilayer', 'Bilayer', YELLOW, 28.088116646 / 9.41157),
           ('monolayer', 'Monolayer', GREEN, 24.044058323 / 5.807056948202938),
           ('o-B14_n128_k34', 'Bulk', BLUE, 1.)]


def dielectric(folder, factor=1.):
    with h5py.File(ROOT / '5.1_dielectric_function' / folder / 'vaspout.h5') as f:
        group = f['results/linear_response']
        energy = group['energies_dielectric_function'][:]
        tensor = group['density_density_dielectric_function'][:]
    real = np.array([tensor[i, i, :, 0] for i in range(3)]).T
    imag = np.array([tensor[i, i, :, 1] for i in range(3)]).T
    return energy, 1 + factor * (real - 1), factor * imag


def quantity(energy, real, imag, kind):
    modulus = np.hypot(real, imag)
    n = np.sqrt(np.maximum(0, (modulus + real) / 2))
    k = np.sqrt(np.maximum(0, (modulus - real) / 2))
    return {'real': real, 'imag': imag, 'n': n, 'k': k,
            'alpha': 2 * energy[:, None] * k / (6.582119569e-16 * 2.99792458e18),
            'loss': imag / (real**2 + imag**2),
            'R': ((n - 1)**2 + k**2) / ((n + 1)**2 + k**2)}[kind]


def optical_figure(name, rows, labels):
    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.0 * len(rows) + 1.0), squeeze=False)
    for row, (kind, ylabel) in enumerate(rows):
        for col, component in enumerate(('xx', 'yy', 'zz')):
            ax = axes[row, col]
            for folder, label, color, factor in systems:
                energy, real, imag = data[folder]
                selected = (energy >= 0) & (energy <= 12)
                ax.plot(energy[selected], quantity(energy, real, imag, kind)[selected, col],
                        color=color, label=label)
            if kind == 'real':
                lo, hi = ax.get_ylim(); ax.set_ylim(max(-60, lo), min(60, hi))
            elif kind == 'imag':
                hi = min(60, ax.get_ylim()[1]); ax.set_ylim(-.05*hi, hi)
            ax.set_xlim(0, 12)
            ax.set_xticks([0, 4, 8, 12])
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            frame(ax)
            tab(ax, f'({labels[row]}) {component}' if col == 0 else component)
            if col == 0: ax.set_ylabel(ylabel)
            if row == len(rows) - 1: ax.set_xlabel('Photon energy (eV)')
            else: ax.tick_params(labelbottom=False)
    legend(fig, axes[0, 0])
    fig.subplots_adjust(left=.11, right=.985, bottom=.32 if len(rows) == 1 else .23,
                        top=.95, wspace=.27, hspace=.13)
    save(fig, name)


data = {folder: dielectric(folder, factor) for folder, _, _, factor in systems}
optical_figure('fig2.10.pdf', [('real', r'$\varepsilon_1$'), ('imag', r'$\varepsilon_2$')], ['a', 'b'])
optical_figure('fig2.11.pdf', [('alpha', r'Absorption ($\mathrm{\AA}^{-1}$)'),
                            ('loss', r'Energy loss')], ['a', 'b'])
optical_figure('fig2.12.pdf', [('R', 'Reflectivity'), ('n', 'Refractive index')], ['a', 'b'])
optical_figure('S2.18.pdf', [('k', 'Extinction coefficient')], ['a'])

# Convergence uses the original density-density arrays, without thickness scaling.
for name, folders, labels, limits in [
 ('S2.4.pdf', [f'o-B14_n128_k{k}' for k in (10,20,26,30,32,34)],
  ['10×14×9', '20×28×18', '26×37×24', '30×42×28', '32×45×29', '34×48×31'], [(1,8),(5,15),(0,4)]),
 ('S2.5.pdf', [f'o-B14_n{k}_k10' for k in (32,64,128,256)],
  ['48 bands', '72 bands', '144 bands', '264 bands'], [(15,25),(15,30),(10,30)])]:
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for folder, label, color in zip(folders, labels, [GREEN, '#E65050', ORANGE, YELLOW, CYAN, BLUE]):
        energy, real, imag = dielectric(folder)
        for row, values in enumerate((real, imag)):
            for col in range(3):
                axes[row,col].plot(energy, values[:,col], color=color, label=label)
    for i, ax in enumerate(axes.flat):
        row, col = divmod(i,3)
        ax.set_xlim(limits[col]); frame(ax)
        tab(ax, ('xx','yy','zz')[col])
        if col == 0: ax.set_ylabel(r'$\varepsilon_1$' if row == 0 else r'$\varepsilon_2$')
        if row == 1: ax.set_xlabel('Photon energy (eV)')
        else: ax.tick_params(labelbottom=False)
        # Autoscale to the displayed energy interval, including all plotted datasets.
        shown = [line.get_ydata()[(line.get_xdata() >= limits[col][0]) & (line.get_xdata() <= limits[col][1])]
                 for line in ax.lines]
        lo, hi = min(a.min() for a in shown), max(a.max() for a in shown)
        pad = .08 * (hi - lo)
        ax.set_ylim(lo-pad, hi+pad)
    legend(fig, axes[0,0], ncol=3)
    fig.subplots_adjust(left=.1,right=.985,bottom=.23,top=.95,wspace=.27,hspace=.13)
    save(fig,name)
