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


# %% Original dielectric tensor figure: three columns, real above imaginary.
data = {folder: dielectric(folder, factor) for folder, _, _, factor in systems}
fig, axes = plt.subplots(2, 3, figsize=(24, 12))
for row, kind in enumerate(('real', 'imag')):
    for col, component in enumerate(('xx', 'yy', 'zz')):
        ax = axes[row, col]
        for folder, label, color, factor in systems:
            energy, real, imag = data[folder]
            chosen = (energy >= 0) & (energy <= 12)
            ax.plot(energy[chosen], quantity(energy, real, imag, kind)[chosen, col], color=color, label=label)
        if kind == 'real':
            lo, hi = ax.get_ylim(); ax.set_ylim(max(-60, lo), min(60, hi))
        else:
            hi = min(60, ax.get_ylim()[1]); ax.set_ylim(-.05*hi, hi)
        ax.set_xlim(0, 12); frame(ax)
        ax.set_title(f'{"Real" if row == 0 else "Imaginary"} part for {component}-component')
        if col == 0: ax.set_ylabel('Dielectric function', fontsize=20)
        if row == 1: ax.set_xlabel('Photon energy (eV)', fontsize=18)
        ax.legend(loc='best')
fig.suptitle('Dielectric function', fontsize=20)
fig.subplots_adjust(left=.055, right=.985, bottom=.08, top=.91, wspace=.18, hspace=.28)
save(fig, 'fig2.10.pdf')

# %% Each original 24 x 6, three-panel optical figure becomes a 16 x 12 grid.
for name, kind, title, ylabel in [
    ('fig2.11a.pdf', 'alpha', 'Absorption coefficient', r'Absorption coefficient ($\mathrm{\AA}^{-1}$)'),
    ('fig2.11b.pdf', 'loss', 'Energy-loss spectrum', 'Energy-loss spectrum'),
    ('fig2.12a.pdf', 'R', 'Reflectivity', 'Reflectivity'),
    ('fig2.12b.pdf', 'n', 'Refractive index', 'Refractive index'),
    ('S2.18.pdf', 'k', 'Extinction coefficient', 'Extinction coefficient')]:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for col, ax in enumerate(axes.flat[:3]):
        for folder, label, color, factor in systems:
            energy, real, imag = data[folder]
            chosen = (energy >= 0) & (energy <= 12)
            ax.plot(energy[chosen], quantity(energy, real, imag, kind)[chosen, col], color=color, label=label)
        ax.set(xlim=(0, 12), xlabel='Photon energy (eV)', ylabel=ylabel,
               title=f'({"abc"[col]}) {("xx", "yy", "zz")[col]}-component')
        frame(ax)
    axes[1, 1].axis('off')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 1].legend(handles, labels, loc='center')
    fig.suptitle(title, fontsize=20)
    fig.subplots_adjust(left=.08, right=.97, bottom=.07, top=.90, wspace=.22, hspace=.25)
    save(fig, name)

# %% Original 24 x 12 dielectric convergence figures.
for name, folders, labels, limits, title in [
 ('S2.4.pdf', [f'o-B14_n128_k{k}' for k in (10,20,26,30,32,34)],
  ['10×14×9', '20×28×18', '26×37×24', '30×42×28', '32×45×29', '34×48×31'],
  [(1,8),(5,15),(0,4)], 'Dielectric function versus k-points for bulk o-B$_{14}$'),
 ('S2.5.pdf', [f'o-B14_n{k}_k10' for k in (32,64,128,256)],
  ['NBANDS = 48', 'NBANDS = 72', 'NBANDS = 144', 'NBANDS = 264'],
  [(15,25),(15,30),(10,30)], 'Dielectric function versus NBANDS for bulk o-B$_{14}$')]:
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    for folder, label, color in zip(folders, labels, [GREEN, '#E65050', ORANGE, YELLOW, CYAN, BLUE]):
        energy, real, imag = dielectric(folder)
        for row, values in enumerate((real, imag)):
            for col in range(3): axes[row, col].plot(energy, values[:, col], color=color, label=label)
    for i, ax in enumerate(axes.flat):
        row, col = divmod(i, 3)
        ax.set_xlim(limits[col]); frame(ax)
        ax.set_title(f'{"Real" if row == 0 else "Imaginary"} part for {("xx","yy","zz")[col]}-component')
        if col == 0: ax.set_ylabel('Dielectric function', fontsize=20)
        if row == 1: ax.set_xlabel('Photon energy (eV)', fontsize=18)
        shown = [line.get_ydata()[(line.get_xdata() >= limits[col][0]) & (line.get_xdata() <= limits[col][1])] for line in ax.lines]
        lo, hi = min(a.min() for a in shown), max(a.max() for a in shown)
        ax.set_ylim(lo-.08*(hi-lo), hi+.08*(hi-lo))
        ax.legend(loc='best')
    fig.suptitle(title, fontsize=20)
    fig.subplots_adjust(left=.055, right=.985, bottom=.08, top=.91, wspace=.18, hspace=.28)
    save(fig, name)
