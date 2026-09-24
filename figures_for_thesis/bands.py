"""Original paper band/DOS layouts using the verified numerical readers."""
from electronic_data import *

# %% Three original 9 x 6 panels, arranged as 2 x 2 with one shared legend.
fig, grid = plt.subplots(2, 2, figsize=(18, 12))
for ax, folder, title in zip(grid.flat[:3],
        ['monolayer_FM_HSE06', 'monolayer_AFM_HSE06', 'bilayer_HSE'],
        ['(a) Band structure for FM monolayer o-B$_{14}$',
         '(b) Band structure for AFM monolayer o-B$_{14}$',
         '(c) Band structure for bilayer o-B$_{14}$']):
    draw_bands(ax, folder, spin=folder!='bilayer_HSE')
    ax.set_ylabel('Energy (eV)'); ax.set_title(title)
grid[1, 1].axis('off')
handles = [Line2D([], [], color=PURPLE, label='Spin up'),
           Line2D([], [], color=CYAN, ls=(0, (4, 3)), label='Spin down'),
           Line2D([], [], color=BLUE, label='Bilayer bands'),
           Line2D([], [], color='#5A3C8C', ls='--', label='Fermi energy')]
grid[1, 1].legend(handles=handles, loc='center')
fig.subplots_adjust(left=.07, right=.97, bottom=.06, top=.94, wspace=.18, hspace=.22)
save(fig, 'fig2.6.pdf')

# %% Original 9 x 6 panels, side by side.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, folder, method in zip(axes, ['monolayer_FM', 'monolayer_FM_HSE06'], ['GGA-PBE', 'HSE06']):
    draw_bands(ax, folder, spin=True)
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Band structure for FM monolayer o-B$_{14}$')
    ax.text(.03, .96, method, transform=ax.transAxes, va='top', fontsize=16,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=.75,
                      edgecolor='#B4B4B4', linewidth=1.5))
    ax.legend(handles=handles[:2]+handles[-1:], loc='upper right')
fig.subplots_adjust(left=.06, right=.97, bottom=.10, top=.88, wspace=.18)
save(fig, 'S2.11.pdf')

# %% Original single-plot canvases and internal legends.
for filename, folders, labels, title in [
 ('S2.13.pdf', ['monolayer', 'monolayer_HSE', 'monolayer_R2SCAN'],
  ['GGA-PBE', 'HSE06', 'R2SCAN'], 'Band structure for monolayer o-B$_{14}$'),
 ('S2.14.pdf', ['monolayer', 'monolayer_shifting', 'monolayer_sym_off'],
  ['Bands of GGA-PBE', 'Bands of shifted atoms', 'Bands with symmetry off'],
  'Band structure for monolayer o-B$_{14}$ for symmetry testing')]:
    fig, ax = plt.subplots(figsize=(10, 6))
    for folder, label, color in zip(folders, labels, [BLUE, ORANGE, '#8CAF28']):
        draw_bands(ax, folder, color=color, label=label)
    ax.set_ylim(-4, 4); ax.set_ylabel('Energy (eV)'); ax.set_title(title)
    lines, names = ax.get_legend_handles_labels()
    ax.legend(lines+handles[-1:], names+['Fermi energy'], loc='upper right')
    fig.subplots_adjust(left=.10, right=.97, bottom=.10, top=.89)
    save(fig, filename)

# %% Original 12 x 6 FM and AFM plots, stacked without equalizing their printed font size.
fig, axes = plt.subplots(2, 1, figsize=(12, 12))
for ax, folder, order in zip(axes, ['monolayer_FM_ollie', 'monolayer_AFM_ollie'], ['FM', 'AFM']):
    energy, channels, _ = dos(folder)
    ax.plot(energy, channels[0]+channels[1], color=BLUE, label='Total')
    ax.plot(energy, channels[0], color=ORANGE, label='Spin up')
    ax.plot(energy, -channels[1], color=CYAN, label='Spin down')
    ax.axvline(0, color='#5A3C8C', ls='--')
    ax.set(xlim=(-14, 6), ylim=(-8, 15), xlabel='Energy (eV)',
           ylabel='Density of States (states/eV)',
           title=f'Spin-polarized DoS of {order} monolayer o-B$_{{14}}$')
    ax.legend(loc='upper right'); frame(ax)
fig.subplots_adjust(left=.10, right=.97, bottom=.06, top=.95, hspace=.30)
save(fig, 'S2.16.pdf')

# %% Total DOS, original 10 x 6 canvas.
fig, ax = plt.subplots(figsize=(10, 6))
for folder, label, color in [('o-B14_K20', 'Bulk', BLUE), ('monolayer', 'Monolayer', GREEN),
        ('bilayer', 'Bilayer', YELLOW), ('bilayer_with_Hydrogen', 'H-terminated bilayer', ORANGE)]:
    energy, channels, _ = dos(folder)
    ax.plot(energy, channels[0], color=color, label=label)
ax.axvline(0, color='#5A3C8C', ls='--')
ax.set(xlim=(-6, 6), ylim=(0, 27), xlabel='Energy (eV)', ylabel='Density of States',
       title='Total DoS for o-B$_{14}$ systems')
frame(ax); ax.legend(loc='upper right')
fig.subplots_adjust(left=.10, right=.97, bottom=.13, top=.89)
save(fig, 'S2.12.pdf')

# %% Original 12 x 6 band/DOS layout with 3:1 widths.
fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]}, sharey=True)
draw_bands(axes[0], 'bilayer_with_Hydrogen'); axes[0].set_ylim(-4, 4)
energy, channels, _ = dos('bilayer_with_Hydrogen')
axes[1].plot(channels[0], energy, color=BLUE)
axes[1].axhline(0, color='#5A3C8C', ls='--'); axes[1].set_xlim(0, 20)
axes[0].set_ylabel('Energy (eV)')
axes[0].set_title('Band structure', fontsize=18); axes[1].set_title('DoS (a.u.)', fontsize=18)
frame(axes[1])
fig.suptitle('Band structure and DoS for bilayer o-B$_{14}$ with hydrogen termination', fontsize=20)
fig.subplots_adjust(left=.09, right=.96, bottom=.10, top=.84, wspace=.08)
save(fig, 'fig2.8.pdf')
