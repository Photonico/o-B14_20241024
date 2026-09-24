"""Convergence plots from archived tabular postprocessing data."""
import csv, sys
import numpy as np
from style import *
sys.path.insert(0,str(ROOT))
from vmatplot.algorithms import fit_birch_murnaghan
base=ROOT/'1.0_energy/o-B14'


def table(folder,name='energy_parameters.dat'):
    with open(base/folder/name) as f: rows=list(csv.DictReader(f,delimiter='\t'))
    aliases={'kpoints(x y z)':'kpoints mesh','encut':'energy cutoff (encut)'}
    return [{aliases.get(k.lower(),k.lower()):v for k,v in row.items()} for row in rows]


def series(folder,xkey='total kpoints',ykey='total energy',lower=40,name='energy_parameters.dat'):
    rows=table(folder,name)
    rows=sorted([r for r in rows if float(r[xkey])>=lower],key=lambda r:float(r[xkey]))
    return np.array([float(r[xkey]) for r in rows]),np.array([float(r[ykey]) for r in rows]),rows


# %% Dual-x convergence plot and lattice scan, side by side.
fig, axes = plt.subplots(1, 2, figsize=(20, 7.5))
ax = axes[0]; upper = ax.twiny()
for cutoff, color in [(450, GREEN), (480, BLUE)]:
    x, y, rows = series(f'energy_kpoints_{cutoff}')
    nk = np.array([int(r['kpoints mesh'].strip('()').split(',')[0]) for r in rows])
    ax.plot(nk, y, 'o-', ms=4, color=color, label=rf'$E_{{\rm cut}}={cutoff}$ eV')
for grid, label, color in [('1260', r'$10\times14\times9$', PURPLE),
                          ('8721', r'$19\times27\times17$', '#D25ADC')]:
    x, y, _ = series(f'energy_encut_{grid}', 'energy cutoff (encut)', lower=400)
    upper.plot(x, y, 'o-', ms=4, color=color, label=label)
ax.set_xlabel(r'Grid index ($n_x$)', color=BLUE)
ax.set_xticks([4, 8, 12, 16, 20, 24]); ax.set_ylabel('Energy (eV)')
upper.set_xlabel('Energy cutoff (eV)', color=PURPLE)
upper.tick_params(direction='in', colors=PURPLE)
ax.tick_params(direction='in', right=True, colors='black')
ax.tick_params(axis='x', colors=BLUE)
ax.ticklabel_format(axis='y', style='plain', useOffset=False)
ax.set_title('(a) Energy for bulk o-B$_{14}$', pad=56)
lines = ax.get_lines()+upper.get_lines()
ax.legend(lines, [line.get_label() for line in lines], loc='upper right')
x, y, _ = series('energy_lattice', 'lattice constant', lower=0)
params, xx, yy = fit_birch_murnaghan(x, y, sample_count=100)
ax = axes[1]
ax.plot(xx, yy, color=BLUE, label='EOS fit')
ax.plot(x, y, 'o', ms=4, color=BLUE, label='Sampled data')
imin = np.argmin(y)
ax.axvline(x[imin], color=GREY, ls='--', label=rf'Minimum: {x[imin]:.5f} $\mathrm{{\AA}}$')
ax.set(xlabel=r'Lattice constant ($\mathrm{\AA}$)', ylabel='Energy (eV)')
ax.set_title('(b) Energy versus lattice constant for bulk o-B$_{14}$', pad=56)
ax.ticklabel_format(axis='both', style='plain', useOffset=False)
frame(ax); ax.legend(loc='upper right')
fig.subplots_adjust(left=.07, right=.97, bottom=.14, top=.82, wspace=.25)
save(fig, 'S2.1.pdf')

# %% Original 10 x 7.5 cohesive-energy canvas.
fig,ax=plt.subplots(figsize=(10,7.5))
upper=ax.twiny()
for axis,folder,xkey,color,lower,label in [
 (ax,'energy_kpoints_480','total kpoints',BLUE,40,r'k-grid, $E_{\rm cut}=480$ eV'),
 (upper,'energy_encut_1260','energy cutoff (encut)',PURPLE,400,r'Cutoff, $10\times14\times9$')]:
    rows=table(folder,'cohesive_energy.dat')
    ykey=next(k for k in rows[0] if 'cohesive' in k.lower())
    x,y,rows=series(folder,xkey,ykey,lower,'cohesive_energy.dat')
    if xkey=='total kpoints': x=np.array([int(r['kpoints mesh'].strip('()').split(',')[0]) for r in rows])
    axis.plot(x,y,'o-',ms=4,color=color,label=label)
    axis.ticklabel_format(axis='y',style='plain',useOffset=False)
ax.tick_params(direction='in',right=True,top=False)
upper.tick_params(axis='x',direction='in',colors=PURPLE)
ax.tick_params(axis='x',colors=BLUE)
ax.set_xticks([4,8,12,16,20,24]);ax.set_xlabel(r'Grid index ($n_x$)',color=BLUE)
upper.set_xticks([400,600,800,1000,1200]);upper.set_xlabel('Energy cutoff (eV)',color=PURPLE)
ax.set_ylabel('Cohesive energy (eV/atom)')
handles=ax.get_lines()+upper.get_lines()
ax.legend(handles,[line.get_label() for line in handles],loc='upper right',
           frameon=True,fancybox=True,borderpad=.25,labelspacing=.25)
ax.set_title('Cohesive energy for bulk o-B$_{14}$', pad=56)
fig.subplots_adjust(left=.13,right=.97,bottom=.13,top=.79)
save(fig,'S2.2.pdf')

# %% Original single energy-versus-height canvas.
with open(ROOT/'2.1_geometry_optimization/monolayer/energy_vacuum/energy_parameters.dat') as f:
    rows=list(csv.DictReader(f,delimiter='\t'))
x=np.array([float(r['a3']) for r in rows]); y=np.array([float(r['total energy']) for r in rows])
fig,ax=plt.subplots(figsize=(10,6));ax.plot(x,y,'o-',color=BLUE,ms=4)
ax.set(xlabel=r'Supercell height ($a_3$, $\mathrm{\AA}$)',ylabel='Energy (eV)')
ax.ticklabel_format(axis='y',style='plain',useOffset=False);frame(ax)
ax.set_title('Energy versus supercell height for monolayer o-B$_{14}$')
fig.subplots_adjust(left=.13,right=.97,bottom=.14,top=.89);save(fig,'S2.3.pdf')

# %% Fixed-spin energies.
# This supplied tabulation is the source of the thesis fixed-spin figure.
with open(OUT/'fixed_spin_data.csv') as f: rows=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(10,6))
for method,color,label in [('PBE+D3',BLUE,'PBE+D3'),('r2SCAN',ORANGE,r'r$^2$SCAN')]:
    selected=[r for r in rows if r['method']==method]
    ax.plot([float(r['fixed_spin_muB']) for r in selected],
            [float(r['relative_energy_eV']) for r in selected],'o-',ms=4,color=color,label=label)
ax.set(xlabel=r'Fixed spin moment ($\mu_{\mathrm{B}}$)',ylabel='Relative energy (eV)')
frame(ax);ax.legend(loc='upper left',ncol=1,frameon=True,fancybox=True)
ax.set_title('Fixed-spin energy profile')
fig.subplots_adjust(left=.11,right=.97,bottom=.14,top=.89)
save(fig,'S2.15.pdf')
