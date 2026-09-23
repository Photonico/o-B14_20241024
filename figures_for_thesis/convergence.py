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


fig,grid=plt.subplots(2,2,figsize=(8,6))
axes=grid.flat[:3];legend_ax=grid[1,1];legend_ax.axis('off')
fig.subplots_adjust(left=.13,right=.988,bottom=.105,top=.96,wspace=.25,hspace=.27)
for cutoff,color in [(450,GREEN),(480,BLUE)]:
    x,y,rows=series(f'energy_kpoints_{cutoff}')
    # Plot the actual first Monkhorst-Pack grid integer; the full meshes remain in the source table.
    nk=np.array([int(r['kpoints mesh'].strip('()').split(',')[0]) for r in rows])
    axes[0].plot(nk,y,'o-',ms=4,color=color,label=rf'$E_{{\rm cut}}={cutoff}$ eV')
for grid,label,color in [('1260',r'$10\times14\times9$',PURPLE),('8721',r'$19\times27\times17$','#D25ADC')]:
    x,y,_=series(f'energy_encut_{grid}','energy cutoff (encut)',lower=400)
    axes[1].plot(x,y,'o-',ms=3,color=color,label=label)
for i,ax in enumerate(axes[:2]):
    ax.set_ylabel('Energy (eV)'); ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    frame(ax);tab(ax,['(a) k-point grid','(b) Energy cutoff'][i])
axes[0].set_xlabel(r'Grid index ($n_x$)');axes[0].set_xticks([4,8,12,16,20,24])
axes[1].set_xlabel('Energy cutoff (eV)')
axes[1].set_ylabel('')
x,y,_=series('energy_lattice','lattice constant',lower=0)
params,xx,yy=fit_birch_murnaghan(x,y,sample_count=100)
axes[2].plot(xx,yy,color=BLUE,label='EOS fit')
axes[2].plot(x,y,'o',ms=4,color=BLUE,label='Sampled data')
imin=np.argmin(y);axes[2].axvline(x[imin],color=GREY,ls='--',label=rf'Minimum: {x[imin]:.5f} $\mathrm{{\AA}}$')
axes[2].set(xlabel=r'Lattice constant ($\mathrm{\AA}$)',ylabel='Energy (eV)')
axes[2].ticklabel_format(axis='both',style='plain',useOffset=False)
axes[2].set_xticks([5.685,5.695,5.705,5.715])
frame(axes[2]);tab(axes[2],'(c) Lattice constant',loc='right')
handles=[];labels=[]
for letter,ax in zip('abc',axes):
    lo,hi=ax.get_ylim();ax.set_ylim(lo,hi+.08*(hi-lo))
    lines,names=ax.get_legend_handles_labels()
    handles.extend(lines);labels.extend([f'({letter}) {name}' for name in names])
legend_ax.legend(handles,labels,loc='center',frameon=True,fancybox=True)
save(fig,'S2.1.pdf')

fig,ax=plt.subplots(figsize=(6,4.5))
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
fig.subplots_adjust(left=.18,right=.985,bottom=.145,top=.85)
save(fig,'S2.2.pdf')

with open(ROOT/'2.1_geometry_optimization/monolayer/energy_vacuum/energy_parameters.dat') as f:
    rows=list(csv.DictReader(f,delimiter='\t'))
x=np.array([float(r['a3']) for r in rows]); y=np.array([float(r['total energy']) for r in rows])
fig,ax=plt.subplots(figsize=(6,3.6));ax.plot(x,y,'o-',color=BLUE,ms=4)
ax.set(xlabel=r'Supercell height ($a_3$, $\mathrm{\AA}$)',ylabel='Energy (eV)')
ax.ticklabel_format(axis='y',style='plain',useOffset=False);frame(ax)
fig.subplots_adjust(left=.18,right=.985,bottom=.18,top=.96);save(fig,'S2.3.pdf')

# This supplied tabulation is the source of the thesis fixed-spin figure.
with open(OUT/'fixed_spin_data.csv') as f: rows=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(6,3.6))
for method,color,label in [('PBE+D3',BLUE,'PBE+D3'),('r2SCAN',ORANGE,r'r$^2$SCAN')]:
    selected=[r for r in rows if r['method']==method]
    ax.plot([float(r['fixed_spin_muB']) for r in selected],
            [float(r['relative_energy_eV']) for r in selected],'o-',ms=4,color=color,label=label)
ax.set(xlabel=r'Fixed spin moment ($\mu_{\mathrm{B}}$)',ylabel='Relative energy (eV)')
frame(ax);ax.legend(loc='upper left',ncol=1,frameon=True,fancybox=True)
fig.subplots_adjust(left=.14,right=.985,bottom=.18,top=.96)
save(fig,'S2.15.pdf')
