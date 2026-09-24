"""Combine related numerical panels with the original BZ and atomic views."""
import re
import fitz
import numpy as np
import xml.etree.ElementTree as ET
from electronic_data import *

# %% Bands, Brillouin zone, PDoS and a shared legend.
fig, axes = plt.subplots(2, 2, figsize=(15, 13.2),
                         gridspec_kw={'width_ratios': [2, 1]})
ax = axes[0, 0]
draw_bands(ax, 'o-B14_K48')
ax.set(ylim=(-4, 4), ylabel='Energy (eV)', title='(a) Band structure for bulk o-B$_{14}$')
with fitz.open(ROOT/'kpath_tide.pdf') as bz_pdf:
    pixels = bz_pdf[0].get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
axes[0, 1].imshow(np.frombuffer(pixels.samples, dtype=np.uint8).reshape(
    pixels.height, pixels.width, 3))
axes[0, 1].axis('off')
axes[0, 1].set_title('(b) Brillouin zone')
r = ET.parse(ROOT/'4.1_PDoS/o-B14_K20/vasprun.xml').getroot(); section = r.find('.//dos')
fermi = float(section.find("i[@name='efermi']").text)
arrays = np.array([[[float(v) for v in row.text.split()] for row in ion.findall('./set/r')]
                   for ion in section.findall('./partial/array/set/set')])
energy = arrays[0, :, 0]-fermi
ax = axes[1, 0]
for group, atoms, color in [(2, range(4,8), BLUE), (3, [0,1,2,3,12,13], ORANGE), (7, range(14), PURPLE)]:
    values = arrays[list(atoms)].sum(axis=0)
    ax.plot(energy, values[:,2:5].sum(axis=1), color=color, label=rf'$p$ for G{group}')
    ax.plot(energy, values[:,1], color=color, ls='--', label=rf'$s$ for G{group}')
ax.axvline(0, color='#5A3C8C', ls='--', label='Fermi energy')
ax.set(xlim=(-6,6), ylim=(0,6), xlabel='Energy (eV)', ylabel='Density of States',
       title='(c) PDoS for bulk o-B$_{14}$')
frame(ax)
axes[1, 1].axis('off')
handles, labels = ax.get_legend_handles_labels()
handles.insert(0, Line2D([], [], color=BLUE, label='Bulk bands'))
labels.insert(0, 'Bulk bands')
axes[1, 1].legend(handles, labels, loc='center', frameon=True)
fig.subplots_adjust(left=.08, right=.98, bottom=.07, top=.94, wspace=.18, hspace=.24)
save(fig, 'fig2.2.pdf')

# %% Original 10 x 6 AIMD curves with the original atomic views on the right.
rows=[]
for line in (ROOT/'6.0_AIMD/bilayer_H/OSZICAR').read_text().splitlines():
    m=re.search(r'^\s*(\d+)\s+T=\s*([\d.Ee+\-]+).*?F=\s*([\d.Ee+\-]+)',line)
    if m:rows.append([float(x) for x in m.groups()])
step,temp,energy=np.array(rows).T;assert len(step)==5500
fig=plt.figure(figsize=(14,6));gs=fig.add_gridspec(2,2,width_ratios=[2.6,1],left=.09,right=.98,bottom=.13,top=.84,hspace=.13,wspace=.07)
axes=[fig.add_subplot(gs[i,0]) for i in range(2)]
axes[0].plot(step/1000,energy,color=BLUE);axes[1].plot(step/1000,temp,color='#19A0A0')
axes[0].set_ylabel('Electronic free\nenergy (eV)');axes[1].set_ylabel('Temperature (K)')
axes[0].tick_params(labelbottom=False);axes[1].set_xlabel('Time (ps)');axes[1].set_ylim(0,800)
for ax in axes:ax.set_xlim(0,5.5);frame(ax)
for row,filename,title in [(0,'S2.10b1.png','Top\nview'),(1,'S2.10b2.png','Side\nview')]:
    ax=fig.add_subplot(gs[row,1]);ax.imshow(plt.imread(ROOT/'figures_collection'/filename));ax.axis('off');ax.set_title(title.replace('\n',' '),fontsize=18)
fig.suptitle('AIMD simulation for H-terminated bilayer o-B$_{14}$',fontsize=20)
save(fig,'S2.10.pdf')
