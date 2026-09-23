"""Combine related numerical panels with the original BZ and atomic views."""
import io, re
import fitz
import numpy as np
import xml.etree.ElementTree as ET
from electronic_data import *

# Bulk bands / Brillouin zone / grouped PDOS in one fixed-width figure.
fig=plt.figure(figsize=(10,8.2))
gs=fig.add_gridspec(2,2,width_ratios=[1.6,1],left=.095,right=.98,bottom=.23,top=.95,hspace=.28,wspace=.1)
ax=fig.add_subplot(gs[0,0]); bz=fig.add_subplot(gs[0,1]); pd=fig.add_subplot(gs[1,:]); bz.axis('off')
draw_bands(ax,'o-B14_K48');ax.set_ylim(-4,4);ax.set_ylabel(r'$E-E_{\mathrm{F}}$ (eV)')
tab(ax,'(a) Bands');tab(bz,'(b) Brillouin zone')
r=ET.parse(ROOT/'4.1_PDoS/o-B14_K20/vasprun.xml').getroot();section=r.find('.//dos')
fermi=float(section.find("i[@name='efermi']").text)
arrays=np.array([[[float(v) for v in row.text.split()] for row in ion.findall('./set/r')]
                 for ion in section.findall('./partial/array/set/set')])
energy=arrays[0,:,0]-fermi
for group,atoms,color in [(2,range(4,8),BLUE),(3,[0,1,2,3,12,13],ORANGE),(7,range(14),PURPLE)]:
    values=arrays[list(atoms)].sum(axis=0)
    pd.plot(energy,values[:,2:5].sum(axis=1),color=color,label=rf'G{group}: $p$')
    pd.plot(energy,values[:,1],color=color,ls='--',label=rf'G{group}: $s$')
pd.axvline(0,color=GREY,ls='--');pd.set(xlim=(-6,6),ylim=(0,6),xlabel=r'$E-E_{\mathrm{F}}$ (eV)',ylabel='Projected DOS')
tab(pd,'(c) Orbital projections');frame(pd);legend(fig,pd,3)
# The projected BZ segments are copied exactly; labels use the same native font size.
source=fitz.open(ROOT/'kpath_tide.pdf')[0]
for drawing in source.get_drawings():
    if drawing['color'] is None: continue
    for item in drawing['items']:
        if item[0]=='l':
            p1,p2=item[1:]
            bz.plot([p1.x,p2.x],[p1.y,p2.y],color=drawing['color'],
                    ls='--' if drawing['dashes']!='[] 0' else '-',lw=1.5)
for block in source.get_text('dict')['blocks']:
    for line in block.get('lines',[]):
        for span in line['spans']:
            color=tuple(v/255 for v in fitz.sRGB_to_rgb(span['color']))
            bz.text(*span['origin'],span['text'],fontsize=14,color=color,va='baseline')
bz.text(122.5,143,r'$\Gamma$',fontsize=14,color='#C82864')
bz.set(xlim=(0,288),ylim=(288,0),aspect='equal')
save(fig,'fig2.2.pdf')

# AIMD curves keep their full source font sizes; atomic views occupy a separate column.
rows=[]
for line in (ROOT/'6.0_AIMD/bilayer_H/OSZICAR').read_text().splitlines():
    m=re.search(r'^\s*(\d+)\s+T=\s*([\d.Ee+\-]+).*?F=\s*([\d.Ee+\-]+)',line)
    if m:rows.append([float(x) for x in m.groups()])
step,temp,energy=np.array(rows).T;assert len(step)==5500
fig=plt.figure(figsize=(10,5.4));gs=fig.add_gridspec(2,2,width_ratios=[2.1,1],left=.14,right=.99,bottom=.14,top=.95,hspace=.13,wspace=.08)
axes=[fig.add_subplot(gs[i,0]) for i in range(2)]
axes[0].plot(step/1000,energy,color=BLUE);axes[1].plot(step/1000,temp,color='#19A0A0')
axes[0].set_ylabel('Electronic free\nenergy (eV)');axes[1].set_ylabel('Temperature (K)')
axes[0].tick_params(labelbottom=False);axes[1].set_xlabel('Time (ps)');axes[1].set_ylim(0,800)
for ax in axes:ax.set_xlim(0,5.5);frame(ax)
for row,filename,title in [(0,'S2.10b1.png','Top view'),(1,'S2.10b2.png','Side view')]:
    ax=fig.add_subplot(gs[row,1]);ax.imshow(plt.imread(ROOT/'figures_collection'/filename));ax.axis('off');tab(ax,title)
save(fig,'S2.10.pdf')
