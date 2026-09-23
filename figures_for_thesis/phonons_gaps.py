"""Phonon dispersions from the archived calculation outputs."""
import sys, yaml
from functools import lru_cache
import numpy as np
from style import *
sys.path.insert(0,str(ROOT))
from vmatplot.phonon import extract_phonon_bands, extract_qpath


@lru_cache(None)
def phonon(folder):
    directory=ROOT/folder
    if (directory/'band.yaml').exists():
        with open(directory/'band.yaml') as f: raw=yaml.load(f,Loader=yaml.CSafeLoader)
        x=np.array([q['distance'] for q in raw['phonon']])
        y=np.array([[b['frequency'] for b in q['band']] for q in raw['phonon']])
        assert np.all(np.diff(x)>=-1e-8)
        ends=np.cumsum(raw['segment_nqpoint'])
        ticks=[x[0]]
        for end in ends:
            if x[end-1]>ticks[-1]+1e-10: ticks.append(x[end-1])
        labels=['Γ','Z','T','Y','Γ']; assert len(ticks)==len(labels)
        for end in reversed(ends[:-1]):
            x=np.insert(x,end,np.nan); y=np.insert(y,end,np.nan,axis=0)
        return x,y,ticks,labels
    else:
        raw=extract_phonon_bands(str(directory)); x=np.array(extract_qpath(str(directory)))
        n=len(x)//4
        ticks=x[[0,n-1,2*n-1,3*n-1,len(x)-1]]; labels=['Γ','Z','T','Y','Γ']
    return x, np.array(raw['bands']).T, ticks, labels


def draw_phonon(ax,folder,label,color):
    x,y,ticks,labels=phonon(folder)
    lines=ax.plot(x,y,color=color); lines[0].set_label(label)
    ax.set_xticks(ticks,labels); ax.set_xlim(np.nanmin(x),np.nanmax(x))
    ax.set_xlabel(r'Wave vector ($q$)'); frame(ax)


pristine=[('3.0_phonon_dispersion_vasp/monolayer_3','Monolayer',CYAN),
          ('3.0_phonon_dispersion_vasp/bilayer_3','Bilayer',YELLOW)]
hydrogen=[('3.0_phonon_dispersion_phononpy/monolayer_top','H-terminated monolayer','#8CAF28'),
          ('3.0_phonon_dispersion_phononpy/bilayer_top','H-terminated bilayer',ORANGE)]
fig,grid=plt.subplots(1,3,figsize=(8,3.4),
                       gridspec_kw={'width_ratios':[1,1,.55]})
axes=grid[:2]
for ax,group,title in zip(axes,[pristine,hydrogen],['(a) Pristine','(b) Hydrogen-terminated']):
    for folder,label,color in group: draw_phonon(ax,folder,label.replace('H-terminated ','H-terminated\n'),color)
    ax.axhline(0,color=GREY,ls='--',zorder=0); ax.set_ylim(-3.3,8); tab(ax,title)
axes[0].set_ylabel('Frequency (THz)')
axes[1].tick_params(labelleft=False)
grid[2].axis('off')
handles=[];labels=[]
for ax in axes:
    h,l=ax.get_legend_handles_labels();handles.extend(h);labels.extend(l)
grid[2].legend(handles,labels,loc='center right',frameon=True,fancybox=True,
               borderaxespad=0,borderpad=.35,labelspacing=.45,handlelength=1.3)
fig.subplots_adjust(left=.09,right=.99,bottom=.18,top=.95,wspace=.10)
save(fig,'S2.9.pdf')

for filename,group,ylim in [('fig2.5.pdf',pristine,(-1,7)),
 ('S2.8.pdf',[(f'3.0_phonon_dispersion_vasp/{folder}',label,color)
  for folder,label,color in [('monolayer_2','Monolayer',CYAN),('monolayer_H_2','H-terminated monolayer','#8CAF28'),
  ('bilayer_2','Bilayer',YELLOW),('bilayer_H_2','H-terminated bilayer',ORANGE)]],(-3,4))]:
    if filename=='fig2.5.pdf':
        fig,(ax,key)=plt.subplots(1,2,figsize=(7,3.6),
                                  gridspec_kw={'width_ratios':[1,.3]})
        key.axis('off')
    else:
        fig,ax=plt.subplots(figsize=(6,3.6));key=None
    for folder,label,color in group: draw_phonon(ax,folder,label,color)
    ax.axhline(0,color=GREY,ls='--',zorder=0); ax.set_ylim(ylim); ax.set_ylabel('Frequency (THz)')
    if key is not None:
        handles,labels=ax.get_legend_handles_labels()
        key.legend(handles,labels,loc='center',frameon=True,fancybox=True,
                    borderpad=.35,labelspacing=.45,handlelength=1.4)
    else:
        # This broad negative-frequency space is clear; the left-hand negative branch remains visible.
        ax.legend(loc='lower right',bbox_to_anchor=(.99,.015),frameon=True,fancybox=True,
                  borderpad=.3,labelspacing=.25,handlelength=1.4)
    fig.subplots_adjust(left=.11 if key is not None else .13,right=.985,
                        bottom=.18,top=.96,wspace=.10)
    save(fig,filename)
