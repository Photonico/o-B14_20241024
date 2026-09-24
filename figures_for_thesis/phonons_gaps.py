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
    frame(ax)


pristine=[('3.0_phonon_dispersion_vasp/monolayer_3','Monolayer',CYAN),
          ('3.0_phonon_dispersion_vasp/bilayer_3','Bilayer',YELLOW)]
hydrogen=[('3.0_phonon_dispersion_phononpy/monolayer_top','H-terminated monolayer','#8CAF28'),
          ('3.0_phonon_dispersion_phononpy/bilayer_top','H-terminated bilayer',ORANGE)]
# %% Original two 9 x 6 phonon plots, side by side.
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for ax, group in zip(axes, [pristine, hydrogen]):
    for folder, label, color in group: draw_phonon(ax, folder, label+' 3×3', color)
    ax.axhline(0, color='#5A3C8C', ls='--', zorder=0)
    ax.set(ylim=(-3.3, 8), ylabel='Frequency (THz)', title='Phonon dispersion')
    ax.legend(loc='upper right')
fig.subplots_adjust(left=.06, right=.97, bottom=.10, top=.88, wspace=.18)
save(fig, 'S2.9.pdf')

# %% Original single phonon figures.
for filename, group, ylim in [('fig2.5.pdf', pristine, (-1, 7)),
 ('S2.8.pdf', [(f'3.0_phonon_dispersion_vasp/{folder}', label+' 2×2', color)
  for folder, label, color in [('monolayer_2','Monolayer',CYAN),
  ('monolayer_H_2','H-terminated monolayer','#8CAF28'),
  ('bilayer_2','Bilayer',YELLOW),('bilayer_H_2','H-terminated bilayer',ORANGE)]], (-3, 4))]:
    fig, ax = plt.subplots(figsize=(10, 6))
    for folder, label, color in group: draw_phonon(ax, folder, label, color)
    ax.axhline(0, color='#5A3C8C', ls='--', zorder=0)
    ax.set(ylim=ylim, ylabel='Frequency (THz)', title='Phonon dispersion')
    ax.legend(loc='upper right')
    fig.subplots_adjust(left=.10, right=.97, bottom=.10, top=.89)
    save(fig, filename)
