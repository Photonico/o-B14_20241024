"""Plot the archived EPW gap histograms at their actual temperatures."""
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from style import *

fig,axes=plt.subplots(1,2,figsize=(16,6))
cmap=LinearSegmentedColormap.from_list('gap_density',['white',BLUE])
records={}
for ax,folder,title in zip(axes,['bulk','bilayer_with_Hydrogen'],
                          ['(a) Bulk o-B$_{14}$','(b) H-terminated bilayer o-B$_{14}$']):
    rows=[]
    for file in sorted((ROOT/'superconductivity'/folder).glob('B14.imag_aniso_gap0_*')):
        temperature=float(file.name.rsplit('_',1)[1])
        data=np.loadtxt(file,ndmin=2)
        density=data[:,0]-temperature
        delta=data[:,1]
        assert np.isfinite(data).all() and np.all(density>=0)
        assert np.isclose(density.max(),1) and np.all(density<=1+1e-8)
        # Column 1 is T + rho/max(rho), not an independent temperature sample.
        points=ax.scatter(np.full(len(delta),temperature),delta,c=density,
                          cmap=cmap,vmin=0,vmax=1,marker='_',s=14,linewidths=.9)
        rows.append({'temperature_K':temperature,'bins':len(delta),
                     'gap_min_meV':float(delta.min()),'gap_max_meV':float(delta.max()),
                     'density_mean_meV':float(np.average(delta,weights=density)),
                     'density_min':float(density.min()),'density_max':float(density.max())})
    ax.plot([r['temperature_K'] for r in rows],[r['density_mean_meV'] for r in rows],
            color=ORANGE,label='Distribution mean')
    ax.set(xlim=(0,31),ylim=(0,6.6),xlabel='Temperature (K)')
    ax.set_title(title); ax.set_ylabel(r'Superconducting gap $\Delta$ (meV)'); frame(ax)
    ax.legend(loc='upper right')
    records[folder]=rows

fig.subplots_adjust(left=.065,right=.90,bottom=.14,top=.87,wspace=.20)
colorbar=fig.colorbar(points,cax=fig.add_axes([.925,.14,.020,.73]),
                     orientation='vertical',ticks=[0,.5,1])
colorbar.set_label('Normalized gap density',fontsize=12)
colorbar.ax.tick_params(direction='in',labelsize=12)
save(fig,'fig2.9.pdf')
(OUT/'gap_summary.json').write_text(json.dumps(records,indent=2)+'\n')
