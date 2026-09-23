"""Grouped band and DOS figures from original VASP data."""
from electronic_data import *

fig, grid = plt.subplots(2,2,figsize=(8,5.6))
axes=grid.flat[:3]

for ax, folder, title in zip(axes, ['monolayer_FM_HSE06','monolayer_AFM_HSE06','bilayer_HSE'],
                            ['(a) FM monolayer','(b) AFM monolayer','(c) Bilayer']):
    draw_bands(ax,folder,spin=folder!='bilayer_HSE'); tab(ax,title)

axes[0].set_ylabel(r'$E-E_{\mathrm{F}}$ (eV)')
axes[2].set_ylabel(r'$E-E_{\mathrm{F}}$ (eV)')
axes[1].tick_params(labelleft=False)
grid[1,1].axis('off')
handles=[Line2D([],[],color=PURPLE,label='Spin up'),
         Line2D([],[],color=CYAN,ls=(0,(4,3)),label='Spin down'),
         Line2D([],[],color=BLUE,label='Bilayer bands'),
         Line2D([],[],color=GREY,ls='--',label=r'$E_{\mathrm{F}}=0$')]
grid[1,1].legend(handles=handles,loc='center',frameon=True,fancybox=True)
fig.subplots_adjust(left=.095,right=.988,bottom=.115,top=.96,wspace=.14,hspace=.29)

save(fig,'fig2.6.pdf')

fig, grid = plt.subplots(1,3,figsize=(8,3.4),
                         gridspec_kw={'width_ratios':[1,1,.42]})
axes=grid[:2]

for ax,folder,title in zip(axes,['monolayer_FM','monolayer_FM_HSE06'],['(a) GGA-PBE','(b) HSE06']):
    draw_bands(ax,folder,spin=True); tab(ax,title)

axes[0].set_ylabel(r'$E-E_{\mathrm{F}}$ (eV)')

axes[1].tick_params(labelleft=False)
grid[2].axis('off')
spin_legend(fig,grid[2])

fig.subplots_adjust(left=.09,right=.99,bottom=.18,top=.95,wspace=.09)

save(fig,'S2.11.pdf')

for filename, folders, labels in [
 ('S2.13.pdf',['monolayer','monolayer_HSE','monolayer_R2SCAN'],['GGA-PBE','HSE06','R2SCAN']),
 ('S2.14.pdf',['monolayer','monolayer_shifting','monolayer_sym_off'],['Original\nstructure','Shifted\natoms','Symmetry off'])]:
    fig, (ax,key) = plt.subplots(1,2,figsize=(7,3.6),
                                 gridspec_kw={'width_ratios':[1,.3]})
    for folder,label,color in zip(folders,labels,[BLUE,ORANGE,'#8CAF28']):
        draw_bands(ax,folder,color=color,label=label)
    ax.set_ylim(-4,4); ax.set_ylabel(r'$E-E_{\mathrm{F}}$ (eV)')
    tab(ax,'Monolayer')
    key.axis('off')
    handles,names=ax.get_legend_handles_labels()
    key.legend(handles,names,loc='center',frameon=True,fancybox=True,
                borderpad=.35,labelspacing=.45,handlelength=1.4)
    fig.subplots_adjust(left=.11,right=.985,bottom=.18,top=.96,wspace=.10)
    save(fig,filename)

fig, axes = plt.subplots(2,1,figsize=(6,6),sharex=True,sharey=True)

for ax, folder, title in zip(axes,['monolayer_FM_ollie','monolayer_AFM_ollie'],['(a) FM','(b) AFM']):
    energy, channels, _ = dos(folder)
    ax.plot(energy,channels[0]+channels[1],color=BLUE,label='Total')
    ax.plot(energy,channels[0],color=ORANGE,label='Spin up')
    ax.plot(energy,-channels[1],color=CYAN,label='Spin down')
    ax.axvline(0,color=GREY,ls='--',label=r'$E_{\mathrm{F}}=0$')
    ax.set_xlim(-14,6); ax.set_ylim(-8,15); ax.set_xlabel(r'$E-E_{\mathrm{F}}$ (eV)')
    tab(ax,title); frame(ax)

for ax in axes: ax.set_ylabel('Density of states')
axes[0].set_xlabel('')

axes[0].legend(loc='upper right',ncol=2,frameon=True,fancybox=True)

fig.subplots_adjust(left=.14,right=.985,bottom=.105,top=.97,hspace=.10)

save(fig,'S2.16.pdf')

fig, ax = plt.subplots(figsize=(6,3.6))

for folder,label,color in [('o-B14_K20','Bulk',BLUE),('monolayer','Monolayer',GREEN),
                            ('bilayer','Bilayer',YELLOW),('bilayer_with_Hydrogen','H-terminated bilayer',ORANGE)]:
    energy, channels, _ = dos(folder)
    ax.plot(energy,channels[0],color=color,label=label)

ax.axvline(0,color=GREY,ls='--')

ax.set(xlim=(-6,6),ylim=(0,26),
           xlabel=r'$E-E_{\mathrm{F}}$ (eV)',ylabel='Density of states')

frame(ax)

ax.legend(loc='upper right',ncol=2,frameon=True,fancybox=True)

fig.subplots_adjust(left=.135,right=.985,bottom=.18,top=.97)

save(fig,'S2.12.pdf')

fig, axes = plt.subplots(1,2,figsize=(8,4),gridspec_kw={'width_ratios':[3,1]},sharey=True)

draw_bands(axes[0],'bilayer_with_Hydrogen')

axes[0].set_ylim(-4,4)

energy, channels, _ = dos('bilayer_with_Hydrogen')

axes[1].plot(channels[0],energy,color=BLUE)

axes[1].axhline(0,color=GREY,ls='--')

axes[1].set_xlim(0,20)

axes[0].set_ylabel(r'$E-E_{\mathrm{F}}$ (eV)')

axes[1].set_xlabel('DOS (a.u.)')

tab(axes[0],'(a) Bands')

tab(axes[1],'(b) DOS')

frame(axes[1])

fig.subplots_adjust(left=.10,right=.985,bottom=.16,top=.96,wspace=.10)

save(fig,'fig2.8.pdf')
