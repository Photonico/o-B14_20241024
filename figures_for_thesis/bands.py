"""Grouped band and DOS figures from original VASP data."""
from electronic_data import *

fig, axes = plt.subplots(1,3,figsize=(10,4.6),sharey=True)

for ax, folder, title in zip(axes, ['monolayer_FM_HSE06','monolayer_AFM_HSE06','bilayer_HSE'],
                            ['(a) FM monolayer','(b) AFM monolayer','(c) Bilayer']):
    draw_bands(ax,folder,spin=folder!='bilayer_HSE'); tab(ax,title)

axes[0].set_ylabel(r'$E-E_{\mathrm{F}}$ (eV)')

spin_legend(fig,axes[0],True)

fig.subplots_adjust(left=.085,right=.985,bottom=.25,top=.95,wspace=.09)

save(fig,'fig2.6.pdf')

fig, axes = plt.subplots(1,2,figsize=(10,4.7),sharey=True)

for ax,folder,title in zip(axes,['monolayer_FM','monolayer_FM_HSE06'],['(a) GGA-PBE','(b) HSE06']):
    draw_bands(ax,folder,spin=True); tab(ax,title)

axes[0].set_ylabel(r'$E-E_{\mathrm{F}}$ (eV)')

spin_legend(fig,axes[0])

fig.subplots_adjust(left=.085,right=.985,bottom=.24,top=.95,wspace=.08)

save(fig,'S2.11.pdf')

for filename, folders, labels in [
 ('S2.13.pdf',['monolayer','monolayer_HSE','monolayer_R2SCAN'],['GGA-PBE','HSE06','R2SCAN']),
 ('S2.14.pdf',['monolayer','monolayer_shifting','monolayer_sym_off'],['Original structure','Shifted atoms','Symmetry off'])]:
    fig, ax = plt.subplots(figsize=(10,4.8))
    for folder,label,color in zip(folders,labels,[BLUE,ORANGE,'#8CAF28']):
        draw_bands(ax,folder,color=color,label=label)
    ax.set_ylim(-4,4); ax.set_ylabel(r'$E-E_{\mathrm{F}}$ (eV)')
    tab(ax,'Monolayer'); legend(fig,ax,3)
    fig.subplots_adjust(left=.09,right=.985,bottom=.24,top=.95)
    save(fig,filename)

fig, axes = plt.subplots(1,2,figsize=(10,4.8),sharey=True)

for ax, folder, title in zip(axes,['monolayer_FM_ollie','monolayer_AFM_ollie'],['(a) FM','(b) AFM']):
    energy, channels, _ = dos(folder)
    ax.plot(energy,channels[0]+channels[1],color=BLUE,label='Total')
    ax.plot(energy,channels[0],color=ORANGE,label='Spin up')
    ax.plot(energy,-channels[1],color=CYAN,label='Spin down')
    ax.axvline(0,color=GREY,ls='--',label=r'$E_{\mathrm{F}}=0$')
    ax.set_xlim(-14,6); ax.set_ylim(-8,14); ax.set_xlabel(r'$E-E_{\mathrm{F}}$ (eV)')
    tab(ax,title); frame(ax)

axes[0].set_ylabel('Density of states')

legend(fig,axes[0],4)

fig.subplots_adjust(left=.085,right=.985,bottom=.24,top=.95,wspace=.08)

save(fig,'S2.16.pdf')

fig, ax = plt.subplots(figsize=(10,4.8))

for folder,label,color in [('o-B14_K20','Bulk',BLUE),('monolayer','Monolayer',GREEN),
                            ('bilayer','Bilayer',YELLOW),('bilayer_with_Hydrogen','H-terminated bilayer',ORANGE)]:
    energy, channels, _ = dos(folder)
    ax.plot(energy,channels[0],color=color,label=label)

ax.axvline(0,color=GREY,ls='--')

ax.set(xlim=(-6,6),ylim=(0,26),
           xlabel=r'$E-E_{\mathrm{F}}$ (eV)',ylabel='Density of states')

frame(ax)

legend(fig,ax,2)

fig.subplots_adjust(left=.085,right=.985,bottom=.28,top=.95)

save(fig,'S2.12.pdf')

fig, axes = plt.subplots(1,2,figsize=(10,5),gridspec_kw={'width_ratios':[3,1]},sharey=True)

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

fig.subplots_adjust(left=.085,right=.985,bottom=.17,top=.95,wspace=.10)

save(fig,'fig2.8.pdf')
