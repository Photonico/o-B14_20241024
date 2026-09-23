"""Original electronic data readers and small shared panel helpers."""
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache
import numpy as np
from matplotlib.lines import Line2D
from style import *
sys.path.insert(0, str(ROOT))
from vmatplot.bandstructure import extract_kpath, kpoints_path_lists
from vmatplot.dos import (_read_outcar_parameters, _read_outcar_eigen_matrices,
                          _read_outcar_kpoint_weights)

@lru_cache(None)
def bands(folder):
    directory = ROOT / '3.1_bandstructure' / folder
    root = ET.parse(directory / 'vasprun.xml').getroot()
    if (directory / 'KPOINTS_OPT').exists():
        eigen = root.find('./calculation/eigenvalues_kpoints_opt/eigenvalues')
        fermi = float(root.find("./calculation/dos[@comment='kpoints_opt']/i[@name='efermi']").text)
    else:
        eigen = root.find('./calculation/eigenvalues')
        fermi = float(root.find(".//i[@name='efermi']").text)
    energies = []
    for spin in eigen.findall('./array/set/set'):
        energies.append(np.array([[float(r.text.split()[0]) for r in k.findall('r')]
                                   for k in spin.findall('set')]) - fermi)
    x, breaks = extract_kpath(str(directory), return_breaks=True)
    ticks, labels = kpoints_path_lists(str(directory))
    return np.array(x), energies, ticks, labels, breaks, fermi

def draw_bands(ax, folder, spin=False, color=BLUE, label=None, linestyle='-'):
    x, energies, ticks, labels, breaks, _ = bands(folder)
    for channel, values in enumerate(energies if spin else energies[:1]):
        xx, yy = x.copy(), values.copy()
        for i in reversed(breaks):
            xx = np.insert(xx, i, xx[i]); yy = np.insert(yy, i, np.nan, axis=0)
        line = ax.plot(xx, yy, color=(PURPLE, CYAN)[channel] if spin else color,
                       ls=('-', (0,(4,3)))[channel] if spin else linestyle)
        if label: line[0].set_label(label)
    for tick in ticks[1:-1]: ax.axvline(tick, color=GREY, ls='--', alpha=.4, zorder=0)
    ax.axhline(0, color=GREY, ls='--', zorder=1)
    ax.set_xticks(ticks, labels); ax.set_xlim(x[0], x[-1]); ax.set_ylim(-4, 3)
    ax.set_xlabel(r'Wave vector ($k$)'); frame(ax)

def spin_legend(fig, ax, bilayer=False):
    handles = [Line2D([],[],color=PURPLE,label='Spin up'),
               Line2D([],[],color=CYAN,ls=(0,(4,3)),label='Spin down')]
    if bilayer: handles.append(Line2D([],[],color=BLUE,label='Bilayer bands'))
    handles.append(Line2D([],[],color=GREY,ls='--',label=r'$E_{\mathrm{F}}=0$'))
    ax.legend(handles=handles,loc='center',frameon=True,fancybox=True,
              borderpad=.3,labelspacing=.35,handlelength=1.5)

@lru_cache(None)
def dos(folder):
    directory = ROOT / '4.1_PDoS' / folder
    if folder.endswith('_ollie'):
        # Both channels must be reconstructed on the same absolute energy grid.
        lines=(directory/'OUTCAR').read_text().splitlines()
        params=_read_outcar_parameters(lines)
        eigen,_=_read_outcar_eigen_matrices(lines)
        sigma=params['sigma']; assert sigma == .05
        assert eigen[1].shape == eigen[2].shape == (32,25)
        weights=_read_outcar_kpoint_weights(lines,25)
        lo=min(a.min() for a in eigen.values())-6*sigma
        hi=max(a.max() for a in eigen.values())+6*sigma
        energy=np.linspace(lo,hi,params['nedos'])
        channels=[]
        for spin in (1,2):
            values=np.zeros_like(energy)
            for k,weight in enumerate(weights):
                z=(energy[:,None]-eigen[spin][:,k])/sigma
                values += weight*np.exp(-.5*z*z).sum(axis=1)/(sigma*np.sqrt(2*np.pi))
            channels.append(values)
        return energy-params['efermi'],channels,params['efermi']
    root = ET.parse(directory / 'vasprun.xml').getroot()
    section = root.find('.//dos')
    fermi = float(section.find("i[@name='efermi']").text)
    arrays = [np.array([[float(v) for v in r.text.split()] for r in spin.findall('r')])
              for spin in section.findall('./total/array/set/set')]
    return arrays[0][:,0] - fermi, [a[:,1] for a in arrays], fermi
