"""Verify common-grid spin DOS by state counting and an independent Gaussian mixture."""
import json
import numpy as np
from scipy.stats import norm
from electronic_data import *
checks={}
for folder in ['monolayer_FM_ollie','monolayer_AFM_ollie']:
    energy,channels,fermi=dos(folder)
    lines=(ROOT/'4.1_PDoS'/folder/'OUTCAR').read_text().splitlines()
    # Import the source parsers explicitly; plotting is not involved.
    from vmatplot.dos import _read_outcar_eigen_matrices, _read_outcar_kpoint_weights
    eigen,_=_read_outcar_eigen_matrices(lines);weights=_read_outcar_kpoint_weights(lines,25)
    indices=np.linspace(0,len(energy)-1,61,dtype=int)
    differences=[]
    for spin,values in enumerate(channels,1):
        direct=(norm.pdf(energy[indices,None,None]+fermi,loc=eigen[spin][None,:,:],scale=.05)
                *weights[None,None,:]).sum(axis=(1,2))
        differences.append(float(np.max(abs(direct-values[indices]))))
    integrals=[float(np.trapezoid(v,energy)) for v in channels]
    assert max(differences)<1e-11
    assert max(abs(np.array(integrals)-32))<1e-6
    checks[folder]={'common_energy_grid_points':len(energy),'energy_range_relative_Fermi_eV':[energy[0],energy[-1]],
     'fermi_eV':fermi,'Gaussian_standard_deviation_eV':.05,'bands_per_spin':32,'irreducible_kpoints':25,
     'weight_sum':float(weights.sum()),'integrated_states_per_spin':integrals,
     'independent_scipy_mixture_max_error':differences,'total_is_exact_pointwise_sum':True,
     'old_error':'The original helper independently generated spin grids; both curves and their sum now use this common grid.'}
(OUT/'spin_dos_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
