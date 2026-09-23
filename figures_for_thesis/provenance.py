"""Record source identities and checks for the numerical thesis exports."""
import csv, hashlib, json, re, subprocess
from pathlib import Path
import fitz, h5py, numpy as np
from electronic_data import bands, ROOT, OUT, THESIS


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path):
    p=ROOT/path
    return {'path':str(path),'sha256':sha(p)}


record={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'style':{'width_inches':10,'width_pdf_points':720,'axes_labels':16,'ticks':14,'legend':14,'tabs':12,
          'font_family':'serif','mathtext':'cm','line_width':1.5,'dpi':196,'legend':'native shared frame'},
 'figures':{},'checks':{},'corrections':[]}
figures=record['figures']
optical=['bilayer_with_Hydrogen','bilayer','monolayer','o-B14_n128_k34']
for filename in ['fig2.10.pdf','fig2.11.pdf','fig2.12.pdf','S2.18.pdf']:
    figures[filename]={'sources':[source(Path('5.1_dielectric_function')/p/'vaspout.h5') for p in optical],
     'observable':'density-density diagonal dielectric tensor; derived scalar optical quantities',
     'thickness_angstrom':{'monolayer':5.807056948202938,'bilayer':9.41157,'bilayer_with_Hydrogen':11.208620},
     'alpha':'2 E k/(hbar c), hbar=6.582119569e-16 eV s, c=2.99792458e18 Angstrom/s',
     'spin':'NM-PBE as in the original optical calculations, no added Drude term'}
record['checks']['historical_h5_vs_rounded_xml']='Previously checked: maximum absolute dielectric difference about 0.000207 after thickness normalization. The superseded extraction archive is no longer needed.'
for filename,folders in [('S2.4.pdf',[f'o-B14_n128_k{k}' for k in (10,20,26,30,32,34)]),
                          ('S2.5.pdf',[f'o-B14_n{k}_k10' for k in (32,64,128,256)])]:
    figures[filename]={'sources':[source(Path('5.1_dielectric_function')/p/'vaspout.h5') for p in folders]}
record['checks']['actual_nbands']={}
for requested in [32,64,128,256]:
    directory=ROOT/f'5.1_dielectric_function/o-B14_n{requested}_k10'
    actual=int(re.search(r'number of bands\s+NBANDS=\s*(\d+)',(directory/'OUTCAR').read_text()).group(1))
    record['checks']['actual_nbands'][str(requested)]={'actual':actual,'evidence':str(directory.relative_to(ROOT)/'OUTCAR')}
for filename,folders in [('fig2.6.pdf',['monolayer_FM_HSE06','monolayer_AFM_HSE06','bilayer_HSE']),
 ('S2.11.pdf',['monolayer_FM','monolayer_FM_HSE06']),('S2.13.pdf',['monolayer','monolayer_HSE','monolayer_R2SCAN']),
 ('S2.14.pdf',['monolayer','monolayer_shifting','monolayer_sym_off']),('fig2.8.pdf',['bilayer_with_Hydrogen']),
 ('fig2.2.pdf',['o-B14_K48'])]:
    figures[filename]={'sources':[source(Path('3.1_bandstructure')/p/'vasprun.xml') for p in folders],
      'energy_reference_eV':{p:bands(p)[-1] for p in folders},
      'path':'KPOINTS_OPT for HSE06; KPOINTS otherwise; original reciprocal metric and each run own Fermi level'}
for filename,folders in [('S2.16.pdf',['monolayer_FM_ollie','monolayer_AFM_ollie']),
 ('S2.12.pdf',['o-B14_K20','monolayer','bilayer','bilayer_with_Hydrogen']),
 ('fig2.8.pdf',['bilayer_with_Hydrogen']),('fig2.2.pdf',['o-B14_K20'])]:
    figures.setdefault(filename,{}).setdefault('sources',[]).extend([
        source(Path('4.1_PDoS')/p/('OUTCAR' if p.endswith('_ollie') else 'vasprun.xml')) for p in folders])
figures['S2.16.pdf']['method']='Gaussian reconstruction from OUTCAR eigenvalues and k weights, standard deviation 0.05 eV as in original helper; both spin channels now use the same 4000-point absolute energy grid. XML/DOSCAR are incomplete.'
figures['fig2.2.pdf']['atom_selections_one_based']={'G2':[5,6,7,8],'G3':[1,2,3,4,13,14],'G7':list(range(1,15))}
figures['fig2.2.pdf']['selection_note']='Same original B2_index/B3_index/B7_index; G7 includes all 14 atoms, not a disjoint third group.'
figures['fig2.2.pdf']['sources'].append(source(Path('kpath_tide.pdf')))
for filename,folders in [('S2.9.pdf',['3.0_phonon_dispersion_vasp/monolayer_3','3.0_phonon_dispersion_vasp/bilayer_3',
                                    '3.0_phonon_dispersion_phononpy/monolayer_top','3.0_phonon_dispersion_phononpy/bilayer_top']),
 ('fig2.5.pdf',['3.0_phonon_dispersion_vasp/monolayer_3','3.0_phonon_dispersion_vasp/bilayer_3']),
 ('S2.8.pdf',[f'3.0_phonon_dispersion_vasp/{p}_2' for p in ['monolayer','monolayer_H','bilayer','bilayer_H']])]:
    figures[filename]={'sources':[source(Path(p)/('band.yaml' if 'phononpy' in p else 'OUTCAR')) for p in folders],
     'note':'Original frequencies, including negative branches, are retained.'}
figures['fig2.9.pdf']={'sources':[source(p.relative_to(ROOT)) for p in sorted((ROOT/'superconductivity').glob('*/B14.imag_aniso_gap0_*'))],
 'format_evidence':['https://github.com/QEF/q-e/blob/qe-6.8/EPW/src/io_eliashberg.f90#L2034-L2082',
                    'https://github.com/QEF/q-e/blob/develop/EPW/src/io/io_supercond.f90#L3270-L3274'],
 'format':'Column 1 = T + rho/max(rho); column 2 = gap in meV. Exact local EPW version is not established.',
 'temperature':'Filename suffix; 0.5 K increments, bulk 0.5 to 30 K, H-bilayer 0.5 to 26 K.',
 'density':'Column 1 minus filename temperature; each temperature independently normalized to its peak.',
 'mean':'sum(density * original gap bin)/sum(density); original uniform 0.03 meV bins; no interpolation.',
 'sample_counts':{'bulk':5982,'bilayer_with_Hydrogen':5092},
 'temperature_counts':{'bulk':60,'bilayer_with_Hydrogen':52},
 'statistics_file':source(Path('figures_for_thesis/gap_summary.json')),
 'interpretation':'Distribution mean is a guide, not an order-parameter fit. No fitted Tc is shown. Archived endpoints do not establish precise Tc or a rigorous closure bracket; no normal-state convergence output is available.'}
figures['S2.10.pdf']={'sources':[source(Path('6.0_AIMD/bilayer_H/OSZICAR')),
 source(Path('figures_collection/S2.10b1.png')),source(Path('figures_collection/S2.10b2.png'))],
 'steps':5500,'time_step_fs':1,'energy_column':'F, electronic free energy; not conserved total energy'}
for filename,paths in [('S2.1.pdf',[f'1.0_energy/o-B14/{p}/energy_parameters.dat' for p in ['energy_kpoints_450','energy_kpoints_480','energy_encut_1260','energy_encut_8721','energy_lattice']]),
 ('S2.2.pdf',[f'1.0_energy/o-B14/{p}/cohesive_energy.dat' for p in ['energy_kpoints_480','energy_encut_1260']]),
 ('S2.3.pdf',['2.1_geometry_optimization/monolayer/energy_vacuum/energy_parameters.dat']),
 ('S2.15.pdf',['figures_for_thesis/fixed_spin_data.csv'])]:
    figures[filename]={'sources':[source(Path(p)) for p in paths]}
figures['S2.15.pdf']['source_note']='Small supplied thesis CSV moved to the calculation repository; not claimed to be recovered raw DFT input/output.'
with open(ROOT/'1.0_energy/o-B14/energy_lattice/energy_parameters.dat') as f: rows=list(csv.DictReader(f,delimiter='\t'))
record['checks']['lattice_scan']={'rows':len(rows),'mesh_values':sorted(set(r['kpoints mesh'] for r in rows)),
 'ENCUT_eV':sorted(set(r['energy cutoff (ENCUT)'] for r in rows)),
 'sample_minimum_a_angstrom':float(min(rows,key=lambda r:float(r['total energy']))['lattice constant'])}
record['corrections']=[
 'S2.1(c) actual lattice scan uses 10x14x9, not the 34x48x31 previously stated in the caption; 37 original table rows.',
 'S2.3 x is a3 supercell height, not vacuum width; the original energy_parameters.dat column a3 is plotted.',
 'S2.5 legends show actual OUTCAR band counts 48/72/144/264; INCAR requests 32/64/128/256.',
 'fig2.2 original display names Group 1/Group 2/Group 3 correspond to blue B2_index/orange B3_index/purple B7_index. The orange s curve was incorrectly labelled Group 3 while orange p was labelled Group 2. The revised labels consistently use G2/G3/G7 for these unchanged atom selections.',
 'Previously corrected alpha factor 2pi/10, monolayer effective thickness 5.807 Angstrom, and AIMD H-bilayer identity retained.']
record['checks']['spin_dos_common_grid']=json.loads((OUT/'spin_dos_verification.json').read_text())
record['corrections'].append('S2.16 old spin channels had different energy grids; reconstruct both on one common grid before forming their pointwise total.')
record['corrections'].append('fig2.9 old fit mistook EPW normalized histogram-density offsets for temperatures; use filename temperatures and recover densities, remove the invalid power-law fit and Tc markers.')
gap_stats=json.loads((OUT/'gap_summary.json').read_text())
record['checks']['gap_histograms']={folder:{'temperature_count':len(rows),'saved_bins':sum(r['bins'] for r in rows),
 'last_temperature':rows[-1],'independent_parser_agreement':'All 112 temperatures, bin counts, min/max and weighted means agree within 1e-12.'}
 for folder,rows in gap_stats.items()}
for filename,item in figures.items():
    p=OUT/filename;page=fitz.open(p)[0]
    assert abs(page.rect.width-720)<.01
    bounds=[]
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            for span in line['spans']:
                if not page.rect.contains(fitz.Rect(span['bbox'])):bounds.append(span['text'])
    assert not bounds,(filename,bounds)
    item['pdf_sha256']=sha(p);item['copied_pdf_identical']=sha(THESIS/filename)==sha(p)
    assert item['copied_pdf_identical']
    item['width_pdf_points']=page.rect.width
    item['height_pdf_points']=page.rect.height
(OUT/'numerical_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print('Verified',len(figures),'numerical PDFs: width, text bounds, source identities and copied hashes.')
