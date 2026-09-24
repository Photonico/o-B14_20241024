# Project 2 thesis figures

Open `figures_for_thesis.ipynb` and run it from top to bottom with Python, NumPy, SciPy, Matplotlib, h5py, PyYAML, PyMuPDF, Pillow and IPython. It works from the repository root or this directory. Its first cell exposes the font settings; subsequent cells contain the actual plotting commands, `figsize`, margins and legends for each natural figure group. Numerical figures display before saving. Their PDFs are written here and copied to `../PhD_thesis_20251216/figures_proj2` relative to the repository root. The five original structural PNGs are included directly from that thesis directory. Saved notebook outputs are cleared after execution to keep the file small.

The original paper settings are restored: serif text, Computer Modern mathematics, 16 pt axis labels, 14 pt ticks, 12 pt legends, 20 pt titles and 18 pt subtitles. Settings are explicit and do not call `output_settings`. The calculation readers remain in `vmatplot`. No DFT calculation is rerun.

Canvas sizes follow the original source PDFs rather than equalizing printed font sizes. The original three-panel rows retain the requested 2×2 arrangement, with a common legend in the lower-right quadrant:

- `fig2.6`: three original 9×6 in panels, now an 18×12 in grid.
- `fig2.11a`, `fig2.11b`, `fig2.12a`, `fig2.12b`, `S2.18`: each original 24×6 in row becomes a separate 16×12 in grid.
- `fig2.10`, `S2.4`, `S2.5`: original 24×12 in six-panel layouts.
- `fig2.2`: bands, Brillouin zone and PDoS form a 2×2 grid with a shared legend in the fourth cell; the column widths are 2:1, on a 15×13.2 in canvas.
- `S2.1`: dual-x convergence plot and lattice scan are side by side on a 20×7.5 in canvas, labelled (a)/(b).
- `S2.9`: two original 9×6 in panels, combined at 18×6 in. `S2.11`: the paired panels use a 14×5 in canvas to enlarge the text at the same thesis insertion width. `S2.16`: two original 12×6 in DOS plots, stacked at 12×12 in.
- `fig2.8`: 12×6 in; `fig2.9`: two original 8×6 in panels, combined at 16×6 in with a vertical colorbar at the right; `S2.10`: 14×6 in including the original atomic views. `S2.11` has translucent white method labels with light grey rounded borders. Other numerical single plots are 10×6 in, except the original 10×7.5 in dual-x `S2.2`.

The five structural figures (`fig2.1`, `fig2.3`, `fig2.4`, `fig2.7`, `S2.17`) are included from the original labelled high-resolution PNG files. The notebook records and previews them without converting them to PDF. The merged optical `fig2.11.pdf` and `fig2.12.pdf` are superseded by their separate a/b files.

The `.py` files provide a batch equivalent. `python figures_for_thesis/make_figures.py` exports numerical figures; `python figures_for_thesis/structures.py` records the direct PNG references. They do not read later notebook edits, so use the notebook as the editing entry point. Running the numerical batch script overwrites the corresponding PDFs.

## Preserved scientific corrections

- Both reconstructed spin-DOS channels use the same 4000-point grid and the original 0.05 eV Gaussian width; their total is the pointwise sum. Incomplete XML/DOSCAR records have not been treated as complete DOS outputs.
- EPW gap distributions use filename temperatures and decode column 1 as `T + rho/max(rho)`. All 112 temperatures and 11,074 saved bins remain; the line joins weighted means. The invalid power-law fit and fitted critical-temperature markers remain removed. Archived endpoints alone do not establish precise critical temperatures.
- Original negative phonon frequencies remain, including the H-monolayer minimum below −2.8 THz. The AFM DOS maximum remains within the plotted range.
- Optical arrays retain NM-PBE identity, original effective thicknesses and corrected absorption units. `S2.5` labels actual OUTCAR NBANDS values 48/72/144/264.
- `S2.1(b)` retains the actual lattice-scan settings, 600 eV and 10×14×9. `S2.3` identifies total supercell height. `fig2.2` retains the original overlapping atom selections with consistent G2/G3/G7 labels. AIMD retains all 5500 H-bilayer steps and the electronic free-energy column.

`numerical_manifest.json` records sources, scientific details, canvas dimensions, live TeX insertion fractions, text bounds and matching copied-PDF hashes. `structure_manifest.json` records the original image identities. `gap_summary.json` and `spin_dos_verification.json` retain the numerical evidence; `notebook_execution.json` records the completed notebook run. Source calculations and original notebooks are unchanged; no backup or extra raw-data copy is created.
