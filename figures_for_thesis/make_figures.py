"""Run from any directory; PDFs are also copied to the sibling thesis repository."""
from pathlib import Path
import os, subprocess, sys
folder=Path(__file__).resolve().parent
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
for script in ['bands.py','optics.py','phonons_gaps.py','gaps.py','convergence.py','composites.py','verify_spin_dos.py','provenance.py']:
    subprocess.run([sys.executable,str(folder/script)],check=True,env=env)
