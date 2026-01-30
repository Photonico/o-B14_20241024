import os
import shutil

#PATH is your calculation directory which we navigate to.
PATH = os.getcwd()
os.chdir(PATH)
FILES = os.listdir(PATH)

#Return a list of all the phonopy generated POSCAR-xxx files
DISP_FILES = [F for F in FILES if("POSCAR-" in F)]

#Create the disp-xxx directories and copy the files to calculate the POSCAR-xxx jobs
for POSCAR in DISP_FILES:
    FILE = POSCAR.strip().split("-")
    INDEX = FILE[1]
    NEW_DIR = "strain-"+INDEX
    os.mkdir(NEW_DIR)
    shutil.copy("INCAR", NEW_DIR)
    shutil.copy("KPOINTS", NEW_DIR)
    shutil.copy("POTCAR", NEW_DIR)
    shutil.copy("POSCAR-"+INDEX, NEW_DIR)
    shutil.copy("run_cpu.csh", NEW_DIR)
    os.rename(NEW_DIR+"/POSCAR-"+INDEX,NEW_DIR+"/POSCAR")

print("Finished!")