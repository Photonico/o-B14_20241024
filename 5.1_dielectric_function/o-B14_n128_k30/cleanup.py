#### cleanup

#!/usr/bin/env python3.6
# %%
import os

# List of files to delete
files_to_delete = [
    "CHG", "CHGCAR", "CONTCAR", "DOSCAR", "EIGENVAL", "IBZKPT", "OSZICAR",
    "OUTCAR", "output.txt", "PCDAT", "REPORT", "vasp.log", "vaspout.h5",
    "vasprun.xml", "XDATCAR"
]

def cleanup(files):
    # Traverse current directory and all subdirectories
    for dirpath, dirnames, filenames in os.walk(os.getcwd()):
        for filename in filenames:
            if filename in files:
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted {file_path}")
                except (FileNotFoundError, PermissionError) as e:
                    print(f"Error deleting {file_path}: {e}")
    print("Cleanup complete.")

# Execute cleanup function with the list of files to delete

cleanup(files_to_delete)

# %%
