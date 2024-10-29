#### Common codes
# pylint: disable = C0103, C0114, C0116, C0301, C0321, R0913, R0914

import xml.etree.ElementTree as ET
import os

import numpy as np

from typing import Tuple, Union

def check_vasprun(directory="."):
    """Find folders with complete vasprun.xml and print incomplete ones."""
    # Check if the user asked for help
    if directory == "help":
        print("Please use this function on the parent directory of the project's main folder.")
        return []
    complete_folders = []
    # Traverse all folders under "directory"
    for dirpath, _, filenames in os.walk(directory):
        if "vasprun.xml" in filenames:
            xml_path = os.path.join(dirpath, "vasprun.xml")

            # Check if vasprun.xml is complete
            try:
                with open(xml_path, "r", encoding="utf-8") as xml_file:
                    # Check the last few lines for the closing tag
                    last_lines = xml_file.readlines()[-10:] # read the last 10 lines
                    for line in last_lines:
                        if "</modeling>" in line or "</vasp>" in line:
                            complete_folders.append(dirpath)
                            break
                    else:
                        print(f"vasprun.xml in {dirpath} is incomplete.")
            except IOError as e:                        # Change from Exception to IOError
                print(f"Error reading {xml_path}: {e}")
    return complete_folders
