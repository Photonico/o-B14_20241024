#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

TASK_MARKERS = ("INCAR", "POSCAR", "KPOINTS", "POTCAR")  # 用于判断是否像个VASP任务目录


def is_task_dir(d: Path) -> bool:
    """Heuristic: directory contains typical VASP input files."""
    return any((d / name).is_file() for name in TASK_MARKERS)


def vasprun_complete_fast(vxml: Path, tail_bytes: int = 8192) -> bool:
    """
    Fast check:
    - file exists, non-trivial size
    - tail contains '</modeling>' (final closing tag of vasprun.xml)
    """
    try:
        st = vxml.stat()
        if st.st_size < 1024:  # too small to be a real vasprun.xml
            return False
        with vxml.open("rb") as f:
            if st.st_size > tail_bytes:
                f.seek(-tail_bytes, os.SEEK_END)
            tail = f.read()
        return b"</modeling>" in tail
    except OSError:
        return False


def vasprun_complete_strict(vxml: Path) -> bool:
    """
    Strict check: stream-parse the entire XML.
    If parsing reaches the end without ParseError -> complete.
    """
    try:
        st = vxml.stat()
        if st.st_size < 1024:
            return False
        # iterparse is streaming, avoids loading entire file into memory
        for _event, _elem in ET.iterparse(vxml, events=("end",)):
            pass
        return True
    except (ET.ParseError, OSError):
        return False


def iter_dirs(root: Path):
    """Yield directories under root including nested, skipping some common junk dirs."""
    skip_names = {".git", "__pycache__", ".vscode", ".idea"}
    for dirpath, dirnames, _filenames in os.walk(root):
        # prune skips
        dirnames[:] = [d for d in dirnames if d not in skip_names and not d.startswith(".")]
        yield Path(dirpath)


def main():
    ap = argparse.ArgumentParser(
        description="Check VASP jobs under subdirectories by verifying vasprun.xml completeness."
    )
    ap.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to scan (default: current directory).",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Strict XML parse check (slower but more reliable).",
    )
    ap.add_argument(
        "--only-task-dirs",
        action="store_true",
        help="Only report directories that look like VASP tasks (have INCAR/POSCAR/KPOINTS/POTCAR).",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    checker = vasprun_complete_strict if args.strict else vasprun_complete_fast

    incomplete = []

    for d in iter_dirs(root):
        # Decide whether to treat as a "task dir"
        if args.only_task_dirs and not is_task_dir(d):
            continue

        vxml = d / "vasprun.xml"
        # If it's a task dir but missing vasprun.xml -> incomplete
        if is_task_dir(d) and not vxml.is_file():
            incomplete.append(str(d))
            continue

        # If vasprun.xml exists, check completeness
        if vxml.is_file() and not checker(vxml):
            incomplete.append(str(d))

    if not incomplete:
        print("All done")
        return 0

    for p in incomplete:
        print(p)
    return 1


if __name__ == "__main__":
    sys.exit(main())
