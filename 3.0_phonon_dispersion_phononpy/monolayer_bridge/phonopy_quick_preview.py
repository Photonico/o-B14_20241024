#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def vasprun_complete_fast(vxml: Path, tail_bytes: int = 8192) -> bool:
    """Fast check: file tail contains </modeling>."""
    try:
        if not vxml.is_file():
            return False
        st = vxml.stat()
        if st.st_size < 1024:
            return False
        with vxml.open("rb") as f:
            if st.st_size > tail_bytes:
                f.seek(-tail_bytes, os.SEEK_END)
            tail = f.read()
        return b"</modeling>" in tail
    except OSError:
        return False


def sort_key_dir(d: Path):
    m = re.search(r"(\d+)", d.name)
    if m:
        return (0, int(m.group(1)), d.name)
    return (1, 0, d.name)


def find_vasprun_in_dir(d: Path) -> Path | None:
    """Prefer d/vasprun.xml; else search recursively and take the shortest-path one."""
    v = d / "vasprun.xml"
    if v.is_file():
        return v
    hits = list(d.rglob("vasprun.xml"))
    if not hits:
        return None
    hits.sort(key=lambda p: (len(p.parts), str(p)))
    return hits[0]


def parse_qpoints_to_bandconf(qpoints_path: Path, dim_line: str) -> str:
    """
    Convert VASP QPOINTS (line-mode) to phonopy band.conf content.
    """
    raw = qpoints_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = [ln.rstrip("\n") for ln in raw]
    lines = [ln for ln in lines]  # keep blanks for structure

    # Find N (points per segment)
    if len(lines) < 5:
        raise ValueError("QPOINTS too short.")
    nseg_points = int(lines[1].split()[0])

    # Parse endpoint lines: skip first 4 header lines; then read non-empty lines in pairs
    point_lines = []
    for ln in lines[4:]:
        ln = ln.strip()
        if not ln:
            continue
        point_lines.append(ln)

    if len(point_lines) % 2 != 0:
        raise ValueError("QPOINTS endpoint lines count is not even (expected pairs).")

    endpoints = []
    seg_labels = []

    for i in range(0, len(point_lines), 2):
        a = point_lines[i].split()
        b = point_lines[i + 1].split()
        if len(a) < 3 or len(b) < 3:
            raise ValueError(f"Bad QPOINTS line pair: {point_lines[i]} / {point_lines[i+1]}")

        endpoints.append((a[0], a[1], a[2]))
        endpoints.append((b[0], b[1], b[2]))

        la = a[3] if len(a) >= 4 else None
        lb = b[3] if len(b) >= 4 else None
        seg_labels.append((la, lb))

    # Labels: take first start label, then each segment end label
    labels = []
    labels.append(seg_labels[0][0] if seg_labels[0][0] else "P0")
    for la, lb in seg_labels:
        labels.append(lb if lb else f"P{len(labels)}")

    band_coords = "  ".join([f"{x} {y} {z}" for (x, y, z) in endpoints])
    band_labels = " ".join(labels)

    conf = []
    conf.append(dim_line)
    conf.append("")
    conf.append(f"BAND = {band_coords}")
    conf.append(f"BAND_POINTS = {nseg_points}")
    conf.append(f"BAND_LABELS = {band_labels}")
    conf.append("BAND_CONNECTION = .TRUE.")
    conf.append("")
    return "\n".join(conf)


def main():
    ap = argparse.ArgumentParser(description="Quick phonopy preview (FORCE_SETS + band.yaml).")
    ap.add_argument("--root", default=".", help="Run directory (should contain phonopy_disp.yaml). Default: current.")
    ap.add_argument(
        "--patterns",
        default="disp-*,pd-*,strain-*",
        help='Comma-separated dir globs to collect jobs. Default: "disp-*,pd-*,strain-*".'
    )
    ap.add_argument("--check", action="store_true", help="Check vasprun.xml completeness before running.")
    ap.add_argument("--dim", default="3 3 1", help='DIM for band.conf if not otherwise known. Default: "3 3 1".')
    args = ap.parse_args()

    root = Path(args.root).resolve()

    if not (root / "phonopy_disp.yaml").is_file() and not (root / "phonopy.yaml").is_file():
        print("[ERROR] phonopy_disp.yaml / phonopy.yaml not found in:", root)
        print("        Please run this script in the phonopy parent directory.")
        return 2

    pats = [p.strip() for p in args.patterns.split(",") if p.strip()]
    dirs = []
    for pat in pats:
        dirs.extend([d for d in root.glob(pat) if d.is_dir()])
    # de-dup
    uniq = {}
    for d in dirs:
        uniq[str(d)] = d
    dirs = sorted(uniq.values(), key=sort_key_dir)

    if not dirs:
        print(f"[ERROR] No directories matched patterns {pats} under:\n  {root}")
        print("        Hint: list subdirs and update --patterns accordingly.")
        return 2

    vxmls = []
    bad = []
    for d in dirs:
        v = find_vasprun_in_dir(d)
        if v is None:
            bad.append((d, "vasprun.xml not found"))
            continue
        if args.check and not vasprun_complete_fast(v):
            bad.append((d, f"vasprun.xml incomplete: {v}"))
            continue
        vxmls.append(v)

    if bad:
        print("[INCOMPLETE] Some jobs are not ready:")
        for d, reason in bad:
            print(f" - {d.name}: {reason}")
        print(f"\nSummary: {len(vxmls)}/{len(dirs)} ready, {len(bad)} not ready.")
        return 1

    print(f"[OK] Found {len(vxmls)} vasprun.xml files. Generating FORCE_SETS...")

    cmd_f = [sys.executable, "-m", "phonopy", "--vasp", "-f"] + [str(v) for v in vxmls]
    print("[RUN]", " ".join(cmd_f))
    r = subprocess.run(cmd_f, cwd=str(root))
    if r.returncode != 0:
        print("[ERROR] phonopy -f failed with code", r.returncode)
        return r.returncode

    # Build band.conf from QPOINTS if present
    qpoints = None
    for name in ("QPOINTS", "QPOINTS_OPT"):
        if (root / name).is_file():
            qpoints = root / name
            break

    if qpoints:
        dim_line = f"DIM = {args.dim}"
        conf_text = parse_qpoints_to_bandconf(qpoints, dim_line=dim_line)
        (root / "band.conf").write_text(conf_text, encoding="utf-8")
        print("[INFO] band.conf generated from", qpoints.name)

        cmd_p = [sys.executable, "-m", "phonopy", "-p", "band.conf"]
        print("[RUN]", " ".join(cmd_p))
        r2 = subprocess.run(cmd_p, cwd=str(root))
        if r2.returncode != 0:
            print("[ERROR] phonopy -p failed with code", r2.returncode)
            return r2.returncode

        print("[DONE] Generated FORCE_SETS and band.yaml in:", root)
        print("       If available: phonopy-bandplot band.yaml -o band.pdf")
    else:
        print("[DONE] Generated FORCE_SETS in:", root)
        print("       QPOINTS/QPOINTS_OPT not found, so band.yaml not generated.")
        print("       Put QPOINTS here or create band.conf manually, then run: python -m phonopy -p band.conf")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
