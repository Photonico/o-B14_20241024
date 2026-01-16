# compress xml

#!/usr/bin/env python3
"""
vasprun_tools.py

Subcommands:
  compress         gzip-compress a file -> .gz (streaming)
  decompress       gunzip a .gz file -> original (streaming)
  merge            merge split vasprun parts into one XML by concatenating <calculation> blocks
  merge-compress   merge parts into XML, then gzip it

Notes:
- Compression does NOT change data; decompression restores exact bytes.
- XML merge assumes each part is a well-formed vasprun-like XML with <calculation> blocks.
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

XML_DECL = '<?xml version="1.0" encoding="UTF-8"?>\n'


def atomic_replace_write(path: str, write_fn):
    """Write to a temp file in same dir, then atomically replace."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmppath = tempfile.mkstemp(prefix=".tmp_vasprun_", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmppath, path)
    except Exception:
        try:
            os.remove(tmppath)
        except OSError:
            pass
        raise


def gzip_compress(inpath: str, outpath: str | None, level: int, keep: bool, deterministic: bool):
    if outpath is None:
        outpath = inpath + ".gz"
    if os.path.abspath(outpath) == os.path.abspath(inpath):
        raise ValueError("Output path must differ from input path.")

    mtime = 0 if deterministic else None

    def _writer(out_fh):
        with open(inpath, "rb") as src:
            with gzip.GzipFile(filename=os.path.basename(inpath), mode="wb", fileobj=out_fh, compresslevel=level, mtime=mtime) as gz:
                shutil.copyfileobj(src, gz, length=1024 * 1024)

    atomic_replace_write(outpath, _writer)

    if not keep:
        os.remove(inpath)


def gzip_decompress(inpath: str, outpath: str | None, keep: bool):
    if not inpath.endswith(".gz") and outpath is None:
        raise ValueError("Input doesn't end with .gz; please provide --out explicitly.")
    if outpath is None:
        outpath = inpath[:-3]

    if os.path.abspath(outpath) == os.path.abspath(inpath):
        raise ValueError("Output path must differ from input path.")

    def _writer(out_fh):
        with gzip.open(inpath, "rb") as src:
            shutil.copyfileobj(src, out_fh, length=1024 * 1024)

    atomic_replace_write(outpath, _writer)

    if not keep:
        os.remove(inpath)


def _root_open_tag(tag: str, attrib: dict) -> str:
    if not attrib:
        return f"<{tag}>\n"
    attrs = " ".join(f'{k}="{escape(v)}"' for k, v in attrib.items())
    return f"<{tag} {attrs}>\n"


def merge_vasprun_parts(out_xml: str, parts: list[str]):
    if not parts:
        raise ValueError("No part files provided.")

    # Parse first part to get root + header + calculations
    first = parts[0]
    root_tag = None
    root_attrib = None

    def _write_merged(out_fh):
        # write text to a binary fh
        def w(s: str):
            out_fh.write(s.encode("utf-8"))

        w(XML_DECL)

        stack = []
        saw_first_calc = False
        opened_root = False

        # -------- first part: write root open tag, header children, calculations --------
        for event, elem in ET.iterparse(first, events=("start", "end")):
            if event == "start":
                stack.append(elem)
                if len(stack) == 1:
                    nonlocal_root_tag = elem.tag
                    nonlocal_root_attrib = dict(elem.attrib)
                    # bind to outer
                    nonlocal root_tag, root_attrib
                    root_tag = nonlocal_root_tag
                    root_attrib = nonlocal_root_attrib
                    w(_root_open_tag(root_tag, root_attrib))
                    opened_root = True
                if len(stack) == 2 and elem.tag == "calculation":
                    saw_first_calc = True
            else:
                depth = len(stack)
                if depth == 2 and elem.tag == "calculation":
                    w(ET.tostring(elem, encoding="unicode"))
                    w("\n")
                    elem.clear()
                elif depth == 2 and (not saw_first_calc) and elem.tag != "calculation":
                    w(ET.tostring(elem, encoding="unicode"))
                    w("\n")
                    elem.clear()

                if depth >= 2 and elem.tag != "calculation":
                    elem.clear()
                stack.pop()

        if not opened_root or root_tag is None:
            raise RuntimeError(f"Failed to parse root from {first}")

        # -------- remaining parts: write only calculation blocks --------
        for p in parts[1:]:
            stack = []
            for event, elem in ET.iterparse(p, events=("start", "end")):
                if event == "start":
                    stack.append(elem)
                else:
                    depth = len(stack)
                    if depth == 1:
                        if elem.tag != root_tag:
                            raise RuntimeError(f"Root mismatch: {p} root <{elem.tag}> != <{root_tag}>")
                    if depth == 2 and elem.tag == "calculation":
                        w(ET.tostring(elem, encoding="unicode"))
                        w("\n")
                        elem.clear()

                    if depth >= 2 and elem.tag != "calculation":
                        elem.clear()
                    stack.pop()

        w(f"</{root_tag}>\n")

    atomic_replace_write(out_xml, _write_merged)


def merge_and_compress(out_gz: str, parts: list[str], level: int, deterministic: bool):
    # write merged xml into a temp file next to output, then compress to out_gz atomically
    outdir = os.path.dirname(os.path.abspath(out_gz)) or "."
    fd, tmpxml = tempfile.mkstemp(prefix=".tmp_merged_", suffix=".xml", dir=outdir)
    os.close(fd)
    try:
        merge_vasprun_parts(tmpxml, parts)
        gzip_compress(tmpxml, out_gz, level=level, keep=False, deterministic=deterministic)
    finally:
        try:
            if os.path.exists(tmpxml):
                os.remove(tmpxml)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_c = sub.add_parser("compress", help="gzip-compress a file")
    ap_c.add_argument("input", help="input file (e.g. vasprun.xml)")
    ap_c.add_argument("--out", help="output .gz path (default: input + .gz)", default=None)
    ap_c.add_argument("--level", type=int, default=6, help="gzip level 1-9 (default 6)")
    ap_c.add_argument("--rm", action="store_true", help="remove original after compress")
    ap_c.add_argument("--deterministic", action="store_true", help="set gzip mtime=0 for reproducible .gz bytes")

    ap_d = sub.add_parser("decompress", help="gunzip a file")
    ap_d.add_argument("input", help="input .gz file (e.g. vasprun.xml.gz)")
    ap_d.add_argument("--out", help="output path (default: strip .gz)", default=None)
    ap_d.add_argument("--rm", action="store_true", help="remove .gz after decompress")

    ap_m = sub.add_parser("merge", help="merge vasprun parts into one XML by <calculation> blocks")
    ap_m.add_argument("-o", "--out", required=True, help="output XML path (e.g. vasprun_merged.xml)")
    ap_m.add_argument("parts", nargs="+", help="part files in order (vasprun_1.xml vasprun_2.xml ...)")

    ap_mc = sub.add_parser("merge-compress", help="merge parts then gzip to .gz")
    ap_mc.add_argument("-o", "--out", required=True, help="output .gz path (e.g. vasprun_merged.xml.gz)")
    ap_mc.add_argument("--level", type=int, default=6, help="gzip level 1-9 (default 6)")
    ap_mc.add_argument("--deterministic", action="store_true", help="set gzip mtime=0 for reproducible .gz bytes")
    ap_mc.add_argument("parts", nargs="+", help="part files in order")

    args = ap.parse_args()

    if args.cmd == "compress":
        if not (1 <= args.level <= 9):
            raise SystemExit("level must be 1..9")
        gzip_compress(args.input, args.out, level=args.level, keep=not args.rm, deterministic=args.deterministic)
        print("OK:", (args.out or (args.input + ".gz")))
    elif args.cmd == "decompress":
        gzip_decompress(args.input, args.out, keep=not args.rm)
        print("OK:", (args.out or args.input[:-3]))
    elif args.cmd == "merge":
        merge_vasprun_parts(args.out, args.parts)
        print("OK:", args.out)
    elif args.cmd == "merge-compress":
        if not (1 <= args.level <= 9):
            raise SystemExit("level must be 1..9")
        merge_and_compress(args.out, args.parts, level=args.level, deterministic=args.deterministic)
        print("OK:", args.out)


if __name__ == "__main__":
    main()


#### how to use
# 1 compress:   python3 vasprun_tools.py compress vasprun.xml --level 9
# 2 decompress: python3 vasprun_tools.py decompress vasprun.xml.gz
