#!/usr/bin/env python3
# /// script
# requires-python = ">=3.14"
# dependencies = ["rich", "rich-argparse", "numpy", "scipy", "tifffile", "imagecodecs", "pillow"]
# ///
"""
hugin - one command for the Fuse and Pano Hugin workflows.

    uv run hugin.py fuse [options]     align and fuse a frame stack (noise reduction)
    uv run hugin.py pano [options]     stitch a panorama

Intermediates live in <outdir>/.hugin_tools and are deleted at the end unless
you ask for them with -k/--keep, a granular --keep-*, or -d/--develop.

Run `uv run hugin.py fuse --help` or `uv run hugin.py pano --help` for the full
option lists.
"""

import sys

from hugintools.cli import main

if __name__ == "__main__":
    sys.exit(main())
