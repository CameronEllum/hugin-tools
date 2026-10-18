# hugin-tools

Two Hugin command-line workflows — **Fuse** (align and fuse a frame stack to
reduce noise) and **Pano** (stitch a panorama) — behind one `hugin_tools`
command, sharing a common core (`hugin_tools/`).

## Requirements

- [uv](https://docs.astral.sh/uv/) (runs the tool and its Python dependencies)
- Hugin's command-line tools (`pto_gen`, `cpfind`, `cpclean`, `pto_var`,
  `autooptimiser`, `pano_modify`, `pano_trafo`, `hugin_executor`,
  `align_image_stack`, `hugin_stacker`) on `PATH`
- ImageMagick (`magick`/`convert`) — optional; needed for JPEGs and GIFs
- exiftool — optional; copies metadata onto the outputs
- Python 3.14+

The Python side (rich, numpy, scipy, Pillow, OpenCV) comes from the script header
in `hugin_tools.py`; uv installs it, so there is nothing else to set up.

## Usage

```
uv run hugin_tools.py fuse [options]     # or:  hugin_tools.bat fuse [options]
uv run hugin_tools.py pano [options]     # or:  hugin_tools.bat pano [options]
```

`fuse.bat` and `pano.bat` remain as shims (`hugin_tools fuse` / `hugin_tools
pano`), so old habits keep working. The command comes first; its options follow.

Run from the folder holding your images, or name them via `-i`/`-o`:

```
hugin_tools.bat fuse --all
hugin_tools.bat fuse --aligner all -i "D:\export" -o "D:\out"
hugin_tools.bat pano -d
```

See `uv run hugin_tools.py fuse --help` and `uv run hugin_tools.py pano --help`
for the full option lists.

## Shared options

| Option | Meaning |
|---|---|
| `-i/--input`, `IMAGE...` | image files, glob patterns, or a folder (default: this folder) |
| `-o/--outdir` | output folder (default: current folder) |
| `--scratch DIR` | where intermediates go (default: `<outdir>/.hugin_tools`) |
| `--clean` / `--no-clean` | clean control points with cpclean (default: on) |
| `--clean-sigma N` | drop a control point when its error is more than this many standard deviations above the mean (cpclean's own statistic; default: 3) |
| `-f/--force` | redo work even if its output already exists |
| `-k/--keep` | keep the scratch folder (every intermediate) |
| `--keep-aligned` / `--keep-merged` / `--keep-ptos` | keep individual kinds |
| `--overlays` | write the detection check sheets (mask / line overlays) and keep them |
| `-d/--develop` | keep every intermediate and overlay, and enable verbose logging |
| `-v/--verbose`, `--log FILE`, `--no-color` | transcript controls |

Both tools clean their control points the same way: cpclean removes the ones that
disagree with the rest, and `--clean-sigma` sets how far out a point has to be
before it goes. `--no-clean` keeps every control point the detector found.

## Intermediates and the scratch folder

Every intermediate — aligned frames, merged TIFFs, motion masks, line overlays,
and the numbered `.pto` projects — goes into `<outdir>/.hugin_tools/<engine>/`.
Deliverables (sharpened JPEGs, the GIF, the stitched panorama) stay in the output
folder.

By default the scratch folder is **deleted at the end of the run**, whether it
succeeded or failed. Keep it to inspect or to resume:

- `-k/--keep`, or a granular `--keep-aligned` / `--keep-merged` / `--keep-ptos`
- `-d/--develop` (keep everything + verbose + overlays)

## Fuse specifics

- `--aligner {ais,manual,all}`, `--motion {high,low}`, `--photometric`
- outputs: `--mean`, `--winsor` (default), `--median`, `--motion-clean`,
  `--motion-action`, `--animation`/`--gif`; or `--all` for everything. Naming any
  output selects exactly those.
- `--motion-clean` leaves whatever moved between frames out of the fusion;
  `--motion-action` composites every detected instance back in.
- `--mask-z`, `--mask-min-area`, `--mask-dilate`, `--mask-feather` tune motion
  detection.

## Pano specifics

- `-l/--lines [N]`, `--line`, `--no-lines` — vertical-line detection, on by
  default with no cap on how many lines a frame contributes
- `--line-min-length FRACTION` — the shortest segment worth keeping, as a share
  of the frame's long side (default: 0.01)
- `--clean-distance` / `--no-clean-distance`, `--threshold PX` — an extra,
  off-by-default hard cut on how far apart a pair may land in the panorama
- `-s/--straighten` (off by default), `-c/--center`
- `-p/--projection`, `--format`, `--prefix`
- `--aligned` renders the remapped frames (off by default; `-d` implies on)

Vertical lines are found with OpenCV's LSD and then checked against the vanishing
point they must share: the longest segments propose candidate points, each is
scored by the total length of the segments that agree with it, and only the
winner's bundle is written into the project as `t1` control points. Hugin's own
`linefind` is deliberately not used — "looks vertical in the picture" is not the
same as vertical in the world once the camera is pitched or rolled, while a
leaning edge can look perfectly upright. Because a `t1` line pins an image's roll
to the world's vertical, the fit levels the panorama by itself, so Hugin's
`--straighten` stays off: measured on the test panorama it only added about half a
degree of roll on top. Without lines, the view is aimed from the camera's own
pitch, as before. The last step reports how vertical the lines came out.

## Development

```
uvx ruff check .        # lint
uvx ruff format .       # format
uvx --with pytest --with numpy --with opencv-python-headless --with pillow pytest
```

The line-detector tests skip themselves when numpy, OpenCV or Pillow are missing,
so a plain `uvx --with pytest pytest` also passes; the full line above exercises
them.
