# hugin-tools

Two Hugin command-line workflows — **Fuse** (align and fuse a frame stack to
reduce noise) and **Pano** (stitch a panorama) — behind one `hugin` command,
sharing a common core (`hugintools/`).

## Requirements

- [uv](https://docs.astral.sh/uv/) (runs the tool and its Python dependencies)
- Hugin's command-line tools (`pto_gen`, `cpfind`, `cpclean`, `pto_var`,
  `autooptimiser`, `linefind`, `pano_modify`, `pano_trafo`, `hugin_executor`,
  `align_image_stack`, `hugin_stacker`) on `PATH`
- ImageMagick (`magick`/`convert`) — optional; needed for JPEGs and GIFs
- exiftool — optional; copies metadata onto the outputs
- Python 3.14+

## Usage

```
uv run hugin.py fuse [options]     # or:  hugin.bat fuse [options]
uv run hugin.py pano [options]     # or:  hugin.bat pano [options]
```

`fuse.bat` and `pano.bat` remain as shims (`hugin fuse` / `hugin pano`), so old
habits keep working. The command comes first; its options follow.

Run from the folder holding your images, or name them via `-i`/`-o`:

```
hugin.bat fuse --all
hugin.bat fuse --aligner all -i "D:\export" -o "D:\out"
hugin.bat pano -d
```

See `uv run hugin.py fuse --help` and `uv run hugin.py pano --help` for the full
option lists.

## Shared options

| Option | Meaning |
|---|---|
| `-i/--input`, `IMAGE...` | image files, glob patterns, or a folder (default: this folder) |
| `-o/--outdir` | output folder (default: current folder) |
| `--scratch DIR` | where intermediates go (default: `<outdir>/.hugin_tools`) |
| `-f/--force` | redo work even if its output already exists |
| `-k/--keep` | keep the scratch folder (every intermediate) |
| `--keep-aligned` / `--keep-merged` / `--keep-ptos` | keep individual kinds |
| `--overlays` | write the detection check sheets (mask / line overlays) and keep them |
| `-d/--develop` | keep every intermediate and overlay, and enable verbose logging |
| `-v/--verbose`, `--log FILE`, `--no-color` | transcript controls |

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
- outputs: `--mean`, `--winsor` (default), `--median`, `--clean`, `--action`,
  `--animation`/`--gif`; or `--all` for everything. Naming any output selects
  exactly those.
- `--mask-z`, `--mask-min-area`, `--mask-dilate`, `--mask-feather` tune motion
  detection.

## Pano specifics

- `--threshold`, `-l/--lines [N|filtered]`, `--line`, `--no-lines`
- `-s/--straighten`, `-c/--center`, `-p/--projection`, `--format`, `--prefix`
- `--aligned` renders the remapped frames (off by default; `-d` implies on)

## Changes from the standalone `fuse.py` / `pano.py`

The two scripts were merged into one command with a shared core:

- `fuse.py` → `hugin fuse`, `pano.py` → `hugin pano` (the `.bat` shims cover this)
- `--masks` (Fuse) and `--line-overlays` (Pano) are now the shared `--overlays`
- intermediates moved from the output folder into `<outdir>/.hugin_tools`
  (Pano's numbered `.pto` files included), swept unless kept
- new shared keep options: `-k/--keep`, `--keep-aligned`, `--keep-merged`,
  `--keep-ptos`
- Pano's `--aligned` now defaults **off** (`-d` implies on)
- `--keep-aligned` / `--keep-merged-tifs` (Fuse) are now the shared keep options

## Development

```
uvx ruff check .        # lint
uvx ruff format .       # format
uvx --with pytest pytest # tests
```
