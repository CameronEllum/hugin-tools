#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["rich", "rich-argparse", "numpy", "scipy", "tifffile", "imagecodecs", "pillow"]
# ///
"""
Fuse - align and fuse a stack of frames to reduce noise.

Aligns a frame stack to each other, then combines (fuses) them - mean / median /
sigma-clipped / winsorised - to produce a noise-free image, plus a sharpened
JPEG and an animated GIF.

Cross-platform: works on Windows, Linux and macOS. Tool install locations are
not hard-coded - each program is looked up on PATH first, then in a few common
install folders.

Run it with uv (it pulls in `rich` automatically):

    uv run fuse.py                      # every image in this folder
    uv run fuse.py PXL_*.jpg            # only these - cmd.exe hands the pattern
    uv run fuse.py -i /photos/export -o /photos/fuse --aligner all

Inputs are whatever you name: image files, glob patterns, or a folder (a folder
is read as before).  `-i/--input` takes the same.  With no names at all, the
current folder is used.  Note that cmd.exe does not expand wildcards, so the
pattern arrives here intact and is expanded by fuse.py itself.

Layout of the output folder
---------------------------
    <outdir>/
        <name>_fuse_mean.jpg        sharpened JPEGs (the deliverables)
        <name>_fuse_winsor.jpg
        <name>_fuse_median.jpg
        <name>_fuse_clean.jpg       the frames with moving parts excluded
        <name>_fuse_action.jpg      every moving part composited back in
        <name>_fuse.gif
        <name>_fuse_*.tif           merged TIFFs  (deleted by default)
        aligned/                    aligned frames ("stack images"; deleted by default)
        masks/                      the motion masks + red overlays (--masks, or -d)

`<name>` is the last frame's file name, and its metadata is copied onto the outputs.
Everything except the JPEGs/GIF is deleted by default.  Use -d/--develop to keep
all intermediates and turn on verbose logging.

Moving parts
------------
A subject that moves between frames is averaged into a ghost by the mean-type
merges, and dropped altogether by the median (a pixel it occupies in fewer than
half the frames is an outlier there).  Neither is what you want when the point
is to show where the subject went:

    --clean  : leave the moving part out, keeping the noise reduction of the
               mean over the frames that did see background there.
    --action : composite every detected instance back at full opacity, so a
               gondola crossing the frame appears once per frame, not smeared.

Both come from one detection pass - the stack median is the reference, and
|frame - median| isolates whatever moved - so they cost the same together as
apart.  Fine-tune with --mask-z, --mask-min-area, --mask-dilate, --mask-feather;
--masks (implied by -d) keeps the masks and a check sheet for inspection.

A bare run produces the winsorised mean and nothing else.  --all asks for every
output type at once (mean, winsor, median, clean, action, animation); options are
read left to right, so a --no-<output> after it leaves that one out again.

Alignment engines
-----------------
    ais    : align_image_stack, non-default (full-res, dense, translation) [default]
    manual : Hugin cpfind -> cpclean -> autooptimiser -> nona remap
    all    : run manual into <outdir>/manual and ais into <outdir>/ais

align_image_stack aligns in place and cannot pad or rescale, so every frame must
have the same pixel size; mixed crops (e.g. one frame straightened in the raw
editor) are refused up front.  The manual engine remaps each frame onto a common
canvas and handles mixed sizes, so use that when the frames are not identical.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

try:
    from rich.console import Console
    from rich.markup import escape
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich_argparse import RawDescriptionRichHelpFormatter

    HAVE_RICH = True
    _HELP_FORMATTER = RawDescriptionRichHelpFormatter
except ImportError:  # keep working (plainer) if rich is somehow unavailable
    HAVE_RICH = False
    _HELP_FORMATTER = argparse.RawDescriptionHelpFormatter

# Only the motion features need these; every run gets them, because the script
# metadata above asks uv for them.
import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi


# ---------------------------------------------------------------------------
#  Console setup
# ---------------------------------------------------------------------------
def setup_console() -> None:
    """Force UTF-8 output (so the marks below render) and enable ANSI on Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    if os.name == "nt":
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_MARKUP = re.compile(r"\[/?[a-zA-Z0-9 #=/_.,'-]+\]")


def _plain(text: str) -> str:
    """Strip rich markup and ANSI codes (for the plain-text log)."""
    return _MARKUP.sub("", _ANSI.sub("", text))


# ---------------------------------------------------------------------------
#  A small presenter: rich when available, plain text otherwise.
# ---------------------------------------------------------------------------
class UI:
    """All user-facing output goes through here, so it stays consistent."""

    def __init__(self, verbose: bool = False, no_color: bool = False, log_path: str | None = None):
        self.verbose = verbose
        self._fh = None
        if log_path:
            try:
                self._fh = open(log_path, "w", encoding="utf-8")
            except OSError as exc:
                print(f"Warning: cannot write log {log_path}: {exc}", file=sys.stderr)
        self._plain: list[str] = []
        self.console = (
            Console(no_color=no_color, highlight=False, emoji=True) if HAVE_RICH else None
        )

    # -- internals ----------------------------------------------------------
    def _log(self, text: str) -> None:
        if self._fh:
            self._fh.write(_plain(text) + "\n")
            self._fh.flush()

    # -- output -------------------------------------------------------------
    def say(self, markup: str, console=None) -> None:
        """Print a line, optionally through a live progress console."""
        if console is not None:
            console.print(markup)
        elif self.console is not None:
            self.console.print(markup)
        else:
            line = _plain(markup)
            print(line, flush=True)
        self._log(markup)

    def print(self, markup: str = "") -> None:
        self.say(markup)

    def dim(self, text: str, console=None) -> None:
        """Print tool output / command lines, de-emphasised."""
        for line in text.splitlines() or [""]:
            if console is not None:
                console.print(f"[grey50]    {escape(line)}[/grey50]")
            elif self.console is not None:
                self.console.print(f"[grey50]    {escape(line)}[/grey50]")
            else:
                print(f"    {line}", flush=True)
        self._log("\n".join("    " + ln for ln in (text.splitlines() or [""])))

    def cmd(self, argv: list[str], console=None) -> None:
        self.say(f"[grey50]    $ {escape(' '.join(argv))}[/grey50]", console)

    # -- blocks -------------------------------------------------------------
    def header(
        self,
        title: str,
        count: int,
        source: Path | str,
        frames: list[Path],
        extra: list[str] | None = None,
    ) -> None:
        """Program name, inputs, frame names, then a Progress: heading."""
        noun = "frame" if count == 1 else "frames"
        self.print(f"📷 [bold]{title}[/bold]  ·  {count} {noun}")
        self.print("")
        self.print(f"📁 [grey50]{escape(str(source))}[/grey50]")
        for p in frames:
            self.print(f"  📄 [grey50]{escape(p.name)}[/grey50]")
        for line in extra or []:
            self.print(line)
        self.print("")
        self.print("▶️ [bold]Progress:[/bold]")

    def rule(self, title: str) -> None:
        self.print("")
        if self.console is not None:
            self.console.rule(f"[bold]{escape(title)}[/bold]")
        else:
            self.print(f"=== {title} ===")
        self.print("")

    def summary(self, results: list[tuple[str, Path]]) -> None:
        """An unboxed, grouped listing - no table borders, just aligned text."""
        if not results:
            return
        self.print("")
        self.print("📦 [bold]Outputs[/bold]")
        groups: dict[str, list[Path]] = {}
        for mode, path in results:
            groups.setdefault(mode, []).append(path)
        for mode, paths in groups.items():
            self.print("")
            self.print(f"📂 [bold cyan]{mode.upper()}[/bold cyan]")
            width = max(len(p.name) for p in paths)
            for p in paths:
                try:
                    size = f"{p.stat().st_size / 1_048_576:.1f} MB"
                except OSError:
                    size = ""
                pad = " " * (width - len(p.name))
                self.say(f"    📄 {escape(p.name)}{pad}  [grey50]{size:>8}[/grey50]")
        self.print("")

    def close(self) -> None:
        if self._fh:
            self._fh.write("\n".join(self._plain) + "\n")
            self._fh.close()
            self._fh = None


# ---------------------------------------------------------------------------
#  Tool discovery
# ---------------------------------------------------------------------------
def find_tool(name: str) -> str | None:
    """Return the full path to a program, or None if not found."""
    exe = name + (".exe" if os.name == "nt" else "")
    found = shutil.which(name)
    if found:
        return found

    extra_dirs: list[str] = []
    if os.name == "nt":
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        extra_dirs += [os.path.join(pf, "Hugin", "bin"), r"C:\Program Files\Hugin\bin"]
        extra_dirs += [str(p) for p in Path(pf).glob("ImageMagick-*")]
    else:
        extra_dirs += ["/usr/bin", "/usr/local/bin", "/snap/bin", "/opt/homebrew/bin"]

    for d in extra_dirs:
        cand = Path(d) / exe
        if cand.exists():
            return str(cand)
    return None


def resolve_tools() -> dict:
    """Find every external program we need. Raise if a required one is missing."""
    required = [
        "pto_gen",
        "cpfind",
        "cpclean",
        "pto_var",
        "autooptimiser",
        "pano_modify",
        "hugin_executor",
        "align_image_stack",
        "hugin_stacker",
    ]
    tools = {name: find_tool(name) for name in required}
    tools["magick"] = find_tool("magick") or find_tool("convert")  # IM7 / IM6
    tools["exiftool"] = find_tool("exiftool")  # optional

    missing = sorted(n for n in required if not tools.get(n))
    if missing:
        raise SystemExit(
            "Missing required tool(s): "
            + ", ".join(missing)
            + "\nInstall Hugin (and its command-line tools) and make sure they are on PATH."
        )
    return tools


# ---------------------------------------------------------------------------
#  Source frames
# ---------------------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def is_our_output(path: Path) -> bool:
    """Our own results (…_fuse_…, and legacy …_MFNR_…) are never input frames."""
    return "_fuse" in path.stem or "_MFNR" in path.stem


def collect_frames(specs: list[str]) -> tuple[list[Path], list[Path], list[str]]:
    """Frames to fuse, from folders, files or glob patterns.

    Whatever the shell hands over: a folder is read as before, a pattern the
    shell did not expand is expanded here (cmd.exe leaves wildcards alone, unlike
    PowerShell), and explicit files are taken as given.  Returns
    (frames, skipped, unmatched) - skipped are previous Fuse outputs, unmatched
    are names that matched nothing or are not images, so the caller can say so
    rather than quietly use fewer frames.
    """
    frames: list[Path] = []
    skipped: list[Path] = []
    unmatched: list[str] = []
    for spec in specs:
        text = str(spec)
        path = Path(text)
        if path.is_dir():
            found = [p for p in sorted(path.iterdir()) if is_image(p)]
        elif glob.has_magic(text):
            found = [Path(m) for m in sorted(glob.glob(text, recursive=True)) if is_image(Path(m))]
        elif is_image(path):
            found = [path]
        else:
            found = []
        if not found:
            unmatched.append(text)
        for p in found:
            (skipped if is_our_output(p) else frames).append(p.resolve())

    frames = sorted(dict.fromkeys(frames), key=lambda p: p.name)
    if not frames:
        raise SystemExit(
            "No source frames found in: "
            + ", ".join(specs)
            + "\nName image files, a glob pattern, or a folder holding them"
            f" ({', '.join(sorted(IMAGE_EXTS))})."
        )
    return frames, skipped, unmatched


# ---------------------------------------------------------------------------
#  Frame geometry (read from the file header - no image library needed)
# ---------------------------------------------------------------------------
def _jpeg_size(fh) -> tuple[int, int] | None:
    """Width/height from a JPEG's start-of-frame marker."""
    fh.seek(2)
    while True:
        byte = fh.read(1)
        if not byte:
            return None
        if byte != b"\xff":
            continue
        code = fh.read(1)
        while code == b"\xff":  # fill bytes between segments
            code = fh.read(1)
        if not code:
            return None
        marker = code[0]
        if marker in (0x00, 0x01) or 0xD0 <= marker <= 0xD9:  # padding / restarts
            continue
        raw = fh.read(2)
        if len(raw) < 2:
            return None
        length = int.from_bytes(raw, "big")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            data = fh.read(5)  # precision, height, width
            if len(data) < 5:
                return None
            return int.from_bytes(data[3:5], "big"), int.from_bytes(data[1:3], "big")
        if marker == 0xDA:  # start of scan, so there was no frame header
            return None
        if length < 2:
            return None
        fh.seek(length - 2, 1)


def _tiff_size(fh) -> tuple[int, int] | None:
    """Width/height from a TIFF's first image file directory, SubIFDs included."""
    fh.seek(0)
    head = fh.read(8)
    if len(head) < 8:
        return None
    endian = "little" if head[:2] == b"II" else "big"
    if int.from_bytes(head[2:4], endian) != 42:
        return None

    queue = [int.from_bytes(head[4:8], endian)]
    seen: set[int] = set()
    while queue:
        offset = queue.pop(0)
        if not offset or offset in seen:
            continue
        seen.add(offset)
        fh.seek(offset)
        raw = fh.read(2)
        if len(raw) < 2:
            continue
        count = int.from_bytes(raw, endian)
        blob = fh.read(12 * count)
        values: dict[int, int] = {}
        for i in range(count):
            entry = blob[i * 12 : i * 12 + 12]
            if len(entry) < 12:
                break
            tag = int.from_bytes(entry[0:2], endian)
            typ = int.from_bytes(entry[2:4], endian)
            num = int.from_bytes(entry[4:8], endian)
            size = {1: 1, 3: 2, 4: 4}.get(typ, 0)  # BYTE / SHORT / LONG
            if not size or not num:
                continue
            if size * num <= 4:  # value fits inline, left-justified
                values[tag] = int.from_bytes(entry[8 : 8 + size], endian)
            else:
                back = fh.tell()
                fh.seek(int.from_bytes(entry[8:12], endian))
                values[tag] = int.from_bytes(fh.read(size), endian)
                fh.seek(back)
            if tag == 330:  # SubIFDs: raw-ish files keep the size there
                queue.append(values[tag])
        if 256 in values and 257 in values:  # ImageWidth / ImageLength
            return values[256], values[257]
    return None


def frame_size(path: Path) -> tuple[int, int] | None:
    """Pixel dimensions from an image file's header, or None if unrecognised."""
    try:
        with path.open("rb") as fh:
            head = fh.read(8)
            if head == b"\x89PNG\r\n\x1a\n":
                fh.seek(16)
                data = fh.read(8)
                if len(data) < 8:
                    return None
                return int.from_bytes(data[:4], "big"), int.from_bytes(data[4:8], "big")
            if head[:2] == b"\xff\xd8":
                return _jpeg_size(fh)
            if head[:2] in (b"II", b"MM"):
                return _tiff_size(fh)
    except OSError:
        return None
    return None


def check_same_size(frames: list[Path]) -> None:
    """Refuse a frame stack of mixed pixel sizes before ais is asked to align it.

    align_image_stack works in place and has no way to pad or rescale, so it
    fails with 'images have different sizes'. Frames that were cropped or
    straightened differently on export hit this, and the tool's own error is
    easy to miss - so list the sizes and point at the engine that can cope.
    Unknown sizes (an exotic format) are left to ais to complain about.
    """
    sizes = [(p, frame_size(p)) for p in frames]
    if any(s is None for _, s in sizes) or len({s for _, s in sizes}) == 1:
        return
    listing = "\n".join(f"    {p.name}  {w} x {h}" for p, (w, h) in sizes)
    raise SystemExit(
        f"Frames have different pixel sizes:\n{listing}\n"
        "align_image_stack (the default 'ais' aligner) aligns in place and needs every\n"
        "frame to be the same size - it cannot pad or rescale.\n"
        "Re-export the frames with identical crops and straightening, or run with\n"
        "--aligner manual, which remaps mixed sizes onto one canvas."
    )


# ---------------------------------------------------------------------------
#  Moving parts: find them against the stack median, then fuse around them
# ---------------------------------------------------------------------------
# A frame whose difference from the plate is far above the others' is globally
# unlike them: soft from camera shake, or exposed differently.  It is reported,
# never silently tolerated - widening that frame's threshold would eat the very
# subject the mask is looking for, which is a far worse failure than a broad mask.
UNLIKE_FRAME_RATIO = 1.5


def run_motion_fusion(
    aligned: list[Path],
    out_clean: Path | None,
    out_action: Path | None,
    motion: MotionOptions,
    masks_dir: Path | None = None,
) -> str:
    """Fuse an aligned stack so that whatever moved is excluded, or kept per frame.

    The median of the stack is the reference: a pixel the subject occupies in
    fewer than half the frames is background there, and background is what the
    median shows, so |frame - median| isolates the subject in each frame.

    out_clean   every moving pixel excluded, so the background averages only over
                frames that really saw background there - no ghost.
    out_action  each detected instance composited back at full opacity, in frame
                order, onto that clean plate.
    masks_dir   write the per-frame masks there (debug; see -d / --masks).

    Returns a one-line report for the transcript.
    """
    n = len(aligned)
    if n < 3:
        raise RuntimeError(
            f"motion masking needs at least 3 aligned frames, got {n} - with fewer,"
            " the stack median cannot reject something that moved"
        )

    def rgb_and_valid(img):
        """Split one frame into (RGB, validity).  nona writes RGBA canvases."""
        if img.ndim == 2:
            return np.repeat(img[:, :, None], 3, axis=2), np.ones(img.shape, bool)
        if img.shape[2] == 2:  # grey + alpha
            return np.repeat(img[:, :, :1], 3, axis=2), img[:, :, 1] > 0
        if img.shape[2] >= 4:  # RGBA: alpha marks the remapped area, not colour
            return img[:, :, :3], img[:, :, 3] > 0
        return img, np.ones(img.shape[:2], bool)

    frames = [tifffile.imread(p) for p in aligned]
    shapes = {f.shape[:2] for f in frames}
    if len(shapes) > 1:
        raise RuntimeError(f"aligned frames are not all the same size: {sorted(shapes)}")

    dtype = frames[0].dtype
    max_val = float(np.iinfo(dtype).max)
    parts = [rgb_and_valid(f) for f in frames]
    shape = parts[0][0].shape
    min_area = motion.min_area
    if min_area <= 0:  # scale the floor with the frame so it suits any camera
        min_area = max(300, round(0.0002 * shape[0] * shape[1]))

    def channel(img, c):
        return img[:, :, c].astype(np.float32)

    # The median plate: background, with anything that moved left out of it.
    stack = np.stack([rgb for rgb, _ in parts])
    if n % 2:
        middle = n // 2
        plate = np.partition(stack, middle, axis=0)[middle].copy()
    else:  # an even stack has no middle frame; average the two middle ones
        plate = np.median(stack.astype(np.float32), axis=0)
    del stack

    # Photometry.  One frame in a stack is often brighter than the rest, because
    # the light or the meter changed between shots, and left alone that whole
    # frame reads as "moved": the sky and every flat surface light up at once.
    # Match each frame per channel to the plate before looking for differences,
    # and keep the matched frames for the output, so a pasted instance cannot
    # carry a different exposure onto the plate either.
    gains: list[list[float]] = []
    for rgb, valid in parts:
        frame_gain, usable = [], bool(valid.any())
        for c in range(3):
            level = float(channel(rgb, c)[valid].mean()) if usable else 0.0
            reference = float(channel(plate, c)[valid].mean()) if usable else 0.0
            want = reference / level if level > 0 else 1.0
            frame_gain.append(min(max(want, 0.7), 1.4))  # never a wild correction
        for c in range(3):
            if frame_gain[c] != 1.0:
                scaled = channel(rgb, c) * frame_gain[c]
                np.clip(scaled, 0.0, max_val, out=scaled)
                rgb[:, :, c] = scaled.astype(rgb.dtype)
        gains.append(frame_gain)

    open_s = ndi.iterate_structure(ndi.generate_binary_structure(2, 1), 2)
    close_s = ndi.iterate_structure(ndi.generate_binary_structure(2, 1), 5)
    square = np.ones((3, 3), bool)

    # Where the median has real background behind it: a pixel painted by only
    # one frame sits on a black hole in the plate otherwise, and that hole reads
    # as a huge difference - i.e. as movement.
    valid_stack = np.stack([valid for _, valid in parts])
    painted = np.any(valid_stack, axis=0)
    known = valid_stack.sum(axis=0) > n // 2

    # Difference of every frame from the plate, and how unlike the others each
    # frame is overall.  That second number is a diagnostic, not a tolerance: a
    # globally off frame (shake, exposure) shows up as a broad mask, and it is
    # much better to say so than to widen the threshold and lose its subject.
    diffs: list = []
    unlike: list[float] = []
    for rgb, _valid in parts:
        diff = np.zeros(rgb.shape[:2], np.float32)
        for c in range(3):
            np.maximum(diff, np.abs(channel(rgb, c) - channel(plate, c)), out=diff)
        diff = ndi.gaussian_filter(diff, 2.0)
        diff[~known] = 0.0  # no background in the plate there, so nothing to compare
        diffs.append(diff)
        unlike.append(float(np.median(diff)))

    # How much a pixel may differ between the aligned frames without anything
    # having moved: the stack's own disagreement there, not the scene's local
    # texture.  Static detail is identical in every aligned frame, so a busy wall
    # must not raise the bar above a low-contrast subject - which is exactly how
    # a background ends up showing through a person.  Only real disagreement
    # (sensor noise, residual misregistration, JPEG) may raise it.
    #
    # The median across frames is the right aggregator: a subject standing in one
    # frame cannot move it, while several frames that sit differently - exposure
    # drift, a touch of misregistration - can, which is what should be tolerated.
    spread = np.empty(shape[:2], np.float32)
    for y0 in range(0, shape[0], 256):  # banded, so the stack copy stays small
        y1 = min(y0 + 256, shape[0])
        band = np.stack([d[y0:y1] for d in diffs], axis=0)
        spread[y0:y1] = np.median(band, axis=0)
        del band
    # Spread it a little, so one quiet pixel cannot tighten the bar for its
    # neighbours, where the registration is really no better.
    local = ndi.maximum_filter(spread, size=9).astype(np.float32)
    del spread

    # Two floors.  An absolute one, so flat dark areas do not divide by ~nothing,
    # and a relative one, because a flat *bright* area (sky) has almost no spread
    # of its own and would otherwise call any small exposure or tone-curve
    # difference between frames a moving object.
    mean = ndi.uniform_filter(plate.max(axis=-1).astype(np.float32), 15)
    local = np.sqrt(local * local + (0.04 * mean) ** 2 + (0.015 * max_val) ** 2)
    del mean

    typical = float(np.median(unlike)) if unlike else 0.0
    unlike_frames = [
        i + 1
        for i, u in enumerate(unlike)
        if typical > 0 and u >= 0.005 * max_val and u / typical > UNLIKE_FRAME_RATIO
    ]

    hard: list = []  # exclusion: the pixels the subject owns in this frame
    soft: list = []  # inclusion: the same, feathered for compositing, as uint8
    coverage: list[float] = []

    found = 0
    for k, (_rgb, valid) in enumerate(parts):
        diff = diffs[k]
        diffs[k] = None  # release frame by frame: these are full-frame arrays
        mask = (diff / local) > motion.z_thresh
        del diff
        mask &= valid & known  # ignore the unpainted border of the canvas
        mask = ndi.binary_fill_holes(ndi.binary_closing(ndi.binary_opening(mask, open_s), close_s))
        # Judge features *before* growing them: on a textured surface (stone,
        # foliage) the residual differences of static detail are small and thin,
        # but once dilated they merge into one huge blob and sail past the area
        # floor.  Keep the blobs that really are a feature, plus their own
        # smaller parts (shoes, a carriage) that sit within reach of one.
        labels, count = ndi.label(mask)
        if count:
            areas = np.bincount(labels.ravel())
            areas[0] = 0  # label 0 is the background
            big = areas >= min_area
            found += int(big.sum())
            if not big.any():
                mask[:] = False
            else:
                reach = ndi.binary_dilation(big[labels], square, iterations=motion.dilate)
                mask &= reach
        del labels
        # Grow what is left over the subject's travel between frames and its blur
        # within one frame.
        grown = ndi.binary_dilation(mask, square, iterations=motion.dilate)
        hard.append(grown)
        feathered = ndi.gaussian_filter(grown.astype(np.float32), motion.feather)
        soft.append(np.rint(np.clip(feathered, 0.0, 1.0) * 255.0).astype(np.uint8))
        coverage.append(100.0 * grown.sum() / max(int(painted.sum()), 1))
    del diffs

    # Clean plate: weighted mean over the frames that saw background at a pixel.
    acc = np.zeros(shape, np.float32)
    weight = np.zeros(shape[:2], np.float32)
    for (rgb, valid), masked in zip(parts, hard, strict=True):
        w = np.logical_and(valid, ~masked).astype(np.float32)
        for c in range(3):
            acc[:, :, c] += channel(rgb, c) * w
        weight += w
    weak = weight <= 0.05  # the subject covered this pixel in every frame
    fine = weight.copy()
    fine[weak] = 1.0
    acc /= fine[:, :, None]
    if weak.any():
        acc[weak] = plate[weak].astype(np.float32)

    # The thin band only one or two frames painted: there is no averaged
    # background to be had and the plate is a hole there, so borrow a real pixel
    # from whichever frame has one instead of letting the edge fall to black.
    gaps = painted & ~known
    if gaps.any():
        for rgb, valid in parts:
            fill = gaps & valid
            if fill.any():
                acc[fill] = rgb[fill].astype(np.float32)
                gaps &= ~fill

    alpha = painted.astype(dtype) * np.iinfo(dtype).max

    def write(path: Path, rgb) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        img = np.clip(np.rint(rgb), 0, max_val).astype(dtype)
        tifffile.imwrite(path, np.dstack([img, alpha]), compression="lzw")

    if out_clean is not None:
        write(out_clean, acc)

    if out_action is not None:
        # Compose by ownership rather than by order.  Where several frames claim
        # the same pixel - the subject's body overlapping between frames, or a
        # mask straying onto the background - the strongest claim wins, so a
        # later frame's static background can never be pasted over an earlier
        # instance.  The winning alpha is then used to feather the edge against
        # the clean plate, so interiors stay solid and boundaries stay smooth.
        best = np.zeros(shape[:2], np.float32)
        owner = np.zeros(shape, np.float32)
        for (rgb, _valid), mask255 in zip(parts, soft, strict=True):
            mask = mask255.astype(np.float32) * (1.0 / 255.0)
            stronger = mask > best
            if not stronger.any():
                continue
            for c in range(3):
                owner[:, :, c][stronger] = channel(rgb, c)[stronger]
            best[stronger] = mask[stronger]
        composed = acc * (1.0 - best[:, :, None]) + owner * best[:, :, None]
        write(out_action, composed)

    if masks_dir is not None:
        # Keep the masks themselves, plus a look-at-me version of each frame with
        # the mask painted over it in red, the way a mask reads in Lightroom.
        # Done here because the frames and masks are already in memory.
        masks_dir.mkdir(parents=True, exist_ok=True)
        red = np.array([255.0, 0.0, 0.0], np.float32)
        for src, (rgb, _valid), mask255, masked in zip(aligned, parts, soft, hard, strict=True):
            tifffile.imwrite(masks_dir / f"{src.stem}_mask.tif", mask255)
            outline = masked & ~ndi.binary_erosion(masked, square, iterations=3)
            tint = mask255.astype(np.float32) * (0.4 / 255.0)
            over = rgb.astype(np.float32) * (1.0 - tint[:, :, None]) + red * tint[:, :, None]
            over[outline] = over[outline] * 0.2 + red * 0.8
            image = Image.fromarray(np.clip(over, 0, 255).astype(np.uint8))
            image.thumbnail((1600, 1600))  # small enough to flip through
            image.save(masks_dir / f"{src.stem}_overlay.jpg", quality=88)

    report = (
        f"{found} moving instance(s) over {n} frames; masks cover "
        + "/".join(f"{c:.2f}%" for c in coverage)
        + f" of each frame, grown by {motion.dilate}px"
        + f", blobs under {min_area}px dropped"
    )
    if masks_dir is not None:
        report += f"; masks and red overlays in {masks_dir.name}"
    if unlike_frames:
        report += (
            "; WARNING: frame(s) "
            + ", ".join(str(i) for i in unlike_frames)
            + " unlike the rest (softer or exposed differently?) - a broad mask there"
            " is that frame, not movement"
        )
    if found == 0:
        report += " - nothing moved, so clean and action match a plain mean"
    elif max(coverage) > 10.0:
        report += " - WARNING: that is a lot of the frame; check --mask-z or the stack"
    return report


# ---------------------------------------------------------------------------
#  The plan: an ordered list of steps, built up-front so the progress bar can
#  show real progress, then executed one at a time.
# ---------------------------------------------------------------------------
@dataclass
class Step:
    label: str
    cmd: list[str] | None = None
    cmd_fn: Callable[[], list[str]] | None = None  # built when the step runs
    run_fn: Callable[[], str | None] | None = None  # in-process Python step
    note: str | None = None  # informational step: no command, just a message
    on_done: Callable[[], None] | None = None


# Merge types: name -> (extra hugin_stacker args, friendly label).
MERGE_TYPES = {
    "mean": (["--mode=sigma", "--max-sigma=2"], "sigma-clipped mean"),
    "winsor": (["--mode=winsor", "--winsor-trim=0.2"], "winsorised mean"),
    "median": (["--mode=median"], "median"),
}


@dataclass
class MotionOptions:
    """What to do about the parts of a stack that move between frames."""

    clean: bool = False  # emit the plate with the moving parts excluded
    action: bool = False  # emit the composite of every moving instance
    masks: bool = False  # keep the masks and a check sheet
    z_thresh: float = 3.5  # how far from the stack median a pixel must sit
    min_area: int = 0  # drop blobs below this many pixels (0 = scale with frame)
    dilate: int = 12  # grow each mask by this radius, in pixels
    feather: float = 4.0  # soften the pasted edge by this sigma, in pixels

    @property
    def enabled(self) -> bool:
        return self.clean or self.action


def manual_steps(
    tools: dict, frames: list[Path], aligned_dir: Path, motion: str, photometric: bool
) -> list[Step]:
    """The original Hugin pipeline."""
    t = tools
    img, cps, clean = (
        aligned_dir / "images.pto",
        aligned_dir / "control_points.pto",
        aligned_dir / "cleaned.pto",
    )
    s0, s1, s2 = aligned_dir / "stack0.pto", aligned_dir / "stack1.pto", aligned_dir / "stack2.pto"
    frame_args = [str(p) for p in frames]

    steps = [
        Step("Reading frames", [t["pto_gen"], "-o", str(img), *frame_args]),
        Step(
            "Finding matching features",
            [
                t["cpfind"],
                f"--output={cps}",
                "--sieve2width=7",
                "--sieve2height=7",
                "--sieve2size=3",
                "--fullscale",
                "--celeste",
                "--ncores=3",
                str(img),
            ],
        ),
        # Clean BEFORE optimising so outliers don't skew the geometry.
        Step(
            "Removing bad matches",
            [t["cpclean"], "--whole-pano-checking", f"--output={clean}", str(cps)],
        ),
    ]

    opt = "y,p,r" if motion == "low" else "y,p,r,TrX,TrY,TrZ,v"
    steps.append(
        Step(
            "Choosing alignment parameters", [t["pto_var"], "--opt", opt, "-o", str(s0), str(clean)]
        )
    )
    steps += [
        Step("Optimising alignment (pass 1)", [t["autooptimiser"], "-n", "-o", str(s1), str(s0)]),
        Step("Refining matches", [t["cpclean"], f"--output={s1}", "--max-distance=3.3", str(s1)]),
        Step("Optimising alignment (pass 2)", [t["autooptimiser"], "-n", "-o", str(s2), str(s1)]),
    ]
    if photometric:
        # Solve per-frame exposure / vignetting / white balance so all frames
        # match before they are averaged.
        steps.append(
            Step(
                "Matching exposure, vignetting and white balance",
                [t["autooptimiser"], "-m", "-o", str(s2), str(s2)],
            )
        )

    steps += [
        Step(
            "Preparing output",
            [
                t["pano_modify"],
                "-o",
                str(s2),
                "--center",
                "--canvas=AUTO",
                "--crop=AUTO",
                "--projection=0",
                "--output-uncropped-tiff",
                "--output-type=REMAP",
                str(s2),
            ],
        ),
        Step(
            "Rendering aligned frames",
            [t["hugin_executor"], "--stitching", f"--prefix={aligned_dir / 'rendered_'}", str(s2)],
        ),
    ]
    return steps


def ais_steps(tools: dict, frames: list[Path], aligned_dir: Path) -> list[Step]:
    """align_image_stack with stronger-than-default settings."""
    return [
        Step(
            "Aligning frames",
            [
                tools["align_image_stack"],
                "-a",
                str(aligned_dir / "ais_"),
                "-C",
                "-s",
                "0",
                "-c",
                "50",
                "-g",
                "10",
                "-t",
                "3",
                "-x",
                "-y",
                "-z",
                "--use-given-order",
                *[str(p) for p in frames],
            ],
        )
    ]


def output_steps(
    tools: dict,
    outdir: Path,
    aligned_dir: Path,
    ref: Path,
    merge_types: list[str],
    keep_merged: bool,
    keep_aligned: bool,
    motion: MotionOptions,
    animation: bool,
) -> list[Step]:
    """Combine / sharpen / (delete) steps for one engine's output folder."""
    base = f"{ref.stem}_fuse"
    stacker = tools["hugin_stacker"]
    magick = tools.get("magick")
    steps: list[Step] = []

    def aligned_frames() -> list[Path]:
        # Resolved at run time, so it works whether alignment was skipped or
        # just performed. (Only aligned frames live in aligned_dir.)
        found = sorted(p for p in aligned_dir.glob("*.tif") if not p.name.endswith("_mask.tif"))
        if not found:
            raise RuntimeError(f"No aligned frames found in {aligned_dir}")
        return found

    def jpeg_steps(tif: Path) -> list[Step]:
        """The sharpened JPEG for one merged product, plus its TIFF cleanup."""
        if not magick:
            # Without ImageMagick the TIFF is the only artefact there is.
            if keep_merged:
                return [Step(f"Merged TIFF ready: {tif.name}", note=f"kept {tif.name}")]
            return []
        jpg = tif.with_suffix(".jpg")

        def sharpen_step(tif=tif, jpg=jpg):
            return [magick, str(tif), "-unsharp", "0x0.7+1.0+0", "-quality", "95", str(jpg)]

        # Keep the TIFF if asked, or when there is no ImageMagick to make a JPEG.
        def delete_tif(tif=tif):
            if tif.exists():
                tif.unlink()

        return [
            Step(
                f"Sharpening {jpg.name}",
                cmd_fn=sharpen_step,
                on_done=None if keep_merged else delete_tif,
            )
        ]

    for name in merge_types:
        extra, label = MERGE_TYPES[name]
        tif = outdir / f"{base}_{name}.tif"

        def combine_step(tif=tif, extra=extra):
            return [stacker, *extra, f"--output={tif}", *[str(p) for p in aligned_frames()]]

        steps.append(Step(f"Combining frames ({label})", cmd_fn=combine_step))
        steps += jpeg_steps(tif)

    # Whatever moves between frames: one detection pass feeds both products.
    if motion.enabled:
        clean_tif = outdir / f"{base}_clean.tif" if motion.clean else None
        action_tif = outdir / f"{base}_action.tif" if motion.action else None
        masks_dir = outdir / "masks" if motion.masks else None

        def motion_step(clean_tif=clean_tif, action_tif=action_tif, masks_dir=masks_dir):
            return run_motion_fusion(aligned_frames(), clean_tif, action_tif, motion, masks_dir)

        steps.append(Step("Detecting motion", run_fn=motion_step))
        for tif in (clean_tif, action_tif):
            if tif is not None:
                steps += jpeg_steps(tif)

    if magick and animation:
        gif = outdir / f"{base}.gif"

        def gif_step(gif=gif):
            return [
                magick,
                *[str(p) for p in aligned_frames()],
                "-resize",
                "1080x",
                "-delay",
                "30",
                "-loop",
                "0",
                str(gif),
            ]

        steps.append(Step(f"Building animation {gif.name}", cmd_fn=gif_step))
    elif not magick:
        steps.append(
            Step(
                "ImageMagick not found - skipping JPEG/animation",
                note="ImageMagick not found - skipping JPEG/animation",
            )
        )

    exiftool = tools.get("exiftool")
    if exiftool:

        def tag_step():
            targets = [p for p in sorted(outdir.glob(f"{base}_*")) if p.suffix in (".jpg", ".tif")]
            argv = [exiftool, "-overwrite_original"]
            # exiftool needs "-charset filename=utf8" on Linux/macOS for accented
            # paths, but on Windows it uses wide filenames and that option makes
            # it fail to find the file, so omit it there.
            if os.name != "nt":
                argv += ["-charset", "filename=utf8"]
            # -Orientation#=1 forces the tag to "normal" (the "#" means numeric):
            # hugin_stacker writes its TIFFs tagged "rotate 180", which flips them
            # in viewers even though the pixels are fine.
            argv += ["-TagsFromFile", str(ref), "-all:all", "-Orientation#=1"]
            argv += [str(p) for p in targets]
            return argv

        steps.append(Step(f"Copying metadata from {ref.name}", cmd_fn=tag_step))

    # The aligned frames are working files; remove them unless asked to keep.
    if not keep_aligned:

        def remove_aligned():
            shutil.rmtree(aligned_dir, ignore_errors=True)

        steps.append(Step("Removing aligned frames", on_done=remove_aligned))

    return steps


def build_plan(tools: dict, frames: list[Path], outdir: Path, args) -> dict[str, list[Step]]:
    """Group the steps by alignment engine ('all' runs both, each in its own folder)."""
    plan: dict[str, list[Step]] = {}
    modes = ("manual", "ais") if args.aligner == "all" else (args.aligner,)
    for mode in modes:
        mode_dir = outdir / mode if args.aligner == "all" else outdir
        aligned_dir = mode_dir / "aligned"
        # The output folders must exist before any tool writes into them.
        mode_dir.mkdir(parents=True, exist_ok=True)
        aligned_dir.mkdir(parents=True, exist_ok=True)
        pattern = "ais_*.tif" if mode == "ais" else "rendered_*.tif"

        existing = sorted(aligned_dir.glob(pattern))
        if not args.force and len(existing) == len(frames):
            steps = [
                Step(
                    "Skipping alignment",
                    note=f"Found {len(existing)} aligned frames - skipping alignment",
                )
            ]
        elif mode == "ais":
            steps = ais_steps(tools, frames, aligned_dir)
        else:
            steps = manual_steps(tools, frames, aligned_dir, args.motion, args.photometric)

        steps += output_steps(
            tools,
            mode_dir,
            aligned_dir,
            frames[-1],
            args.merge_types,
            args.keep_merged,
            args.keep_aligned,
            args.motion_options,
            args.do_animation,
        )
        plan[mode] = steps
    return plan


# ---------------------------------------------------------------------------
#  Executing the plan
# ---------------------------------------------------------------------------
# A leading icon for each primary task's line (first prefix match wins).
STEP_ICONS: tuple[tuple[str, str], ...] = (
    ("Reading frames", "📖"),
    ("Finding matching features", "🔍"),
    ("Removing bad matches", "🧹"),
    ("Choosing alignment parameters", "🧭"),
    ("Optimising alignment", "🎯"),
    ("Refining matches", "🎯"),
    ("Matching exposure", "🌗"),
    ("Preparing output", "🧭"),
    ("Rendering aligned frames", "🖼️"),
    ("Aligning frames", "🎯"),
    ("Skipping alignment", "⏩"),
    ("Combining frames", "🧩"),
    ("Detecting motion", "🏃"),
    ("Sharpening", "✨"),
    ("Building animation", "🎞️"),
    ("ImageMagick not found", "⚠️"),
    ("Copying metadata", "🏷️"),
    ("Removing aligned frames", "🧹"),
    ("Merged TIFF", "📦"),
)


def step_icon(label: str) -> str:
    for prefix, icon in STEP_ICONS:
        if label.startswith(prefix):
            return icon
    return "•"


def _failure_tail(lines: list[str], keep: int = 6) -> str:
    """The last few lines a tool printed, for a failure message.

    A bare exit code says nothing about why a command failed, and these tools
    put the reason on their last lines - 'No active images in ROI' instead of a
    silent stop, for instance.
    """
    tail = [line.strip() for line in lines if line.strip()][-keep:]
    if not tail:
        return ""
    return "\n" + "\n".join(f"    {line}" for line in tail)


def execute_step(ui: UI, step: Step, console=None) -> tuple[float, str | None]:
    """Run one step's command or Python callable.

    Returns (elapsed seconds, optional report line).  Raises on failure.
    """
    start = time.time()
    if step.run_fn is not None:  # in-process step: no subprocess to supervise
        report = step.run_fn()
        return time.time() - start, report

    argv = step.cmd_fn() if step.cmd_fn is not None else step.cmd
    if ui.verbose:
        ui.cmd(argv, console)

    buf: list[str] = []
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip()
            buf.append(line)
            if ui.verbose:
                ui.dim(line, console)
        if proc.wait() != 0:
            raise RuntimeError(f"{step.label} failed (exit {proc.returncode}){_failure_tail(buf)}")
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        raise
    finally:
        if proc.stdout:
            proc.stdout.close()
    return time.time() - start, None


def run_plan(ui: UI, plan: dict[str, list[Step]]) -> int:
    """Execute every step. Returns the number of engines that failed.

    The progress bar is transient: it shows live progress while steps run and
    disappears when done, leaving the plain line-by-line transcript behind.
    """
    total = sum(len(steps) for steps in plan.values())
    progress = None
    if ui.console is not None:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=ui.console,
            transient=True,
        )

    failures = 0
    if progress is not None:
        with progress:
            task = progress.add_task("Starting", total=total)
            for mode, steps in plan.items():
                err = None
                done = 0
                for step in steps:
                    progress.update(task, description=step.label)
                    if (
                        step.cmd is None and step.cmd_fn is None and step.run_fn is None
                    ):  # informational / action-only
                        if step.on_done:
                            step.on_done()
                        ui.say(
                            f"  {step_icon(step.label)}"
                            f" [grey50]{escape(step.note or step.label)}[/grey50]",
                            progress.console,
                        )
                        progress.advance(task)
                        continue
                    try:
                        took, report = execute_step(ui, step, progress.console)
                    except Exception as exc:  # noqa: BLE001 - report and continue
                        err = str(exc)
                        break
                    ui.say(
                        f"  {step_icon(step.label)} {escape(step.label)}"
                        f" [grey50]({took:.1f}s)[/grey50]",
                        progress.console,
                    )
                    if report:
                        ui.dim(report, progress.console)
                    if step.on_done:
                        step.on_done()
                    progress.advance(task)
                    done += 1
                if err is not None:
                    failures += 1
                    progress.advance(task, len(steps) - done - 1)
                    ui.say(
                        f"  ❌ {escape(err)} - skipping the rest of {mode}",
                        progress.console,
                    )
    else:
        for mode, steps in plan.items():
            err = None
            for step in steps:
                if (
                    step.cmd is None and step.cmd_fn is None and step.run_fn is None
                ):  # informational / action-only
                    if step.on_done:
                        step.on_done()
                    ui.print(
                        f"  {step_icon(step.label)}"
                        f" [grey50]{escape(step.note or step.label)}[/grey50]"
                    )
                    continue
                try:
                    took, report = execute_step(ui, step)
                except Exception as exc:  # noqa: BLE001 - report and continue
                    err = str(exc)
                    break
                ui.print(
                    f"  {step_icon(step.label)} {escape(step.label)} [grey50]({took:.1f}s)[/grey50]"
                )
                if report:
                    ui.dim(report)
                if step.on_done:
                    step.on_done()
            if err is not None:
                failures += 1
                ui.print(f"  ❌ {escape(err)} - skipping the rest of {mode}")
    return failures


# ---------------------------------------------------------------------------
#  Options
# ---------------------------------------------------------------------------
class Toggle(argparse.Action):
    """A boolean flag with explicit on/off option names, shown on one help line.

    Example: Toggle("--keep-aligned", "--delete-aligned") sets the value True for
    --keep-aligned and False for --delete-aligned.  For the simple --X/--no-X
    case argparse's stdlib BooleanOptionalAction does this out of the box; we
    need a custom action only because the "off" names here are --delete-*.
    """

    def __init__(self, option_strings, dest, off=(), default=None, help=None, **kwargs):
        self._off = set(off)
        super().__init__(option_strings, dest, nargs=0, default=default, help=help, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, option_string not in self._off)


class AllOutputs(argparse.Action):
    """--all switches on every output type at the point it appears.

    Options are applied in the order they are given, so "--all --no-median"
    produces everything except the median merge, while "--no-action --all" turns
    the action composite back on.
    """

    OUTPUTS = ("do_mean", "do_winsor", "do_median", "do_clean", "do_action", "do_animation")

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, default=False, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)
        for name in self.OUTPUTS:
            setattr(namespace, name, True)


def collect_outputs(mode_dir: Path) -> list[Path]:
    """The deliverables actually sitting in an engine's output folder."""
    found = []
    for p in sorted(mode_dir.iterdir()):
        if not p.is_file() or "_fuse" not in p.name:
            continue
        if p.suffix.lower() in {".jpg", ".tif", ".tiff"} or p.suffix.lower() == ".gif":
            found.append(p)
    return found


def fmt_duration(seconds: float) -> str:
    """Format seconds as 12s, 1m 23s or 2h 04m."""
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def display_dir(path: Path) -> str:
    """Render an output folder so it reads as a directory: absolute paths as-is,
    relative ones prefixed with '/' ('.' becomes '/')."""
    s = path.as_posix()
    if path.is_absolute():
        return s
    if s.startswith("./"):
        s = s[2:]
    return "/" if s == "." else f"/{s}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fuse - align and fuse a stack of frames to reduce noise.",
        formatter_class=_HELP_FORMATTER,
    )
    ap.add_argument(
        "--aligner",
        choices=["manual", "ais", "all"],
        default="ais",
        help="alignment engine (default: ais)",
    )
    ap.add_argument(
        "--motion",
        choices=["high", "low"],
        default="high",
        help="manual engine: 'low' for a steady tripod (default: high)",
    )
    ap.add_argument(
        "images",
        nargs="*",
        metavar="IMAGE",
        help="frames to fuse: image files, glob patterns, or a folder (default: this folder)",
    )
    ap.add_argument(
        "-i",
        "--input",
        dest="source",
        metavar="INPUT",
        default=None,
        help="frames to fuse, as with IMAGE above (default: the current folder)",
    )
    ap.add_argument(
        "-o",
        "--outdir",
        default=None,
        help="output folder (default: current folder); 'all' uses sub-folders",
    )
    ap.add_argument(
        "--photometric",
        "--no-photometric",
        dest="photometric",
        action=Toggle,
        off=["--no-photometric"],
        default=False,
        help="manual engine: also match exposure/vignetting/WB between frames",
    )
    ap.add_argument(
        "--force", action="store_true", help="re-align even if aligned frames already exist"
    )
    ap.add_argument("--log", default=None, help="also write a transcript to this file")
    ap.add_argument("--no-color", action="store_true", help="disable coloured output")
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print the full command line and tool output of every step",
    )
    ap.add_argument(
        "--keep-merged-tifs",
        "--delete-merged-tifs",
        dest="keep_merged",
        action=Toggle,
        off=["--delete-merged-tifs"],
        default=False,
        help="keep the merged <name>_fuse_*.tif files (default: delete)",
    )
    ap.add_argument(
        "--keep-aligned",
        "--delete-aligned",
        dest="keep_aligned",
        action=Toggle,
        off=["--delete-aligned"],
        default=False,
        help="keep the aligned frames in the 'aligned' subfolder (default: delete)",
    )
    ap.add_argument(
        "--all",
        dest="all_outputs",
        action=AllOutputs,
        help="produce every output type: mean, winsor, median, clean, action, animation",
    )
    ap.add_argument(
        "--mean",
        "--no-mean",
        dest="do_mean",
        action=Toggle,
        off=["--no-mean"],
        default=None,
        help="produce the mean merge (default: off)",
    )
    ap.add_argument(
        "--winsor",
        "--no-winsor",
        dest="do_winsor",
        action=Toggle,
        off=["--no-winsor"],
        default=None,
        help="produce the winsorised-mean merge (the default when nothing else is named)",
    )
    ap.add_argument(
        "--median",
        "--no-median",
        dest="do_median",
        action=Toggle,
        off=["--no-median"],
        default=None,
        help="produce the median merge (default: off)",
    )
    ap.add_argument(
        "--clean",
        "--no-clean",
        dest="do_clean",
        action=Toggle,
        off=["--no-clean"],
        default=None,
        help="fuse with whatever moved between frames excluded (default: off)",
    )
    ap.add_argument(
        "--action",
        "--no-action",
        dest="do_action",
        action=Toggle,
        off=["--no-action"],
        default=None,
        help="fuse every detected moving instance back in, one per frame (default: off)",
    )
    ap.add_argument(
        "--animation",
        "--gif",
        "--no-animation",
        "--no-gif",
        dest="do_animation",
        action=Toggle,
        off=["--no-animation", "--no-gif"],
        default=None,
        help="build the animated GIF of the aligned frames (default: off)",
    )
    ap.add_argument(
        "--masks",
        "--no-masks",
        dest="do_masks",
        action=Toggle,
        off=["--no-masks"],
        default=False,
        help="keep the motion masks and a check sheet in 'masks' (default: off; -d implies on)",
    )
    ap.add_argument(
        "--mask-z",
        type=float,
        metavar="Z",
        default=3.5,
        help="motion: how far from the stack median a pixel must sit to count (default: 3.5)",
    )
    ap.add_argument(
        "--mask-min-area",
        type=int,
        metavar="PX",
        default=0,
        help="motion: ignore blobs smaller than this after growing them, in pixels"
        " (default: 0 = auto, 0.02%% of the frame)",
    )
    ap.add_argument(
        "--mask-dilate",
        type=int,
        metavar="PX",
        default=12,
        help="motion: grow each mask by this radius, in pixels (default: 12)",
    )
    ap.add_argument(
        "--mask-feather",
        type=float,
        metavar="PX",
        default=4.0,
        help="motion: soften the pasted edge by this sigma, in pixels (default: 4)",
    )
    ap.add_argument(
        "-d",
        "--develop",
        "--debug",
        action="store_true",
        help="keep everything (merged TIFFs + aligned frames) and enable verbose logging",
    )
    args = ap.parse_args()

    # Output selection: naming any output gives exactly the ones named, so
    # "--action" no longer drags the default winsor along.  A run that names no
    # output at all falls back to the winsorised mean, as it always did.
    outputs = ("do_mean", "do_winsor", "do_median", "do_clean", "do_action", "do_animation")
    if all(getattr(args, name) is None for name in outputs):
        args.do_winsor = True
    for name in outputs:
        if getattr(args, name) is None:
            setattr(args, name, False)
    if not any(getattr(args, name) for name in outputs):
        raise SystemExit(
            "No outputs selected - every output type was switched off.\n"
            "Name the ones you want, or use --all for everything."
        )
    # -d/--develop implies: keep all intermediates + verbose output.
    if args.develop:
        args.keep_merged = True
        args.keep_aligned = True
        args.verbose = True
        args.do_masks = True
    args.motion_options = MotionOptions(
        clean=args.do_clean,
        action=args.do_action,
        masks=args.do_masks,
        z_thresh=args.mask_z,
        min_area=args.mask_min_area,
        dilate=args.mask_dilate,
        feather=args.mask_feather,
    )
    args.merge_types = [
        name
        for name, on in (
            ("mean", args.do_mean),
            ("winsor", args.do_winsor),
            ("median", args.do_median),
        )
        if on
    ]

    setup_console()
    ui = UI(verbose=args.verbose, no_color=args.no_color, log_path=args.log)
    run_start = time.time()
    try:
        tools = resolve_tools()
        outdir = Path(args.outdir) if args.outdir else Path(".")
        specs = [*args.images, *([args.source] if args.source else [])] or [str(Path.cwd())]

        frames, skipped, unmatched = collect_frames(specs)

        # ais aligns in place and cannot pad or rescale, so a stack of mixed
        # sizes would die deep inside align_image_stack - refuse it up front.
        if args.aligner in ("ais", "all"):
            check_same_size(frames)

        extra: list[str] = []
        if skipped:
            names = ", ".join(p.name for p in skipped[:5])
            more = f" (+{len(skipped) - 5} more)" if len(skipped) > 5 else ""
            extra.append(
                f"[yellow]Ignored {len(skipped)} previous Fuse output(s): {names}{more}[/yellow]"
            )
        if unmatched:
            names = ", ".join(unmatched[:5])
            more = f" (+{len(unmatched) - 5} more)" if len(unmatched) > 5 else ""
            extra.append(f"[yellow]No image matched: {names}{more}[/yellow]")

        # Warn but carry on when the frames are not all one size.  The manual
        # engine remaps mixed sizes happily - a cropped frame is a legitimate
        # frame - but a stray merge file in the pile fits the geometry to nothing
        # and the stitcher then stops with "No active images in ROI".
        sizes = {s for p in frames if (s := frame_size(p)) is not None}
        if args.aligner not in ("ais", "all") and len(sizes) > 1:
            extra.append(
                "[yellow]Frames have mixed pixel sizes: "
                + ", ".join(f"{w}x{h}" for w, h in sorted(sizes))
                + " (fine for a cropped frame, but check nothing stray like an old"
                " merge ended up in the pile)[/yellow]"
            )

        where = specs[0] if len(specs) == 1 else " ".join(specs)
        ui.header("Fuse", len(frames), where, frames, extra)

        plan = build_plan(tools, frames, outdir, args)
        failures = run_plan(ui, plan)

        results = [
            (mode, p)
            for mode in plan
            for p in collect_outputs(outdir / mode if args.aligner == "all" else outdir)
        ]
        took = fmt_duration(time.time() - run_start)
        ui.print("")
        ui.summary(results)
        if failures:
            ui.print(f"[red]❌ Finished with {failures} failure(s) in {took}.[/red]")
            return 1
        ui.print(f"🎉 [green]Done in {took}.[/green] Output in: {display_dir(outdir)}")
        return 0
    except KeyboardInterrupt:
        ui.print("")
        ui.print("⏹️ [red]Interrupted.[/red]")
        return 130
    finally:
        ui.close()


if __name__ == "__main__":
    sys.exit(main())
