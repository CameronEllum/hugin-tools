#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["rich", "rich-argparse"]
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

    uv run fuse.py --aligner all -i /photos/stack/export -o /photos/stack/fuse

Layout of the output folder
---------------------------
    <outdir>/
        <name>_fuse_mean.jpg        sharpened JPEGs (the deliverables)
        <name>_fuse_winsor.jpg
        <name>_fuse_median.jpg
        <name>_fuse.gif
        <name>_fuse_*.tif           merged TIFFs  (deleted by default)
        aligned/                    aligned frames ("stack images"; deleted by default)

`<name>` is the last frame's file name, and its metadata is copied onto the outputs.
Everything except the JPEGs/GIF is deleted by default.  Use -d/--develop to keep
all intermediates and turn on verbose logging.

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

import argparse
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
            import ctypes

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
        source: Path,
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


def collect_frames(source: Path) -> tuple[list[Path], list[Path]]:
    """Top-level images in a folder, sorted by name (chronological).

    Returns (frames, skipped) without printing, so the caller can show the
    program header first and the operation metadata afterwards.
    """
    if not source.is_dir():
        raise SystemExit(f"Source folder not found: {source}")

    frames, skipped = [], []
    for p in source.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        # Skip our own outputs (…_fuse_*.jpg, and legacy …_MFNR_*.jpg) so a
        # re-run in the same folder never re-ingests them.
        if "_fuse" in p.stem or "_MFNR" in p.stem:
            skipped.append(p)
        else:
            frames.append(p)
    frames.sort(key=lambda p: p.name)

    if not frames:
        raise SystemExit(f"No source frames ({', '.join(sorted(IMAGE_EXTS))}) found in: {source}")
    return frames, skipped


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
#  The plan: an ordered list of steps, built up-front so the progress bar can
#  show real progress, then executed one at a time.
# ---------------------------------------------------------------------------
@dataclass
class Step:
    label: str
    cmd: list[str] | None = None
    cmd_fn: Callable[[], list[str]] | None = None  # built when the step runs
    note: str | None = None  # informational step: no command, just a message
    on_done: Callable[[], None] | None = None


# Merge types: name -> (extra hugin_stacker args, friendly label).
MERGE_TYPES = {
    "mean": (["--mode=sigma", "--max-sigma=2"], "sigma-clipped mean"),
    "winsor": (["--mode=winsor", "--winsor-trim=0.2"], "winsorised mean"),
    "median": (["--mode=median"], "median"),
}


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

    for name in merge_types:
        extra, label = MERGE_TYPES[name]
        tif = outdir / f"{base}_{name}.tif"

        def combine_step(tif=tif, extra=extra):
            return [stacker, *extra, f"--output={tif}", *[str(p) for p in aligned_frames()]]

        steps.append(Step(f"Combining frames ({label})", cmd_fn=combine_step))

        if magick:
            jpg = tif.with_suffix(".jpg")

            def sharpen_step(tif=tif, jpg=jpg):
                return [magick, str(tif), "-unsharp", "0x0.7+1.0+0", "-quality", "95", str(jpg)]

            # Keep the TIFF if asked, or when there is no ImageMagick to make a JPEG.
            def delete_tif(tif=tif):
                if tif.exists():
                    tif.unlink()

            steps.append(
                Step(
                    f"Sharpening {jpg.name}",
                    cmd_fn=sharpen_step,
                    on_done=None if keep_merged or not magick else delete_tif,
                )
            )
        elif keep_merged:
            steps.append(Step(f"Merged TIFF ready: {tif.name}", note=f"kept {tif.name}"))

    if magick:
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
    else:
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


def execute_step(ui: UI, step: Step, console=None) -> float:
    """Run one step's command. Returns elapsed seconds. Raises on failure."""
    argv = step.cmd_fn() if step.cmd_fn is not None else step.cmd
    if ui.verbose:
        ui.cmd(argv, console)

    start = time.time()
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
            raise RuntimeError(f"{step.label} failed (exit {proc.returncode})")
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        raise
    finally:
        if proc.stdout:
            proc.stdout.close()
    return time.time() - start


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
                    if step.cmd is None and step.cmd_fn is None:  # informational / action-only
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
                        took = execute_step(ui, step, progress.console)
                    except Exception as exc:  # noqa: BLE001 - report and continue
                        err = str(exc)
                        break
                    ui.say(
                        f"  {step_icon(step.label)} {escape(step.label)}"
                        f" [grey50]({took:.1f}s)[/grey50]",
                        progress.console,
                    )
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
                if step.cmd is None and step.cmd_fn is None:  # informational / action-only
                    if step.on_done:
                        step.on_done()
                    ui.print(
                        f"  {step_icon(step.label)}"
                        f" [grey50]{escape(step.note or step.label)}[/grey50]"
                    )
                    continue
                try:
                    took = execute_step(ui, step)
                except Exception as exc:  # noqa: BLE001 - report and continue
                    err = str(exc)
                    break
                ui.print(
                    f"  {step_icon(step.label)} {escape(step.label)} [grey50]({took:.1f}s)[/grey50]"
                )
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
        "-i",
        "--input",
        dest="source",
        metavar="INPUT",
        default=None,
        help="folder with the input frames (default: current folder)",
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
        "--mean",
        "--no-mean",
        dest="do_mean",
        action=Toggle,
        off=["--no-mean"],
        default=False,
        help="produce the mean merge (default: off)",
    )
    ap.add_argument(
        "--winsor",
        "--no-winsor",
        dest="do_winsor",
        action=Toggle,
        off=["--no-winsor"],
        default=True,
        help="produce the winsorised-mean merge (default: on)",
    )
    ap.add_argument(
        "--median",
        "--no-median",
        dest="do_median",
        action=Toggle,
        off=["--no-median"],
        default=False,
        help="produce the median merge (default: off)",
    )
    ap.add_argument(
        "-d",
        "--develop",
        "--debug",
        action="store_true",
        help="keep everything (merged TIFFs + aligned frames) and enable verbose logging",
    )
    args = ap.parse_args()

    # -d/--develop implies: keep all intermediates + verbose output.
    if args.develop:
        args.keep_merged = True
        args.keep_aligned = True
        args.verbose = True
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
        source = Path(args.source) if args.source else Path.cwd()
        outdir = Path(args.outdir) if args.outdir else Path(".")

        frames, skipped = collect_frames(source)

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

        ui.header("Fuse", len(frames), source, frames, extra)

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
