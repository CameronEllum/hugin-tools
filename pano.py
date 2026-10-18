#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["rich", "rich-argparse"]
# ///
"""
Pano - classic Hugin panorama workflow, as a command-line tool.

Mirrors the manual Hugin GUI workflow:

     1. import all jpgs in the folder (pto_gen; middle frame listed first)
     2. find control points (cpfind)
     3. vertical-line control points (linefind, skipped with --no-lines)
     4. middle image as the anchor (it is first, so pair-wise optimisation
        starts from it - see autooptimiser -p)
     5. geometric optimisation of positions, starting from the anchor
     6. remove residuals (per-point pano-space distance above --threshold)
     7. optimise positions again
     8. center yaw, aim the view vertically (the "drag"), then autocrop
     9. calculate optimal canvas size
    10. stitch

A numbered .pto project is saved after every step, so any step can be opened
in the Hugin GUI and inspected.

Run it with uv (it pulls in `rich` automatically):

    uv run pano.py -i /photos/pano-row -o /photos/pano-row

Notes / known gaps (this is a start, not a drop-in for every pano):
  - the anchor is implemented by ordering: the middle frame is listed first in
    the project, so pair-wise optimisation (autooptimiser -p) starts from it.
  - step 6 measures each control point's residual in *panorama* pixels via
    pano_trafo and drops pairs above --threshold (default 3px). Only normal
    (t0) points are filtered; vertical-line points are kept.
  - geometry needs the field of view from the jpgs' EXIF data (pto_gen reads
    it). Without it Hugin assumes 50 degrees, which skews residuals.
  - stitch outputs are named <last frame>_<projection>.<ext> (JPEG by default;
    see --prefix/--format) and get that frame's metadata (plus the anchor
    frame's, with the last frame winning on conflicts).
  - default stitch output is produced with Hugin's default settings.
"""

import argparse
import ctypes
import locale
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
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
        anchor: Path | None = None,
    ) -> None:
        """Program name, inputs, frame names (natural sort order, anchor marked),
        then a Progress: heading."""
        noun = "frame" if count == 1 else "frames"
        self.print(f"📷 [bold]{title}[/bold]  ·  {count} {noun}")
        self.print("")
        self.print(f"📁 [grey50]{escape(str(source))}[/grey50]")
        for p in sorted(frames, key=lambda p: p.name):
            mark = " (anchor)" if anchor is not None and p == anchor else ""
            self.print(f"  📄 [grey50]{escape(p.name)}{mark}[/grey50]")
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
        "linefind",
        "autooptimiser",
        "pano_modify",
        "pano_trafo",
        "hugin_executor",
    ]
    tools = {name: find_tool(name) for name in required}
    tools["exiftool"] = find_tool("exiftool")  # optional; used only to copy metadata

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
IMAGE_EXTS = {".jpg", ".jpeg"}


def collect_frames(source: Path, out_prefix: str | None) -> tuple[list[Path], list[Path]]:
    """Top-level jpgs in a folder, sorted by name. Middle frame listed first.

    Listing the middle frame first is what makes it the anchor: pair-wise
    optimisation (autooptimiser -p) starts from the first image.

    Returns (frames, skipped) without printing, so the caller can show the
    program header first and the operation metadata afterwards.
    """
    if not source.is_dir():
        raise SystemExit(f"Source folder not found: {source}")

    frames, skipped = [], []
    for p in sorted(source.iterdir(), key=lambda p: p.name):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        # Skip this tool's own stitch outputs on re-runs in the same folder:
        # anything ending in a projection tag (_equirectangular/_rectilinear,
        # or legacy _pano), plus anything starting with an explicit --prefix.
        if re.search(r"_(pano|equirectangular|rectilinear)$", p.stem) or (
            out_prefix and p.stem.startswith(out_prefix)
        ):
            skipped.append(p)
            continue
        frames.append(p)

    if len(frames) < 2:
        raise SystemExit(f"Need at least 2 jpgs to stitch, found {len(frames)} in: {source}")

    # Rotate so the middle frame comes first -> it becomes the anchor.
    mid = len(frames) // 2
    frames = [frames[mid]] + frames[:mid] + frames[mid + 1 :]
    return frames, skipped


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
    on_done: Callable[[], str | None] | None = None  # may return a message to print
    outputs: list[Path] = field(default_factory=list)  # project files this step writes
    quiet: bool = False  # run silently: no transcript line unless it fails


# A leading icon for each primary task's line (first prefix match wins).
STEP_ICONS: tuple[tuple[str, str], ...] = (
    ("Reading frames", "📖"),
    ("Finding matching features", "🔍"),
    ("Finding vertical lines", "📏"),
    ("Setting anchor", "⚓"),
    ("Optimising positions", "🎯"),
    ("Removing points", "🧹"),
    ("Centering", "🧭"),
    ("Calculating output size", "📐"),
    ("Stitching", "🧵"),
    ("Skipping", "⏩"),
    ("ImageMagick not found", "⚠️"),
    ("Copying metadata", "🏷️"),
    ("Checking", "🔍"),
    ("Straightening", "↔️"),
)


def step_icon(label: str) -> str:
    for prefix, icon in STEP_ICONS:
        if label.startswith(prefix):
            return icon
    return "•"


def _pto_image_paths(pto_path: Path) -> list[str]:
    """Image file paths listed in a .pto project's image lines."""
    paths = []
    text = Path(pto_path).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if line.startswith("i "):
            m = re.search(r'\bn"([^"]+)"', line)
            if m:
                paths.append(m.group(1))
    return paths


def _normalize_project_text(pto_path: Path) -> int:
    """Rewrite a project file as clean UTF-8. Returns the number of lines fixed.

    Lines that are already valid UTF-8 pass through untouched. Lines that are
    not are decoded with the system ANSI codepage and re-encoded as UTF-8;
    lines undecodable either way are left as-is.
    """
    raw = Path(pto_path).read_bytes()
    fixed = 0
    out: list[bytes] = []
    for line in raw.split(b"\n"):
        try:
            line.decode("utf-8")
            out.append(line)
        except UnicodeDecodeError:
            try:
                out.append(line.decode(locale.getpreferredencoding(False)).encode("utf-8"))
                fixed += 1
            except (UnicodeDecodeError, LookupError):
                out.append(line)
    if fixed:
        Path(pto_path).write_bytes(b"\n".join(out))
    return fixed


def _relativize_image_paths(pto_path: Path, base_dir: Path) -> int:
    """Rewrite problem image paths as bare basenames. Returns paths changed.

    Hugin's stitch path cannot open absolute non-ASCII paths on Windows, but
    bare ASCII names resolve fine (the tool runs from inside the source
    folder). Anything already working is left untouched.
    """
    lines = Path(pto_path).read_text(encoding="utf-8", errors="replace").split("\n")
    changed = 0
    out = []
    for ln in lines:
        if ln.startswith("i "):
            m = re.search(r'\bn"([^"]+)"', ln)
            if m:
                stored = m.group(1)
                bare = Path(stored).name
                p = Path(stored)
                resolvable = (
                    p.exists()
                    or (pto_path.parent / stored).exists()
                    or (base_dir / stored).exists()
                )
                needs_fix = (not resolvable) or (p.is_absolute() and not stored.isascii())
                if needs_fix and bare.isascii() and (base_dir / bare).exists() and bare != stored:
                    ln = ln[: m.start(1)] + bare + ln[m.end(1) :]
                    changed += 1
        out.append(ln)
    if changed:
        Path(pto_path).write_text("\n".join(out), encoding="utf-8")
    return changed


def check_images_action(pto_path: Path, base_dir: Path):
    """Return an on_done check that fails fast if listed images don't resolve.

    Catches stale projects and non-ASCII path mangling (Hugin breaks umlauts
    etc. in project files on Windows) before minutes are spent on later steps.
    Bare filenames are resolved against the project folder first, then the
    source folder (Hugin itself resolves them against the working directory).
    """

    def resolves(p: str) -> bool:
        cand = Path(p)
        if cand.is_absolute():
            return cand.exists()
        return (pto_path.parent / p).exists() or (base_dir / p).exists() or cand.exists()

    def check() -> str | None:
        fixed = _normalize_project_text(pto_path)
        relocated = _relativize_image_paths(pto_path, base_dir)
        missing = [p for p in _pto_image_paths(pto_path) if not resolves(p)]
        if not missing:
            notes = []
            if fixed:
                notes.append(f"Normalised {fixed} line(s) to UTF-8")
            if relocated:
                notes.append(f"Made {relocated} image path(s) relative")
            return "; ".join(notes) if notes else None
        preview = ", ".join(missing[:3])
        raise RuntimeError(
            f"{len(missing)} image(s) listed in {pto_path.name} cannot be opened"
            f" ({preview}). The project is stale or the folder contains"
            " non-ASCII characters (which break Hugin projects on Windows) -"
            " re-run with --force from an ASCII-only path."
        )

    return check


_C_TOKEN = re.compile(r"^(-?[A-Za-z])(.+)$")


def _parse_c_line(text: str) -> dict | None:
    """Parse one 'c' control-point line. Returns None for lines we keep as-is.

    Only normal (t0 / no-t) two-image points are returned for residual checks;
    vertical lines and other modes are passed through untouched.
    """
    toks = text.split()
    if not toks or toks[0] != "c" or len(toks) < 7:
        return None
    d: dict = {"t": 0}
    try:
        for tok in toks[1:]:
            m = _C_TOKEN.match(tok)
            if not m:
                continue
            key, val = m.group(1), m.group(2)
            if key == "n":
                d["n"] = int(val)
            elif key == "N":
                d["N"] = int(val)
            elif key == "x":
                d["x"] = float(val)
            elif key == "y":
                d["y"] = float(val)
            elif key == "X":
                d["X"] = float(val)
            elif key == "Y":
                d["Y"] = float(val)
            elif key == "t":
                d["t"] = int(float(val))
        if not all(k in d for k in ("n", "N", "x", "y", "X", "Y")):
            return None
    except ValueError:
        return None
    return d if d.get("t", 0) == 0 else None


_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

_YP_RE = re.compile(rb"(?:^|\s)([yp])(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?=\s|$)")


def mean_pitch(pto: Path) -> float:
    """Mean pitch (degrees) of the images in a Hugin project file."""
    pitches: list[float] = []
    for line in Path(pto).read_bytes().split(b"\n"):
        if not line.startswith(b"i "):
            continue
        for match in _YP_RE.finditer(line.split(b' n"', 1)[0]):
            if match.group(1) == b"p":
                pitches.append(float(match.group(2)))
    return sum(pitches) / len(pitches) if pitches else 0.0


def filter_residuals(tools: dict, src_pto: Path, dst_pto: Path, threshold: float) -> str:
    """Drop control points whose pano-space distance is above threshold.

    Both points of each normal pair are mapped into panorama coordinates with a
    single batched pano_trafo call; pairs further apart than `threshold` pixels
    are removed from the project. Returns a one-line report.
    """
    lines = Path(src_pto).read_text(encoding="utf-8", errors="replace").split("\n")
    indexed = [
        (i, d) for i, ln in enumerate(lines) if ln.startswith("c ") and (d := _parse_c_line(ln))
    ]

    if not indexed:
        # Nothing checkable (e.g. only line points): pass the file through.
        shutil.copyfile(src_pto, dst_pto)
        return "No normal control points to check - kept everything"

    payload = "".join(
        f"{d['n']} {d['x']} {d['y']}\n{d['N']} {d['X']} {d['Y']}\n" for _, d in indexed
    )
    proc = subprocess.run(
        [tools["pano_trafo"], str(src_pto)],
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        detail = tail[-1] if tail else f"exit {proc.returncode}"
        raise RuntimeError(f"Residual check failed (pano_trafo): {detail}")

    pts: list[tuple[float, float]] = []
    for out_line in proc.stdout.splitlines():
        floats = _FLOAT_RE.findall(out_line)
        if len(floats) < 2:
            raise RuntimeError(f"pano_trafo returned an unparseable line: {out_line.strip()!r}")
        pts.append((float(floats[0]), float(floats[1])))
    if len(pts) != 2 * len(indexed):
        raise RuntimeError(f"pano_trafo returned {len(pts)} points for {2 * len(indexed)} queries")

    drop: set[int] = set()
    dists: list[float] = []
    for k, (line_no, _) in enumerate(indexed):
        dist = math.hypot(pts[2 * k][0] - pts[2 * k + 1][0], pts[2 * k][1] - pts[2 * k + 1][1])
        dists.append(dist)
        if dist > threshold:
            drop.add(line_no)

    kept = [ln for i, ln in enumerate(lines) if i not in drop]
    Path(dst_pto).write_text("\n".join(kept), encoding="utf-8")
    kept_dists = [d for k, d in enumerate(dists) if indexed[k][0] not in drop]
    removed_dists = [d for k, d in enumerate(dists) if indexed[k][0] in drop]
    parts = [f"Removed {len(drop)} of {len(indexed)} points above {threshold:g}px."]
    if removed_dists:
        parts.append(f"Worst removed: {max(removed_dists):.1f}px.")
    if kept_dists:
        parts.append(f"Worst kept: {max(kept_dists):.1f}px.")
    else:
        parts.append("Nothing kept.")
    report = " ".join(parts)
    if indexed and len(drop) == len(indexed):
        # Everything failed the threshold - the geometry itself is suspect (e.g.
        # missing EXIF field of view). Say so loudly instead of proceeding blind.
        report += " - WARNING: nothing left to optimise with"
    return report


def measure_residuals(tools: dict, src_pto: Path) -> list[float]:
    """Pano-space residual (px) for every normal control-point pair in a project."""
    lines = Path(src_pto).read_text(encoding="utf-8", errors="replace").split("\n")
    indexed = [
        (i, d) for i, ln in enumerate(lines) if ln.startswith("c ") and (d := _parse_c_line(ln))
    ]
    if not indexed:
        return []
    payload = "".join(
        f"{d['n']} {d['x']} {d['y']}\n{d['N']} {d['X']} {d['Y']}\n" for _, d in indexed
    )
    proc = subprocess.run(
        [tools["pano_trafo"], str(src_pto)],
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        detail = tail[-1] if tail else f"exit {proc.returncode}"
        raise RuntimeError(f"Residual check failed (pano_trafo): {detail}")
    pts: list[tuple[float, float]] = []
    for out_line in proc.stdout.splitlines():
        floats = _FLOAT_RE.findall(out_line)
        if len(floats) < 2:
            raise RuntimeError(f"pano_trafo returned an unparseable line: {out_line.strip()!r}")
        pts.append((float(floats[0]), float(floats[1])))
    if len(pts) != 2 * len(indexed):
        raise RuntimeError(f"pano_trafo returned {len(pts)} points for {2 * len(indexed)} queries")
    return [
        math.hypot(pts[2 * k][0] - pts[2 * k + 1][0], pts[2 * k][1] - pts[2 * k + 1][1])
        for k in range(len(indexed))
    ]


def residual_stats(tools: dict, src_pto: Path) -> str:
    """One-line max/RMS residual report for a project."""
    dists = measure_residuals(tools, src_pto)
    if not dists:
        return "No normal control points to measure"
    rms = math.sqrt(sum(d * d for d in dists) / len(dists))
    return f"Max residual: {max(dists):.1f}px, RMS residual: {rms:.1f}px."


def build_plan(
    tools: dict, frames: list[Path], outdir: Path, args, ui: UI
) -> dict[str, list[Step]]:
    """One engine ('pano'), numbered projects written after every step."""
    outdir.mkdir(parents=True, exist_ok=True)
    plan: dict[str, list[Step]] = {}
    prefix = args.prefix
    # Hugin's stitch path cannot open absolute non-ASCII paths on Windows: hand
    # it a CWD-relative prefix (a bare name when working in the output folder).
    stitch_prefix = os.path.relpath(outdir / prefix, Path.cwd())
    # Output projection for the stitch (Hugin numbers: 0 = rectilinear, 2 = equirectangular).
    proj = {"equirectangular": 2, "e": 2, "rectilinear": 0, "r": 0}[args.projection]
    # LDR output type for the stitch; JPG gets an explicit quality to match Fuse.
    fmt = args.format.upper()
    ldr = ["--ldr-file=" + fmt] + (["--ldr-compression=95"] if fmt == "JPG" else [])
    anchor_frame, final_frame = frames[0], frames[-1]

    def checkpoint(label: str, path: Path, build) -> Step:
        if not args.force and path.exists():
            return Step(label, note=f"Exists: {path.name} - skipping")
        step = build(path)
        step.outputs.append(path)
        return step

    steps: list[Step] = []

    steps.append(
        checkpoint(
            "Reading frames",
            outdir / "01_import.pto",
            lambda dst: Step(
                "Reading frames", [tools["pto_gen"], "-o", str(dst), *[str(p) for p in frames]]
            ),
        )
    )
    steps.append(
        Step(
            "Checking image paths",
            on_done=check_images_action(outdir / "01_import.pto", frames[0].parent),
            quiet=True,
        )
    )

    steps.append(
        checkpoint(
            "Finding matching features",
            outdir / "02_features.pto",
            lambda dst: Step(
                "Finding matching features",
                [
                    tools["cpfind"],
                    f"--output={dst}",
                    "--sieve2width=7",
                    "--sieve2height=7",
                    "--sieve2size=3",
                    "--fullscale",
                    "--celeste",
                    "--ncores=3",
                    str(outdir / "01_import.pto"),
                ],
            ),
        )
    )
    if args.do_lines:
        steps.append(
            checkpoint(
                "Finding vertical lines",
                outdir / "03_lines.pto",
                lambda dst: Step(
                    "Finding vertical lines",
                    [
                        tools["linefind"],
                        "-o",
                        str(dst),
                        f"--lines={args.line_count}",
                        str(outdir / "02_features.pto"),
                    ],
                ),
            )
        )

    # Without lines the anchor is copied straight from the cpfind project.
    lines_src = outdir / "03_lines.pto" if args.do_lines else outdir / "02_features.pto"

    def anchor_action() -> str | None:
        shutil.copyfile(lines_src, outdir / "04_anchor.pto")
        return f"Middle frame {frames[0].name} is listed first - autooptimiser starts from it"

    steps.append(
        Step(
            "Setting anchor",
            on_done=anchor_action,
            outputs=[outdir / "04_anchor.pto"],
        )
        if args.force or not (outdir / "04_anchor.pto").exists()
        else Step("Setting anchor", note="Exists: 04_anchor.pto - skipping")
    )

    steps.append(
        checkpoint(
            "Optimising positions",
            outdir / "05_geometry1.pto",
            lambda dst: Step(
                "Optimising positions",
                [tools["autooptimiser"], "-p", "-o", str(dst), str(outdir / "04_anchor.pto")],
            ),
        )
    )

    def clean_action() -> str | None:
        return filter_residuals(
            tools, outdir / "05_geometry1.pto", outdir / "06_clean.pto", args.threshold
        )

    steps.append(
        Step(
            "Removing points with error above threshold",
            on_done=clean_action,
            outputs=[outdir / "06_clean.pto"],
        )
        if args.force or not (outdir / "06_clean.pto").exists()
        else Step(
            "Removing points with error above threshold", note="Exists: 06_clean.pto - skipping"
        )
    )

    steps.append(
        checkpoint(
            "Optimising positions again",
            outdir / "07_geometry2.pto",
            lambda dst: Step(
                "Optimising positions again",
                [tools["autooptimiser"], "-p", "-o", str(dst), str(outdir / "06_clean.pto")],
                on_done=lambda: residual_stats(tools, outdir / "07_geometry2.pto"),
            ),
        )
    )
    # Yaw centering is unconditional: pano_modify --center is CentreHorizontally
    # (RotatePanorama(-dYaw, 0, 0)), which centres the horizontal extent on
    # yaw=0 and leaves pitch alone. Projection is set here so the following
    # fit-field-of-view uses the final projection.
    need08 = args.force or not (outdir / "08_preview.pto").exists()
    straighten_tmp = outdir / "08_straight_tmp.pto"
    yaw_tmp = outdir / "08a_yaw.pto"
    if args.straighten and need08:
        # Straighten first: centring/crop must run on the already-levelled
        # geometry, which a single combined call does not guarantee.
        steps.append(
            Step(
                "Straightening",
                [
                    tools["pano_modify"],
                    "-o",
                    str(straighten_tmp),
                    "--straighten",
                    str(outdir / "07_geometry2.pto"),
                ],
            )
        )

    def cleanup_tmp() -> str | None:
        straighten_tmp.unlink(missing_ok=True)
        return None

    yaw_src = straighten_tmp if (args.straighten and need08) else outdir / "07_geometry2.pto"
    steps.append(
        Step(
            "Centering (yaw)",
            [
                tools["pano_modify"],
                "-o",
                str(yaw_tmp),
                "--center",
                f"--projection={proj}",
                str(yaw_src),
            ],
            on_done=cleanup_tmp if (args.straighten and need08) else None,
            outputs=[yaw_tmp],
        )
        if need08
        else Step("Centering (yaw)", note="Exists: 08_preview.pto - skipping")
    )

    # Final aim: the scripted equivalent of dragging the view up/down in Hugin's
    # GL preview. pano_modify --center only centres yaw, so the images' mean
    # pitch is removed here with an explicit rotation (pano_modify applies
    # --rotate before --crop). Crop and projection always run.
    def center_step(dst: Path) -> Step:
        def cmd() -> list[str]:
            argv = [tools["pano_modify"], "-o", str(dst)]
            if args.center:
                pm = mean_pitch(yaw_tmp)
                if abs(pm) > 0.005:
                    argv += ["--rotate", f"0,{-pm:.5f},0"]
            argv += ["--crop=AUTO", f"--projection={proj}", str(yaw_tmp)]
            return argv

        return Step("Centering and cropping", cmd_fn=cmd)

    steps.append(checkpoint("Centering and cropping", outdir / "08_preview.pto", center_step))

    # Step 9 sets the canvas and the output file type.
    def size_step(dst: Path) -> Step:
        return Step(
            "Calculating output size",
            [
                tools["pano_modify"],
                "-o",
                str(dst),
                "--canvas=AUTO",
                *ldr,
                str(outdir / "08_preview.pto"),
            ],
            outputs=[dst],
        )

    steps.append(
        size_step(outdir / "09_sized.pto")
        if args.force or not (outdir / "09_sized.pto").exists()
        else Step("Calculating output size", note="Exists: 09_sized.pto - skipping")
    )

    steps.append(
        Step(
            "Checking image paths",
            on_done=check_images_action(outdir / "09_sized.pto", frames[0].parent),
            quiet=True,
        )
    )

    def stitch_outputs() -> list[Path]:
        return sorted(
            p
            for p in outdir.glob(f"{prefix}*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
        )

    def check_output_prefix() -> str | None:
        if stitch_prefix.isascii():
            return None
        raise RuntimeError(
            f"Output prefix '{stitch_prefix}' contains non-ASCII characters, which"
            " Hugin cannot write on Windows - stitch into an ASCII-only path."
        )

    if args.force or not stitch_outputs():
        steps.append(Step("Checking output path", on_done=check_output_prefix, quiet=True))

    steps.append(
        Step(
            "Stitching",
            [
                tools["hugin_executor"],
                "--stitching",
                f"--prefix={stitch_prefix}",
                str(outdir / "09_sized.pto"),
            ],
        )
        if args.force or not stitch_outputs()
        else Step("Stitching", note=f"Exists: {prefix}* output - skipping")
    )

    # Metadata comes from the anchor frame first and the final frame last, so
    # the final frame (which also gives the output its name) wins conflicts.
    exiftool = tools.get("exiftool")
    if exiftool:

        def tag_step():
            targets = [
                p
                for p in sorted(outdir.glob(f"{prefix}*"))
                if p.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff"}
            ]
            if not targets:
                raise RuntimeError("stitch produced no files to tag")
            argv = [exiftool, "-overwrite_original"]
            if os.name != "nt":
                argv += ["-charset", "filename=utf8"]
            argv += [
                "-TagsFromFile",
                str(anchor_frame),
                "-TagsFromFile",
                str(final_frame),
                "-all:all",
                "-Orientation#=1",
            ]
            argv += [str(p) for p in targets]
            return argv

        steps.append(Step(f"Copying metadata from {final_frame.name}", cmd_fn=tag_step))

    plan["pano"] = steps
    return plan


# ---------------------------------------------------------------------------
#  Executing the plan
# ---------------------------------------------------------------------------
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
                    if step.cmd is None and step.cmd_fn is None:
                        try:
                            msg = step.on_done() if step.on_done else None
                        except Exception as exc:  # noqa: BLE001 - report and continue
                            err = str(exc)
                            break
                        if not step.quiet:
                            tag = "".join(
                                f" [yellow]{escape(f'[{p.name}]')}[/]" for p in step.outputs
                            )
                            ui.say(
                                f"  {step_icon(step.label)}"
                                f" [grey50]{escape(msg or step.note or step.label)}{tag}[/grey50]",
                                progress.console,
                            )
                        progress.advance(task)
                        continue
                    try:
                        took = execute_step(ui, step, progress.console)
                    except Exception as exc:  # noqa: BLE001 - report and continue
                        err = str(exc)
                        break
                    tag = "".join(f" [yellow]{escape(f'[{p.name}]')}[/]" for p in step.outputs)
                    ui.say(
                        f"  {step_icon(step.label)} {escape(step.label)}"
                        f" [grey50]({took:.1f}s){tag}[/grey50]",
                        progress.console,
                    )
                    if step.on_done:
                        extra = step.on_done()
                        if extra:
                            ui.say(
                                f"  {step_icon(step.label)} [grey50]{escape(extra)}[/grey50]",
                                progress.console,
                            )
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
                if step.cmd is None and step.cmd_fn is None:
                    try:
                        msg = step.on_done() if step.on_done else None
                    except Exception as exc:  # noqa: BLE001 - report and continue
                        err = str(exc)
                        break
                    if not step.quiet:
                        tag = "".join(f" [yellow]{escape(f'[{p.name}]')}[/]" for p in step.outputs)
                        ui.print(
                            f"  {step_icon(step.label)}"
                            f" [grey50]{escape(msg or step.note or step.label)}{tag}[/grey50]"
                        )
                    continue
                try:
                    took = execute_step(ui, step)
                except Exception as exc:  # noqa: BLE001 - report and continue
                    err = str(exc)
                    break
                tag = "".join(f" [yellow]{escape(f'[{p.name}]')}[/]" for p in step.outputs)
                ui.print(
                    f"  {step_icon(step.label)} {escape(step.label)}"
                    f" [grey50]({took:.1f}s){tag}[/grey50]"
                )
                if step.on_done:
                    extra = step.on_done()
                    if extra:
                        ui.print(f"  {step_icon(step.label)} [grey50]{escape(extra)}[/grey50]")
            if err is not None:
                failures += 1
                ui.print(f"  ❌ {escape(err)} - skipping the rest of {mode}")
    return failures


# ---------------------------------------------------------------------------
#  Options
# ---------------------------------------------------------------------------
class Toggle(argparse.Action):
    """A boolean flag with explicit on/off option names, shown on one help line.

    Example: Toggle("--lines", "--no-lines") sets the value True for --lines
    and False for --no-lines.
    """

    def __init__(self, option_strings, dest, off=(), default=None, help=None, **kwargs):
        self._off = set(off)
        super().__init__(option_strings, dest, nargs=0, default=default, help=help, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, option_string not in self._off)


def collect_outputs(outdir: Path, prefix: str) -> list[Path]:
    """The stitched panorama files sitting in the output folder."""
    return sorted(
        p
        for p in outdir.glob(f"{prefix}*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
    )


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


class LinesToggle(argparse.Action):
    """--lines [N] / --no-lines on one help line: bare --lines uses linefind
    defaults, --lines N caps the kept lines at N, --no-lines skips detection."""

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs="?", const=True, default=True, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string == "--no-lines":
            namespace.do_lines = False
        elif values is True:
            namespace.do_lines = True
            namespace.line_count = 5
        else:
            try:
                count = int(values)
            except (TypeError, ValueError) as err:
                raise argparse.ArgumentError(self, f"expected an integer, got {values!r}") from err
            if count < 1:
                raise argparse.ArgumentError(self, "count must be >= 1")
            namespace.do_lines = True
            namespace.line_count = count


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pano - classic Hugin panorama workflow. "
        "A numbered .pto project is saved after every step.",
        formatter_class=_HELP_FORMATTER,
    )
    ap.add_argument(
        "-i",
        "--input",
        dest="source",
        metavar="INPUT",
        default=None,
        help="folder with the input jpgs (default: current folder)",
    )
    ap.add_argument(
        "-o",
        "--outdir",
        default=None,
        help="output folder (default: current folder)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="drop control points with a residual above this many pixels (default: 3)",
    )
    ap.add_argument(
        "-l",
        "--lines",
        "--no-lines",
        dest="do_lines",
        action=LinesToggle,
        metavar="N",
        help="detect vertical lines with linefind (default: on; --lines N keeps at most N)",
    )
    ap.set_defaults(line_count=5)
    ap.add_argument(
        "-s",
        "--straighten",
        "--no-straighten",
        dest="straighten",
        action=Toggle,
        off=["--no-straighten"],
        default=False,
        help="level the horizon when centering (default: off; pairs well with --lines)",
    )
    ap.add_argument(
        "-c",
        "--center",
        "--no-center",
        dest="center",
        action=Toggle,
        off=["--no-center"],
        default=True,
        help="aim the view vertically on the images - the scripted 'drag' in Hugin (default: on)",
    )
    ap.add_argument(
        "--prefix",
        default=None,
        help="file name prefix of the stitched panorama (default: <last frame>_<projection>)",
    )
    ap.add_argument(
        "--format",
        choices=["jpg", "tif", "png"],
        default="jpg",
        help="stitched file format (default: jpg)",
    )
    ap.add_argument(
        "-p",
        "--projection",
        choices=["equirectangular", "e", "rectilinear", "r"],
        default="equirectangular",
        help="output projection for the stitch (default: equirectangular)",
    )
    ap.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="redo every step, even if its output already exists",
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
        "-d",
        "--develop",
        "--debug",
        action="store_true",
        help="keep everything and enable verbose logging",
    )
    args = ap.parse_args()

    if args.develop:
        args.verbose = True

    setup_console()
    ui = UI(verbose=args.verbose, no_color=args.no_color, log_path=args.log)
    run_start = time.time()
    try:
        tools = resolve_tools()
        # Resolve everything up front, then run from inside the source folder
        # so bare image names in projects always resolve, wherever we were
        # invoked from.
        source = Path(args.source).resolve() if args.source else Path.cwd()
        outdir = Path(args.outdir).resolve() if args.outdir else Path.cwd()
        if args.log:
            args.log = str(Path(args.log).resolve())
        if not source.is_dir():
            raise SystemExit(f"Source folder not found: {source}")
        os.chdir(source)

        frames, skipped = collect_frames(source, args.prefix)

        extra: list[str] = []
        if skipped:
            names = ", ".join(p.name for p in skipped[:5])
            more = f" (+{len(skipped) - 5} more)" if len(skipped) > 5 else ""
            extra.append(
                f"[yellow]Ignored {len(skipped)} previous pano output(s): {names}{more}[/yellow]"
            )
        ui.header("Pano", len(frames), source, frames, extra, anchor=frames[0])

        # Default output name mirrors Fuse: <last frame>_<projection>.<ext>,
        # tagged with that frame's metadata (plus the anchor frame's, last winning).
        if not args.prefix:
            projname = {
                "equirectangular": "equirectangular",
                "e": "equirectangular",
                "rectilinear": "rectilinear",
                "r": "rectilinear",
            }[args.projection]
            args.prefix = f"{frames[-1].stem}_{projname}"

        plan = build_plan(tools, frames, outdir, args, ui)
        failures = run_plan(ui, plan)

        mode_dir = outdir
        results = [("pano", p) for p in collect_outputs(mode_dir, args.prefix)]
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
