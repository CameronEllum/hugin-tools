"""Pano - classic Hugin panorama workflow, as a command-line tool.

Mirrors the manual Hugin GUI workflow:

     1. import all jpgs in the folder (pto_gen; middle frame listed first)
     2. find control points (cpfind)
     3. vertical-line control points (linefind, skipped with --no-lines)
     4. middle image as the anchor
     5. geometric optimisation of positions, starting from the anchor
     6. remove residuals (per-point pano-space distance above --threshold)
     7. optimise positions again
     8. center yaw, aim the view vertically (the "drag"), then autocrop
     9. calculate optimal canvas size
    10. stitch

Numbered .pto projects are written after every step, under the scratch folder
(<outdir>/.hugin_tools/pano), so any step can be opened in the Hugin GUI and
inspected; they are deleted at the end unless kept with -k/--keep or --keep-ptos.

Notes / known gaps:
  - -l/--lines N keeps at most N of linefind's vertical lines (default 5, no
    filtering).  '-l filtered' (or '--line filtered') asks for a larger candidate
    pool and then keeps only the lines that agree with each other, the longest
    per frame.
  - the aligned (remapped) frames are written to the scratch folder's aligned/
    (one per input, on the panorama canvas, TIFF/LZW) when --aligned is given;
    they cost a second pass and roughly 25 MB per frame, so they are off by
    default (-d implies on).
  - --overlays (implied by -d) writes lines/<frame>_overlay.jpg with the detected
    lines drawn on the frame: red vertical, blue horizontal, and orange for lines
    the filter dropped.
  - the anchor is implemented by ordering: the middle frame is listed first.
  - step 6 measures residuals in panorama pixels via pano_trafo and drops pairs
    above --threshold (default 3px).
  - stitch outputs are named <last frame>_<projection>.<ext> and get that frame's
    metadata (plus the anchor frame's, with the last frame winning on conflicts).
"""

from __future__ import annotations

import argparse
import locale
import math
import os
import re
import subprocess
from pathlib import Path

from . import core
from .core import RunPlan, Step, copy_file, relpath, scratch_dir

# ---------------------------------------------------------------------------
#  Tool discovery
# ---------------------------------------------------------------------------
PANO_REQUIRED = [
    "pto_gen",
    "cpfind",
    "linefind",
    "autooptimiser",
    "pano_modify",
    "pano_trafo",
    "hugin_executor",
]


def resolve_tools() -> dict:
    """Find every external program Pano needs."""
    return core.resolve_tools(PANO_REQUIRED, ["exiftool"])


# ---------------------------------------------------------------------------
#  Source frames
# ---------------------------------------------------------------------------
PANO_EXTS = {".jpg", ".jpeg"}

# Output-projection tag used in the default stitch file name.
PROJ_NAME = {
    "equirectangular": "equirectangular",
    "e": "equirectangular",
    "rectilinear": "rectilinear",
    "r": "rectilinear",
}


def is_our_output(path: Path, out_prefix: str | None) -> bool:
    """This tool's own stitch outputs, which are never input frames.

    Anything ending in a projection tag (_equirectangular/_rectilinear, or
    legacy _pano), plus anything starting with an explicit --prefix.
    """
    if re.search(r"_(pano|equirectangular|rectilinear)$", path.stem):
        return True
    return bool(out_prefix and path.stem.startswith(out_prefix))


# ---------------------------------------------------------------------------
#  .pto readers / writers
# ---------------------------------------------------------------------------
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


def pto_lines(pto_path: Path) -> list[tuple[int, float, float, float, float, str]]:
    """Vertical/horizontal line control points: (image, x1, y1, x2, y2, kind).

    linefind writes each detected line as one 'c' line whose two points sit in
    the same image (n == N), tagged t1 (vertical) or t2 (horizontal).
    """
    found: list[tuple[int, float, float, float, float, str]] = []
    text = Path(pto_path).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if not line.startswith("c "):
            continue
        values: dict[str, str] = {}
        for token in line.split()[1:]:
            values[token[0]] = token[1:].lstrip("=")
        if values.get("t") not in ("1", "2"):
            continue
        try:
            found.append(
                (
                    int(values["n"]),
                    float(values["x"]),
                    float(values["y"]),
                    float(values["X"]),
                    float(values["Y"]),
                    "vertical" if values["t"] == "1" else "horizontal",
                )
            )
        except KeyError, ValueError:
            continue
    return found


def draw_line_overlays(
    pto_path: Path, kept_pto: Path | None, overlays_dir: Path, max_px: int = 1600
) -> str:
    """Write each frame with its detected lines drawn over it.

    The counterpart of Fuse's mask overlays: one small JPEG per frame, so a line
    that was detected - and, more to the point, one that was detected wrongly -
    can be seen where it sits.  Red = vertical, blue = horizontal, and orange for
    lines the filter dropped when filtering has run.
    """
    from PIL import Image, ImageDraw  # only the overlay path needs Pillow

    lines = pto_lines(pto_path)
    if not lines:
        return "no vertical or horizontal lines in the project - nothing to draw"
    kept_keys = {(n, x1, y1, x2, y2) for n, x1, y1, x2, y2, _ in pto_lines(kept_pto or pto_path)}

    sources: list[Path | None] = []
    for raw in _pto_image_paths(pto_path):
        for candidate in (Path(raw), Path.cwd() / raw, pto_path.parent / raw):
            if candidate.exists():
                sources.append(candidate)
                break
        else:
            sources.append(None)

    by_image: dict[int, list[tuple[float, float, float, float, str]]] = {}
    for image, x1, y1, x2, y2, kind in lines:
        by_image.setdefault(image, []).append((x1, y1, x2, y2, kind))

    overlays = Path(overlays_dir)
    overlays.mkdir(parents=True, exist_ok=True)
    drawn: list[Path] = []
    for index, segments in sorted(by_image.items()):
        if index >= len(sources) or sources[index] is None:
            continue
        source = sources[index]
        with Image.open(source) as opened:
            picture = opened.convert("RGB")
        draw = ImageDraw.Draw(picture)
        width = max(2, round(max(picture.size) / 700))
        for x1, y1, x2, y2, kind in segments:
            if kind == "vertical":
                forward = (round(x1), round(y1), round(x2), round(y2))
                kept = forward in {
                    (round(k[1]), round(k[2]), round(k[3]), round(k[4])) for k in kept_keys
                } or forward in {  # either endpoint order
                    (round(k[3]), round(k[4]), round(k[1]), round(k[2])) for k in kept_keys
                }
                colour = (255, 0, 0) if kept else (255, 140, 0)
            else:
                colour = (0, 128, 255)
            draw.line((x1, y1, x2, y2), fill=colour, width=width)
        picture.thumbnail((max_px, max_px))
        target = overlays / f"{source.stem}_overlay.jpg"
        picture.save(target, quality=88)
        drawn.append(target)

    if not drawn:
        return f"{len(lines)} line(s) found, but no frame they refer to could be opened"
    return (
        f"{len(lines)} line(s) on {len(drawn)} frame(s), red = vertical, blue = horizontal"
        + (", orange = dropped by the line filter" if kept_pto is not None else "")
        + f" -> {overlays.name}/"
    )


def _normalize_project_text(pto_path: Path) -> int:
    """Rewrite a project file as clean UTF-8. Returns the number of lines fixed."""
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
            except UnicodeDecodeError, LookupError:
                out.append(line)
    if fixed:
        Path(pto_path).write_bytes(b"\n".join(out))
    return fixed


def _relativize_image_paths(pto_path: Path, base_dir: Path) -> int:
    """Rewrite problem image paths as bare basenames. Returns paths changed.

    Hugin's stitch path cannot open absolute non-ASCII paths on Windows, but
    bare ASCII names resolve fine (the tool runs from inside the source folder).
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
    """Return an on_done check that fails fast if listed images don't resolve."""

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
    """Parse one 'c' control-point line. Returns None for lines we keep as-is."""
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
    """Drop control points whose pano-space distance is above threshold."""
    lines = Path(src_pto).read_text(encoding="utf-8", errors="replace").split("\n")
    indexed = [
        (i, d) for i, ln in enumerate(lines) if ln.startswith("c ") and (d := _parse_c_line(ln))
    ]

    if not indexed:
        copy_file(src_pto, dst_pto)
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


LINE_TOLERANCE = 1.5  # degrees a kept line may sit from the median line
LINE_COUNT_DEFAULT = 5  # what plain -l asks linefind for
LINE_COUNT_FILTERED = 40  # what -l filtered asks for, before filtering


def filter_line_constraints(
    tools: dict, src_pto: Path, dst_pto: Path, tolerance: float = LINE_TOLERANCE
) -> str:
    """Keep only the lines that agree with each other, the longest one per frame.

    linefind flags whatever looks vertical *in the picture*, which is not the same
    as vertical in the world once the camera is pitched.  The good lines broadly
    agree on a tilt and the bad ones stick out: measure every line's tilt in the
    fitted panorama, drop whatever sits more than `tolerance` degrees from the
    median, then keep just the longest survivor per frame.  A frame whose lines
    all fail keeps the one nearest the median rather than losing its constraint.
    """
    lines = Path(src_pto).read_text(encoding="utf-8", errors="replace").split("\n")
    entries = []
    for index, text in enumerate(lines):
        if not text.startswith("c "):
            continue
        values = {}
        for token in text.split()[1:]:
            values[token[0]] = token[1:].lstrip("=")
        if values.get("t") != "1":
            continue
        try:
            entries.append(
                (
                    index,
                    int(values["n"]),
                    float(values["x"]),
                    float(values["y"]),
                    float(values["X"]),
                    float(values["Y"]),
                )
            )
        except KeyError, ValueError:
            continue
    if not entries:
        copy_file(src_pto, dst_pto)
        return "No vertical lines to check - kept everything"

    payload = "".join(f"{n} {x} {y}\n{n} {X} {Y}\n" for _, n, x, y, X, Y in entries)
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
        raise RuntimeError(
            f"Line check failed (pano_trafo): {tail[-1] if tail else proc.returncode}"
        )
    points: list[tuple[float, float]] = []
    for out_line in proc.stdout.splitlines():
        floats = _FLOAT_RE.findall(out_line)
        if len(floats) >= 2:
            points.append((float(floats[0]), float(floats[1])))
    if len(points) != 2 * len(entries):
        raise RuntimeError(
            f"pano_trafo returned {len(points)} points for {2 * len(entries)} queries"
        )

    measured = []
    for i, entry in enumerate(entries):
        (ax, ay), (bx, by) = points[2 * i], points[2 * i + 1]
        tilt = math.degrees(math.atan2(bx - ax, by - ay))
        length = math.hypot(entry[5] - entry[3], entry[4] - entry[2])
        measured.append((tilt, length, entry))
    ordered = sorted(m[0] for m in measured)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2

    kept: dict[int, tuple[float, float, tuple]] = {}
    nearest: dict[int, tuple[float, float, tuple]] = {}
    dropped: list[tuple[float, float, tuple]] = []
    for tilt, length, entry in measured:
        image = entry[1]
        if abs(tilt - median) <= tolerance:
            if image not in kept or length > kept[image][1]:
                kept[image] = (tilt, length, entry)
        else:
            dropped.append((tilt, length, entry))
        if image not in nearest or abs(tilt - median) < abs(nearest[image][0] - median):
            nearest[image] = (tilt, length, entry)
    for image, best in nearest.items():
        kept.setdefault(image, best)  # never leave a frame without a constraint

    kept_indexes = {entry[0] for _, _, entry in kept.values()}
    drop_indexes = {entry[0] for _, _, entry in measured} - kept_indexes
    Path(dst_pto).write_text(
        "\n".join(ln for i, ln in enumerate(lines) if i not in drop_indexes), encoding="utf-8"
    )

    report = f"{len(entries)} line(s) checked, kept {len(kept)} (longest per frame)"
    if dropped:
        worst = max(dropped, key=lambda d: abs(d[0] - median))
        report += (
            f"; dropped {len(drop_indexes)}: image {worst[2][1]} sat {worst[0] - median:+.1f}"
            " deg off the median"
        )
    return report


def residual_stats(tools: dict, src_pto: Path) -> str:
    """One-line max/RMS residual report for a project."""
    dists = measure_residuals(tools, src_pto)
    if not dists:
        return "No normal control points to measure"
    rms = math.sqrt(sum(d * d for d in dists) / len(dists))
    return f"Max residual: {max(dists):.1f}px, RMS residual: {rms:.1f}px."


# ---------------------------------------------------------------------------
#  The plan
# ---------------------------------------------------------------------------
def build(tools: dict, frames: list[Path], outdir: Path, args) -> RunPlan:
    """One engine ('pano'), numbered projects written after every step."""
    outdir.mkdir(parents=True, exist_ok=True)
    if not args.prefix:
        # Default output name mirrors Fuse: <last frame>_<projection>.
        args.prefix = f"{frames[-1].stem}_{PROJ_NAME[args.projection]}"
    work = scratch_dir(outdir, "pano", args.scratch)
    pto = work / "pto"
    aligned_dir = work / "aligned"
    overlays_dir = work / "lines"
    pto.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    # Hugin's stitch path cannot open absolute non-ASCII paths on Windows: hand
    # it a CWD-relative prefix (a bare name when working in the output folder).
    # Across drives there is no relative form, so fall back to the absolute path.
    def safe_prefix(path: Path) -> str:
        try:
            return str(relpath(path, Path.cwd()))
        except ValueError:
            return str(path)

    stitch_prefix = safe_prefix(outdir / prefix)
    # Output projection for the stitch (Hugin numbers: 0 = rectilinear, 2 = equirectangular).
    proj = {"equirectangular": 2, "e": 2, "rectilinear": 0, "r": 0}[args.projection]
    # LDR output type for the stitch; JPG gets an explicit quality to match Fuse.
    fmt = args.format.upper()
    ldr = ["--ldr-file=" + fmt] + (["--ldr-compression=95"] if fmt == "JPG" else [])
    anchor_frame, final_frame = frames[0], frames[-1]

    def checkpoint(label: str, path: Path, build_step) -> Step:
        if not args.force and path.exists():
            return Step(label, note=f"Exists: {path.name} - skipping")
        step = build_step(path)
        step.outputs.append(path)
        return step

    steps: list[Step] = []

    steps.append(
        checkpoint(
            "Reading frames",
            pto / "01_import.pto",
            lambda dst: Step("Reading frames", [tools["pto_gen"], "-o", dst, *frames]),
        )
    )
    steps.append(
        Step(
            "Checking image paths",
            on_done=check_images_action(pto / "01_import.pto", frames[0].parent),
            quiet=True,
        )
    )

    steps.append(
        checkpoint(
            "Finding matching features",
            pto / "02_features.pto",
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
                    pto / "01_import.pto",
                ],
            ),
        )
    )
    if args.do_lines:
        steps.append(
            checkpoint(
                "Finding vertical lines",
                pto / "03_lines.pto",
                lambda dst: Step(
                    "Finding vertical lines",
                    [
                        tools["linefind"],
                        "-o",
                        dst,
                        f"--lines={args.line_count}",
                        pto / "02_features.pto",
                    ],
                ),
            )
        )

    # Without lines the anchor is copied straight from the cpfind project.
    lines_src = pto / "03_lines.pto" if args.do_lines else pto / "02_features.pto"

    def anchor_action() -> str | None:
        copy_file(lines_src, pto / "04_anchor.pto")
        return f"Middle frame {frames[0].name} is listed first - autooptimiser starts from it"

    steps.append(
        Step("Setting anchor", on_done=anchor_action, outputs=[pto / "04_anchor.pto"])
        if args.force or not (pto / "04_anchor.pto").exists()
        else Step("Setting anchor", note="Exists: 04_anchor.pto - skipping")
    )

    steps.append(
        checkpoint(
            "Optimising positions",
            pto / "05_geometry1.pto",
            lambda dst: Step(
                "Optimising positions",
                [tools["autooptimiser"], "-p", "-o", dst, pto / "04_anchor.pto"],
            ),
        )
    )

    # The line filter: linefind will accept a leaning edge, and a few bad
    # constraints pull the whole fit off vertical.  Needs a fitted project below.
    filtered_lines = pto / "05b_lines.pto"
    if args.line_filter and args.do_lines:

        def line_filter_action() -> str:
            return filter_line_constraints(tools, pto / "05_geometry1.pto", filtered_lines)

        steps.append(
            Step("Filtering line constraints", on_done=line_filter_action, outputs=[filtered_lines])
            if args.force or not filtered_lines.exists()
            else Step("Filtering line constraints", note="Exists: 05b_lines.pto - skipping")
        )

    geometry = filtered_lines if args.line_filter and args.do_lines else pto / "05_geometry1.pto"

    if args.overlays_on:
        if args.do_lines:
            kept_pto = filtered_lines if args.line_filter else None

            def overlays_action(kept_pto=kept_pto) -> str:
                return draw_line_overlays(pto / "03_lines.pto", kept_pto, overlays_dir)

            steps.append(Step("Drawing line overlays", on_done=overlays_action))
        else:
            steps.append(
                Step(
                    "Drawing line overlays",
                    note="--no-lines was given, so there are no lines to draw",
                )
            )

    def clean_action() -> str | None:
        return filter_residuals(tools, geometry, pto / "06_clean.pto", args.threshold)

    steps.append(
        Step(
            "Removing points with error above threshold",
            on_done=clean_action,
            outputs=[pto / "06_clean.pto"],
        )
        if args.force or not (pto / "06_clean.pto").exists()
        else Step(
            "Removing points with error above threshold", note="Exists: 06_clean.pto - skipping"
        )
    )

    steps.append(
        checkpoint(
            "Optimising positions again",
            pto / "07_geometry2.pto",
            lambda dst: Step(
                "Optimising positions again",
                [tools["autooptimiser"], "-p", "-o", dst, pto / "06_clean.pto"],
                on_done=lambda: residual_stats(tools, pto / "07_geometry2.pto"),
            ),
        )
    )
    need08 = args.force or not (pto / "08_preview.pto").exists()
    straighten_tmp = pto / "08_straight_tmp.pto"
    yaw_tmp = pto / "08a_yaw.pto"
    if args.straighten and need08:
        steps.append(
            Step(
                "Straightening",
                [
                    tools["pano_modify"],
                    "-o",
                    straighten_tmp,
                    "--straighten",
                    pto / "07_geometry2.pto",
                ],
            )
        )

    def cleanup_tmp() -> str | None:
        straighten_tmp.unlink(missing_ok=True)
        return None

    yaw_src = straighten_tmp if (args.straighten and need08) else pto / "07_geometry2.pto"
    steps.append(
        Step(
            "Centering (yaw)",
            [tools["pano_modify"], "-o", yaw_tmp, "--center", f"--projection={proj}", yaw_src],
            on_done=cleanup_tmp if (args.straighten and need08) else None,
            outputs=[yaw_tmp],
        )
        if need08
        else Step("Centering (yaw)", note="Exists: 08_preview.pto - skipping")
    )

    # Final aim: the scripted equivalent of dragging the view up/down in Hugin's
    # GL preview. pano_modify --center only centres yaw, so the images' mean
    # pitch is removed here with an explicit rotation.
    def center_step(dst: Path) -> Step:
        def cmd() -> list:
            argv = [tools["pano_modify"], "-o", dst]
            if args.center:
                pm = mean_pitch(yaw_tmp)
                if abs(pm) > 0.005:
                    argv += ["--rotate", f"0,{-pm:.5f},0"]
            argv += ["--crop=AUTO", f"--projection={proj}", yaw_tmp]
            return argv

        return Step("Centering and cropping", cmd_fn=cmd)

    steps.append(checkpoint("Centering and cropping", pto / "08_preview.pto", center_step))

    def size_step(dst: Path) -> Step:
        return Step(
            "Calculating output size",
            [tools["pano_modify"], "-o", dst, "--canvas=AUTO", *ldr, pto / "08_preview.pto"],
            outputs=[dst],
        )

    steps.append(
        size_step(pto / "09_sized.pto")
        if args.force or not (pto / "09_sized.pto").exists()
        else Step("Calculating output size", note="Exists: 09_sized.pto - skipping")
    )

    steps.append(
        Step(
            "Checking image paths",
            on_done=check_images_action(pto / "09_sized.pto", frames[0].parent),
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
                pto / "09_sized.pto",
            ],
        )
        if args.force or not stitch_outputs()
        else Step("Stitching", note=f"Exists: {prefix}* output - skipping")
    )

    # The aligned (remapped) frames, in a subfolder the way Fuse keeps its own.
    if args.aligned_on:
        aligned_dir.mkdir(parents=True, exist_ok=True)
        remap_pto = aligned_dir / "remap.pto"
        remap_prefix = safe_prefix(aligned_dir / "frame_")

        def remap_step(remap_pto=remap_pto):
            return [
                tools["pano_modify"],
                "-o",
                remap_pto,
                "--output-type=REMAP",
                "--output-uncropped-tiff",
                "--ldr-file=TIFF",
                "--ldr-compression=LZW",
                pto / "09_sized.pto",
            ]

        steps.append(Step("Preparing aligned frames", cmd_fn=remap_step, outputs=[remap_pto]))
        steps.append(
            Step(
                "Rendering aligned frames",
                [tools["hugin_executor"], "--stitching", f"--prefix={remap_prefix}", remap_pto],
            )
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
                anchor_frame,
                "-TagsFromFile",
                final_frame,
                "-all:all",
                "-Orientation#=1",
            ]
            argv += targets
            return argv

        steps.append(Step(f"Copying metadata from {final_frame.name}", cmd_fn=tag_step))

    return RunPlan(
        plan={"pano": steps},
        scratch={"pano": work},
        output_dir={"pano": outdir},
    )


def collect(runplan: RunPlan, prefix: str) -> list[tuple[str, Path]]:
    """The stitched panorama files sitting in the output folder."""
    outdir = next(iter(runplan.output_dir.values()))
    found = sorted(
        p
        for p in outdir.glob(f"{prefix}*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
    )
    return [("pano", p) for p in found]


# ---------------------------------------------------------------------------
#  Options
# ---------------------------------------------------------------------------
class LinesToggle(argparse.Action):
    """-l/--lines [N] / -l filtered / --no-lines on one help line.

    Bare -l uses linefind's own default count, -l N caps the kept lines at N,
    -l filtered (or --line filtered) takes a larger candidate pool and then
    filters it, and --no-lines skips detection.
    """

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs="?", const=True, default=True, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string == "--no-lines":
            namespace.do_lines = False
        elif values is True:
            namespace.do_lines = True
            namespace.line_count = LINE_COUNT_DEFAULT
        elif isinstance(values, str) and values.lower().startswith("filter"):
            namespace.do_lines = True
            namespace.line_filter = True
            namespace.line_count = LINE_COUNT_FILTERED
        else:
            try:
                count = int(values)
            except (TypeError, ValueError) as err:
                raise argparse.ArgumentError(
                    self, f"expected an integer or 'filtered', got {values!r}"
                ) from err
            if count < 1:
                raise argparse.ArgumentError(self, "count must be >= 1")
            namespace.do_lines = True
            namespace.line_count = count


def add_arguments(sub) -> None:
    """Register Pano's own options on its subcommand parser."""
    sub.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="drop control points with a residual above this many pixels (default: 3)",
    )
    sub.add_argument(
        "-l",
        "--lines",
        "--line",
        "--no-lines",
        dest="do_lines",
        action=LinesToggle,
        metavar="N",
        help="detect vertical lines with linefind (default: on, 5 of them);"
        " '-l N' keeps at most N, '-l filtered' takes a larger candidate pool and"
        " keeps only the lines that agree with each other, --no-lines skips them",
    )
    sub.set_defaults(line_count=LINE_COUNT_DEFAULT, line_filter=False)
    sub.add_argument(
        "-s",
        "--straighten",
        "--no-straighten",
        dest="straighten",
        action=core.Toggle,
        off=["--no-straighten"],
        default=False,
        help="level the horizon when centering (default: off; pairs well with --lines)",
    )
    sub.add_argument(
        "-c",
        "--center",
        "--no-center",
        dest="center",
        action=core.Toggle,
        off=["--no-center"],
        default=True,
        help="aim the view vertically on the images - the scripted 'drag' in Hugin (default: on)",
    )
    sub.add_argument(
        "--prefix",
        default=None,
        help="file name prefix of the stitched panorama (default: <last frame>_<projection>)",
    )
    sub.add_argument(
        "--format",
        choices=["jpg", "tif", "png"],
        default="jpg",
        help="stitched file format (default: jpg)",
    )
    sub.add_argument(
        "-p",
        "--projection",
        choices=["equirectangular", "e", "rectilinear", "r"],
        default="equirectangular",
        help="output projection for the stitch (default: equirectangular)",
    )
    sub.add_argument(
        "--aligned",
        "--no-aligned",
        dest="aligned",
        action=core.Toggle,
        off=["--no-aligned"],
        default=False,
        help="render the aligned (remapped) frames into the scratch folder"
        " (default: off; -d implies on)",
    )


def prepare_args(args) -> None:
    """Resolve Pano's derived flags."""
    args.aligned_on = bool(args.aligned or args.develop)


def scratch_paths(outdir: Path, args) -> list[Path]:
    return [scratch_dir(outdir, "pano", args.scratch)]


def keep(args, tools: dict) -> tuple[bool, set[str]]:
    keep_all = args.keep or args.develop
    names: set[str] = set()
    if args.keep_ptos:
        names.add("pto")
    if args.aligned_on or args.keep_aligned:
        names.add("aligned")
    if args.overlays_on:
        names.add("lines")
    return keep_all, names
