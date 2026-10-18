"""Pano - classic Hugin panorama workflow, as a command-line tool.

Mirrors the manual Hugin GUI workflow:

     1. import all jpgs in the folder (pto_gen; middle frame listed first)
     2. find control points (cpfind)
     3. find vertical lines (OpenCV LSD, validated against their vanishing point;
        skipped with --no-lines)
     4. middle image as the anchor
     5. clean the control points (cpclean)
     6. geometric optimisation of positions, starting from the anchor
     7. clean again, now including the line control points
     8. [--clean-distance] remove pairs by panorama-space distance
     9. optimise positions again
    10. center yaw and aim the view vertically
    11. autocrop, then calculate the optimal canvas size
    12. stitch

Numbered .pto projects are written after every step, under the scratch folder
(<outdir>/.hugin_tools/pano), so any step can be opened in the Hugin GUI and
inspected; they are deleted at the end unless kept with -k/--keep or --keep-ptos.

Notes:
  - linefind is deliberately not used.  It accepts whatever looks vertical *in
    the picture*, which is not vertical in the world once the camera is pitched
    or rolled.  Vertical lines in the world all pass through a single vanishing
    point, so LSD's candidate segments are checked against that point - the
    longest ones propose it and the rest are judged by how well they agree - and
    the ones that disagree are dropped.  cpclean then has a further pass over the
    whole project, lines included (--no-clean switches cleaning off).
  - the fit itself levels the panorama once the vertical lines are in it, because
    a t1 line pins the image's roll to the world's vertical.  pano_modify
    --straighten therefore only adds a correction on top, measured at about 0.5
    degrees of roll on the test panorama: it is off by default (-s turns it on).
  - --clean-sigma tunes that cleaning: a control point goes when its error is
    more than that many standard deviations above the mean (cpclean's own knob).
  - -l/--lines N caps how many line control points each frame contributes
    (default: no cap); --no-lines skips detection entirely.
  - --line-min-length is the shortest segment worth keeping, as a fraction of
    the frame's long side (default: 0.01, i.e. 1%).
  - the aligned (remapped) frames are written to the scratch folder's aligned/
    (one per input, on the panorama canvas, TIFF/LZW) when --aligned is given;
    they cost a second pass and roughly 25 MB per frame, so they are off by
    default (-d implies on).
  - --overlays (implied by -d) writes lines/<frame>_overlay.jpg with the detected
    lines drawn on the frame: red for lines the fit kept, orange for ones cpclean
    dropped.
  - the anchor is implemented by ordering: the middle frame is listed first.
  - --clean-distance (off by default) also drops control-point pairs whose
    panorama-space distance is above --threshold (default 3px); cpclean's
    statistical cut is the primary cleaner and this is a hard backstop.
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
    "cpclean",
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
    lines a later step dropped (the *kept* project, normally after cpclean).
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
        f"{len(lines)} line(s) on {len(drawn)} frame(s), red = kept, blue = horizontal"
        + (", orange = dropped by cpclean" if kept_pto is not None else "")
        + f" -> {overlays_dir.name}/"
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


# ---------------------------------------------------------------------------
#  Vertical lines: OpenCV LSD candidates, validated against their vanishing point
# ---------------------------------------------------------------------------
# linefind looks for whatever seems vertical *in the picture*, which is not
# vertical in the world once the camera is pitched or rolled: on a pitched camera
# honest verticals lean away from the centre of the frame, and a leaning edge can
# look perfectly upright.  What every world-vertical does have in common is its
# vanishing point, so that is what the candidates are checked against.  A coarse
# direction test only seeds the search; the vanishing point does the real work,
# and cpclean then has the final say along with everything else in the project.
LINE_MIN_FRACTION = 0.01  # shortest candidate segment, as a share of the long side
LINE_SEED_DEG = 60.0  # candidates must start within this of image-vertical
LINE_VP_GATE = 1.5  # ... and end within this of the agreed vanishing point
LINE_VP_IRLS = 12  # reweighting passes when refining the vanishing point
LINE_VP_SEED = 10  # longest candidates that get to propose vanishing points
LINE_VP_MIN_SUPPORT = 4  # lines a bundle needs before it is worth believing
LINE_LEAN_TOLERANCE = 0.3  # degrees a line may lean in the finished panorama


def _normalise_angle(angle: float) -> float:
    """Fold an angle from atan2 into (-90, 90]: a line has no direction."""
    while angle > 90.0:
        angle -= 180.0
    while angle <= -90.0:
        angle += 180.0
    return angle


def segment_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    """A segment's direction as an angle from image-vertical, in (-90, 90]."""
    return _normalise_angle(math.degrees(math.atan2(x2 - x1, y2 - y1)))


def load_gray(path: Path):
    """One frame as a grayscale uint8 array, stretched for the detector.

    Hugin's own tools read the frames; this is the only place we look at pixels.
    The stretch matters for night frames, where the useful edges live in a narrow
    slice of the range.
    """
    import numpy as np
    from PIL import Image

    with Image.open(path) as opened:
        gray = np.asarray(opened.convert("L"))
    if gray.size == 0:
        return gray
    low, high = np.percentile(gray, (2.0, 98.0))
    if high > low:
        scaled = (gray.astype(np.float32) - float(low)) * (255.0 / (float(high) - float(low)))
        gray = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
    return gray


def segments_from_array(gray, min_length: float = 0.0) -> list[tuple]:
    """Line segments in one grayscale frame: (x1, y1, x2, y2, length, score).

    OpenCV's LSD finds edges in every direction and knows nothing about verticals
    - that judgement belongs to `select_vertical_segments`.  The score is LSD's
    number-of-false-alarms figure (lower is better) where the build reports one,
    otherwise 0.
    """
    import cv2
    import numpy as np

    if gray is None or getattr(gray, "size", 0) == 0:
        return []
    if gray.dtype != np.uint8:
        gray = np.clip(np.asarray(gray), 0, 255).astype(np.uint8)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    found = detector.detect(gray)
    segments = found[0] if found else None
    if segments is None:
        return []
    nfa = found[3] if len(found) > 3 else None
    out: list[tuple] = []
    for index, segment in enumerate(segments):
        x1, y1, x2, y2 = (float(v) for v in np.asarray(segment).ravel()[:4])
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_length:
            continue
        score = float(nfa[index][0]) if nfa is not None else 0.0
        out.append((x1, y1, x2, y2, length, score))
    return out


def _line_normals(segments):
    """Plane normal per segment, as a unit 3-vector in the camera's own frame.

    The two rays through a segment's endpoints span a plane, and the world's
    vertical direction has to lie in that plane: n = r1 x r2, with the constraint
    d . n == 0.  Normalising the whole normal (not just its first two components,
    which is the segment's length) is what keeps a distant vanishing point from
    winning: the residual stays an angle instead of shrinking with the distance.
    """
    import numpy as np

    rows = [(y1 - y2, x2 - x1, x1 * y2 - x2 * y1) for x1, y1, x2, y2, _l, _s in segments]
    normals = np.asarray(rows, dtype=float)
    norm = np.linalg.norm(normals, axis=1)
    norm[norm == 0.0] = 1.0
    return normals / norm[:, None]


def vp_residuals(segments, v) -> list[float]:
    """Per-segment angular distance (degrees) from a vanishing point.

    Measured as the angle between the segment and the direction from its midpoint
    to the point, which stays meaningful however far away the point is (for a
    level camera it is at infinity).
    """
    import numpy as np

    v = np.asarray(v, dtype=float)
    if float(np.linalg.norm(v)) > 0:
        v = v / float(np.linalg.norm(v))
    finite = abs(v[2]) > 1e-9
    if finite:  # a finite vanishing point in the image plane
        point = np.array([v[0] / v[2], v[1] / v[2]])
    else:  # at infinity: the segments should all share one direction
        direction = np.array([v[0], v[1]])
        size = float(np.linalg.norm(direction))
        direction = direction / size if size else np.array([0.0, -1.0])

    residuals: list[float] = []
    for x1, y1, x2, y2, length, _score in segments:
        if length <= 0.0:
            residuals.append(90.0)
            continue
        segment = np.array([(x2 - x1) / length, (y2 - y1) / length])
        if finite:
            to_point = point - np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
            reach = float(np.linalg.norm(to_point))
            if reach < 1e-9:
                residuals.append(0.0)
                continue
            to_point = to_point / reach
        else:
            to_point = direction
        dot = float(np.clip(abs(segment @ to_point), 0.0, 1.0))
        residuals.append(math.degrees(math.acos(dot)))
    return residuals


def vertical_vp(segments):
    """Fit the vertical vanishing point of a frame's candidate segments.

    Returns (v, residual angles in degrees), or None when there is not enough to
    fit.  v is the vertical *direction* as a unit 3-vector: the vanishing point is
    where that direction meets the image plane, so a third component of zero just
    means the point is at infinity and a level camera needs no special case.

    The longest candidates are the least ambiguous, so they fix the direction
    first; every candidate is then reweighted against it (a Cauchy weight) so the
    direction settles on the bundle most of the picture agrees with.
    """
    import numpy as np

    if len(segments) < 2:
        return None
    normals = _line_normals(segments)
    lengths = np.array([s[4] for s in segments], dtype=float)
    # Seed on the long ones; a passel of short segments would otherwise drag the
    # direction towards a distant, all-of-them-ish point.
    seed = np.argsort(-lengths)[: max(6, len(segments) // 10)]
    scatter = (normals[seed] * lengths[seed][:, None]).T @ normals[seed]
    v = np.linalg.eigh(scatter)[1][:, 0]
    for _ in range(LINE_VP_IRLS):
        residuals = np.asarray(vp_residuals(segments, v), dtype=float)
        spread = 1.4826 * float(np.median(np.abs(residuals - np.median(residuals))))
        weights = lengths / (1.0 + (residuals / max(0.25, spread)) ** 2)
        scatter = (normals * weights[:, None]).T @ normals
        v = np.linalg.eigh(scatter)[1][:, 0]
    return v, vp_residuals(segments, v)


def vertical_candidates(segments, seed_deg: float = LINE_SEED_DEG) -> list[tuple]:
    """Segments whose direction alone does not rule vertical out.

    Only a seed: on a camera held anywhere near upright, a world-vertical cannot
    point much more than 60 degrees away from the image's own vertical, so this
    cheaply drops the horizon, roofs and other horizontals before any vanishing
    point is considered.  Which of the survivors really are vertical is decided by
    their agreement on one point, not by their direction.
    """
    return [s for s in segments if abs(segment_angle_deg(s[0], s[1], s[2], s[3])) <= seed_deg]


def _pairwise_vps(normals, keep):
    """Vanishing-point directions proposed by every pair of the given segments.

    Two lines meet in a point, so each pair proposes one candidate direction
    (d = n1 x n2, the vector at right angles to both planes).  The proposals are
    then judged by how much of the frame agrees with them.
    """
    import numpy as np

    proposals = []
    for a in range(len(keep)):
        for b in range(a + 1, len(keep)):
            direction = np.cross(normals[keep[a]], normals[keep[b]])
            size = float(np.linalg.norm(direction))
            if size > 1e-9:
                proposals.append(direction / size)
    return proposals


def select_vertical_segments(segments) -> tuple[list[tuple], str]:
    """(kept segments, what happened) - the candidates sharing one vertical VP.

    A vanishing point is proposed by pairs of the longest candidates and scored by
    the total length of the candidates that agree with it (each within
    LINE_VP_GATE degrees); the best proposal is then refined against its own
    bundle.  Scoring by length is what keeps a large crowd of short segments from
    voting down a few long building edges.
    """
    import numpy as np

    upright = vertical_candidates(segments)
    if len(upright) < LINE_VP_MIN_SUPPORT:
        return (
            [],
            f"only {len(upright)} candidate(s) point within {LINE_SEED_DEG:g} deg of vertical",
        )
    normals = _line_normals(upright)
    lengths = np.array([s[4] for s in upright], dtype=float)
    keep = np.argsort(-lengths)[: max(LINE_VP_MIN_SUPPORT, min(LINE_VP_SEED, len(upright)))]

    best_score, best, best_bundle = -1.0, None, None
    for direction in _pairwise_vps(normals, keep):
        residuals = np.asarray(vp_residuals(upright, direction), dtype=float)
        bundle = np.flatnonzero(residuals <= LINE_VP_GATE)
        score = float(lengths[bundle].sum())
        if score > best_score:
            best_score, best, best_bundle = score, direction, bundle
    if best is None or len(best_bundle) < LINE_VP_MIN_SUPPORT:
        return [], f"no bundle of {LINE_VP_MIN_SUPPORT} agreed on a vanishing point"

    # Refine on the winning bundle, then judge the whole frame against it.
    refined = vertical_vp([upright[i] for i in best_bundle])
    v = refined[0] if refined is not None else best
    residuals = np.asarray(vp_residuals(upright, v), dtype=float)
    kept = [
        segment
        for segment, residual in zip(upright, residuals, strict=True)
        if residual <= LINE_VP_GATE
    ]
    if len(kept) < LINE_VP_MIN_SUPPORT:
        return [], f"only {len(kept)} line(s) agreed on a vanishing point"
    return kept, f"{len(kept)} of {len(upright)} agree with the vanishing point"


def _resolve_image(raw: str, pto_path: Path) -> Path | None:
    """Find the file an image line names, the way the stitch path would."""
    for candidate in (Path(raw), Path.cwd() / raw, Path(pto_path).parent / raw):
        if candidate.is_file():
            return candidate
    return None


def detect_lines(
    src_pto: Path,
    dst_pto: Path,
    *,
    min_fraction: float = LINE_MIN_FRACTION,
    max_per_image: int | None = None,
) -> str:
    """Write our own vertical-line control points into a copy of a project.

    One t1 line per surviving segment, in linefind's own format
    (``c nI NI x.. y.. X.. Y.. t1``, both ends in the same image), so nothing
    downstream - Hugin's tools included - can tell the difference.  Returns a
    one-line report for the transcript.
    """
    text = Path(src_pto).read_text(encoding="utf-8", errors="replace")
    body = text.split("\n")
    while body and not body[-1].strip():
        body.pop()

    written: list[str] = []
    per_frame: list[str] = []
    total = 0
    for index, raw in enumerate(_pto_image_paths(src_pto)):
        source = _resolve_image(raw, src_pto)
        if source is None:
            per_frame.append(f"{index}: unreadable")
            continue
        gray = load_gray(source)
        segments = segments_from_array(gray, min_fraction * max(gray.shape))
        total += len(segments)
        kept, note = select_vertical_segments(segments)
        if max_per_image:
            kept = sorted(kept, key=lambda segment: segment[4], reverse=True)[:max_per_image]
        for x1, y1, x2, y2, _length, _score in kept:
            written.append(f"c n{index} N{index} x{x1:.3f} y{y1:.3f} X{x2:.3f} Y{y2:.3f} t1")
        per_frame.append(f"{index}: {len(kept)}" if kept else f"{index}: none ({note})")

    Path(dst_pto).write_text("\n".join([*body, *written]) + "\n", encoding="utf-8")
    report = (
        f"{total} segment(s) found, {len(written)} written as vertical lines"
        f" (per frame: {', '.join(per_frame)})"
    )
    if not written:
        report += " - WARNING: no vertical lines, so nothing levels the stitch"
    return report


def _t1_lines(pto_path: Path) -> list[tuple[int, float, float, float, float]]:
    """(image, x1, y1, x2, y2) for every vertical (t1) line control point."""
    found: list[tuple[int, float, float, float, float]] = []
    for line in Path(pto_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("c "):
            continue
        values: dict[str, str] = {}
        for token in line.split()[1:]:
            values[token[0]] = token[1:].lstrip("=")
        if values.get("t") != "1":
            continue
        try:
            found.append(
                (
                    int(values["n"]),
                    float(values["x"]),
                    float(values["y"]),
                    float(values["X"]),
                    float(values["Y"]),
                )
            )
        except KeyError, ValueError:
            continue
    return found


def line_tilts(tools: dict, src_pto: Path) -> list[tuple[int, float]]:
    """(image, tilt from panorama-vertical in degrees) for each t1 line.

    Measured through pano_trafo, so it is what the stitch will actually do.  A
    world-vertical keeps a constant azimuth, so in an equirectangular panorama it
    comes out exactly vertical however the camera was pitched; any tilt here is a
    line that leans, not a projection artefact.
    """
    entries = _t1_lines(src_pto)
    if not entries:
        return []
    payload = "".join(f"{n} {x} {y}\n{n} {X} {Y}\n" for n, x, y, X, Y in entries)
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
    return [
        (entries[i][0], _normalise_angle(math.degrees(math.atan2(bx - ax, by - ay))))
        for i, ((ax, ay), (bx, by)) in enumerate(zip(points[::2], points[1::2], strict=True))
    ]


def line_tilt_report(tools: dict, src_pto: Path) -> str:
    """How vertical the lines came out in the finished panorama.

    A report, not a gate: cpclean has already decided which lines to keep, so this
    only says how well the survivors agree - a set of true verticals lands within
    a few tenths of a degree of each other.
    """
    tilts = line_tilts(tools, src_pto)
    if not tilts:
        return "No vertical lines survived into the final project"
    magnitudes = sorted(abs(tilt) for _, tilt in tilts)
    worst_image, worst = max(tilts, key=lambda item: abs(item[1]))
    outside = sum(1 for value in magnitudes if value > LINE_LEAN_TOLERANCE)
    return (
        f"{len(tilts)} line(s) came out: median {magnitudes[len(magnitudes) // 2]:.2f} deg,"
        f" worst {worst:+.2f} deg (image {worst_image}), {outside} over"
        f" {LINE_LEAN_TOLERANCE:g} deg"
    )


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

        def detect_step(dst: Path) -> Step:
            def run() -> str:
                return detect_lines(
                    pto / "02_features.pto",
                    dst,
                    min_fraction=args.line_min_length,
                    max_per_image=args.line_count,
                )

            return Step("Finding vertical lines", on_done=run)

        steps.append(checkpoint("Finding vertical lines", pto / "03_vlines.pto", detect_step))

    # Without lines the anchor is copied straight from the cpfind project.
    lines_src = pto / "03_vlines.pto" if args.do_lines else pto / "02_features.pto"

    def anchor_action() -> str | None:
        copy_file(lines_src, pto / "04_anchor.pto")
        return f"Middle frame {frames[0].name} is listed first - autooptimiser starts from it"

    steps.append(
        Step("Setting anchor", on_done=anchor_action, outputs=[pto / "04_anchor.pto"])
        if args.force or not (pto / "04_anchor.pto").exists()
        else Step("Setting anchor", note="Exists: 04_anchor.pto - skipping")
    )

    # cpclean before the fit: only the feature matches mean anything yet (our
    # lines have not been judged by any geometry), so this pass trims those.
    sigma = f"--max-distance={args.clean_sigma:g}"
    if args.clean:
        steps.append(
            checkpoint(
                "Removing bad matches",
                pto / "05_clean.pto",
                lambda dst: Step(
                    "Removing bad matches",
                    [
                        tools["cpclean"],
                        "--whole-pano-checking",
                        sigma,
                        f"--output={dst}",
                        pto / "04_anchor.pto",
                    ],
                ),
            )
        )
    geometry0 = pto / "05_clean.pto" if args.clean else pto / "04_anchor.pto"

    steps.append(
        checkpoint(
            "Optimising positions",
            pto / "06_geometry1.pto",
            lambda dst: Step(
                "Optimising positions", [tools["autooptimiser"], "-p", "-o", dst, geometry0]
            ),
        )
    )

    # Now that there is a fit to judge them against, clean again - this time with
    # the line control points included, so cpclean weighs our verticals alongside
    # cpfind's matches.  On the test night stack this cut proved blunt (it removed
    # one line out of hundreds, since a line's error in pano pixels is small
    # against the feature matches'): which lines are vertical is really decided by
    # the detector's vanishing-point test above, and this pass is here for the
    # cases where cpclean does have something to say.
    if args.clean:
        steps.append(
            checkpoint(
                "Cleaning control points",
                pto / "07_cleanlines.pto",
                lambda dst: Step(
                    "Cleaning control points",
                    [
                        tools["cpclean"],
                        "--check-line-cp",
                        sigma,
                        f"--output={dst}",
                        pto / "06_geometry1.pto",
                    ],
                ),
            )
        )
    geometry1 = pto / "07_cleanlines.pto" if args.clean else pto / "06_geometry1.pto"

    if args.overlays_on:
        if args.do_lines:
            kept_pto = pto / "07_cleanlines.pto" if args.clean else None

            def overlays_action(kept_pto=kept_pto) -> str:
                return draw_line_overlays(pto / "03_vlines.pto", kept_pto, overlays_dir)

            steps.append(Step("Drawing line overlays", on_done=overlays_action))
        else:
            steps.append(
                Step(
                    "Drawing line overlays",
                    note="--no-lines was given, so there are no lines to draw",
                )
            )

    # Our own hard backstop, off unless asked for: cpclean's cut is statistical
    # and can leave a large absolute error behind on a uniformly bad stack.
    if args.clean and args.clean_distance:

        def clean_action() -> str | None:
            return filter_residuals(tools, geometry1, pto / "08_residual.pto", args.threshold)

        steps.append(
            Step(
                "Removing points with error above threshold",
                on_done=clean_action,
                outputs=[pto / "08_residual.pto"],
            )
            if args.force or not (pto / "08_residual.pto").exists()
            else Step(
                "Removing points with error above threshold",
                note="Exists: 08_residual.pto - skipping",
            )
        )
    geometry2_src = pto / "08_residual.pto" if (args.clean and args.clean_distance) else geometry1

    steps.append(
        checkpoint(
            "Optimising positions again",
            pto / "09_geometry2.pto",
            lambda dst: Step(
                "Optimising positions again",
                [tools["autooptimiser"], "-p", "-o", dst, geometry2_src],
                on_done=lambda: residual_stats(tools, pto / "09_geometry2.pto"),
            ),
        )
    )
    need_preview = args.force or not (pto / "12_preview.pto").exists()
    straighten_tmp = pto / "10_straight.pto"
    yaw_tmp = pto / "11_yaw.pto"
    if args.straighten and need_preview:
        steps.append(
            Step(
                "Straightening",
                [
                    tools["pano_modify"],
                    "-o",
                    straighten_tmp,
                    "--straighten",
                    pto / "09_geometry2.pto",
                ],
            )
        )

    def cleanup_tmp() -> str | None:
        straighten_tmp.unlink(missing_ok=True)
        return None

    yaw_src = straighten_tmp if (args.straighten and need_preview) else pto / "09_geometry2.pto"
    steps.append(
        Step(
            "Centering (yaw)",
            [tools["pano_modify"], "-o", yaw_tmp, "--center", f"--projection={proj}", yaw_src],
            on_done=cleanup_tmp if (args.straighten and need_preview) else None,
            outputs=[yaw_tmp],
        )
        if need_preview
        else Step("Centering (yaw)", note="Exists: 12_preview.pto - skipping")
    )

    # Final aim: the scripted equivalent of dragging the view up/down in Hugin's
    # GL preview. pano_modify --center only centres yaw, so the images' mean
    # pitch is removed here with an explicit rotation - but only when nothing
    # else has aimed the view: with vertical lines in the fit, the fit has
    # already put the world's vertical where it belongs, and rotating the
    # panorama again would tilt it away from there.
    def center_step(dst: Path) -> Step:
        def cmd() -> list:
            argv = [tools["pano_modify"], "-o", dst]
            if args.center and not _t1_lines(yaw_tmp):
                pm = mean_pitch(yaw_tmp)
                if abs(pm) > 0.005:
                    argv += ["--rotate", f"0,{-pm:.5f},0"]
            argv += ["--crop=AUTO", f"--projection={proj}", yaw_tmp]
            return argv

        return Step("Centering and cropping", cmd_fn=cmd)

    steps.append(checkpoint("Centering and cropping", pto / "12_preview.pto", center_step))

    def size_step(dst: Path) -> Step:
        return Step(
            "Calculating output size",
            [tools["pano_modify"], "-o", dst, "--canvas=AUTO", *ldr, pto / "12_preview.pto"],
            outputs=[dst],
        )

    steps.append(
        size_step(pto / "13_sized.pto")
        if args.force or not (pto / "13_sized.pto").exists()
        else Step("Calculating output size", note="Exists: 13_sized.pto - skipping")
    )

    # A report, not a gate: the lines cpclean kept should now agree with each
    # other to within a few tenths of a degree.
    if args.do_lines:
        steps.append(
            Step(
                "Checking line verticality",
                on_done=lambda: line_tilt_report(tools, pto / "13_sized.pto"),
            )
        )

    steps.append(
        Step(
            "Checking image paths",
            on_done=check_images_action(pto / "13_sized.pto", frames[0].parent),
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
                pto / "13_sized.pto",
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
                pto / "13_sized.pto",
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
    """-l/--lines [N] / --no-lines on one help line.

    Bare -l (the default) writes every line the detector keeps; -l N caps that at
    N per frame; --no-lines skips detection.  '-l filtered' used to ask for a
    larger linefind pool and filter it - cpclean does that now, so it is refused
    with an explanation rather than a bare "expected an integer".
    """

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs="?", const=True, default=True, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string == "--no-lines":
            namespace.do_lines = False
            return
        if values is True:
            namespace.do_lines = True
            namespace.line_count = None
            return
        if isinstance(values, str) and values.lower().startswith("filter"):
            raise argparse.ArgumentError(
                self,
                "'-l filtered' was removed: every detected line is now written and"
                " cpclean drops the ones that disagree (see --clean-sigma)",
            )
        try:
            count = int(values)
        except (TypeError, ValueError) as err:
            raise argparse.ArgumentError(
                self, f"expected a whole number of lines, got {values!r}"
            ) from err
        if count < 1:
            raise argparse.ArgumentError(self, "count must be >= 1")
        namespace.do_lines = True
        namespace.line_count = count


def add_arguments(sub) -> None:
    """Register Pano's own options on its subcommand parser."""
    sub.add_argument(
        "--clean-distance",
        "--no-clean-distance",
        dest="clean_distance",
        action=core.Toggle,
        off=["--no-clean-distance"],
        default=False,
        help="also drop control points further than --threshold apart in the panorama"
        " (default: off; cpclean's statistical cut is the primary cleaner)",
    )
    sub.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        metavar="PX",
        help="the distance --clean-distance drops control points beyond, in pixels"
        " (default: 3; unused unless --clean-distance is given)",
    )
    sub.add_argument(
        "-l",
        "--lines",
        "--line",
        "--no-lines",
        dest="do_lines",
        action=LinesToggle,
        metavar="N",
        help="detect vertical lines and write them as control points (default: on,"
        " every line the detector keeps); '-l N' caps that at N per frame,"
        " --no-lines skips detection",
    )
    sub.set_defaults(line_count=None)
    sub.add_argument(
        "--line-min-length",
        type=float,
        metavar="FRACTION",
        default=LINE_MIN_FRACTION,
        help="ignore line segments shorter than this share of the frame's long side"
        " (default: 0.01, i.e. 1%%)",
    )
    sub.add_argument(
        "-s",
        "--straighten",
        "--no-straighten",
        dest="straighten",
        action=core.Toggle,
        off=["--no-straighten"],
        default=False,
        help="also level the panorama with Hugin's straighten, from its lines"
        " (default: off - the vertical lines already level the fit, and straighten"
        " was measured to add a fraction of a degree of roll on top)",
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
