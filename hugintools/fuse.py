"""Fuse - align and fuse a stack of frames to reduce noise.

Aligns a frame stack to each other, then combines (fuses) them - mean / median /
sigma-clipped / winsorised - to produce a noise-free image, plus a sharpened
JPEG and an animated GIF.

Layout of the output folder
---------------------------
    <outdir>/
        <name>_fuse_mean.jpg        sharpened JPEGs (the deliverables)
        <name>_fuse_winsor.jpg
        <name>_fuse_median.jpg
        <name>_fuse_clean.jpg       the frames with moving parts excluded
        <name>_fuse_action.jpg      every moving part composited back in
        <name>_fuse.gif
        .hugin_tools/fuse/          every intermediate (deleted unless kept)
            aligned/                aligned frames ("stack images")
            merged/                 the merged TIFFs
            masks/                  motion masks + red overlays (--overlays)

Everything except the JPEGs/GIF is deleted by default.  Use -k/--keep to keep
the intermediates, --keep-aligned / --keep-merged / --keep-ptos for individual
kinds, or -d/--develop to keep everything and turn on verbose logging.

Moving parts
------------
    --clean  : leave the moving part out, keeping the noise reduction of the
               mean over the frames that did see background there.
    --action : composite every detected instance back at full opacity.

Both come from one detection pass.  Fine-tune with --mask-z, --mask-min-area,
--mask-dilate, --mask-feather; --overlays (implied by -d) keeps the masks and a
check sheet for inspection.

Alignment engines
-----------------
    ais    : align_image_stack, non-default (full-res, dense, translation) [default]
    manual : Hugin cpfind -> cpclean -> autooptimiser -> nona remap
    all    : run manual into <outdir>/manual and ais into <outdir>/ais
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from . import core
from .core import RunPlan, Step, scratch_dir

# ---------------------------------------------------------------------------
#  Tool discovery
# ---------------------------------------------------------------------------
FUSE_REQUIRED = [
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


def resolve_tools() -> dict:
    """Find every external program Fuse needs."""
    tools = core.resolve_tools(FUSE_REQUIRED, ["exiftool"])
    tools["magick"] = core.find_tool("magick") or core.find_tool("convert")  # IM7 / IM6
    return tools


# ---------------------------------------------------------------------------
#  Source frames
# ---------------------------------------------------------------------------
FUSE_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}


def is_our_output(path: Path) -> bool:
    """Our own results (…_fuse_…, and legacy …_MFNR_…) are never input frames."""
    return "_fuse" in path.stem or "_MFNR" in path.stem


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


def frame_warnings(frames: list[Path], aligner: str) -> list[str]:
    """Warn but carry on when a manual engine stack is not all one size."""
    if aligner in ("ais", "all"):
        return []
    sizes = {s for p in frames if (s := frame_size(p)) is not None}
    if len(sizes) <= 1:
        return []
    return [
        "[yellow]Frames have mixed pixel sizes: "
        + ", ".join(f"{w}x{h}" for w, h in sorted(sizes))
        + " (fine for a cropped frame, but check nothing stray like an old"
        " merge ended up in the pile)[/yellow]"
    ]


# ---------------------------------------------------------------------------
#  Moving parts: find them against the stack median, then fuse around them
# ---------------------------------------------------------------------------
# A frame whose difference from the plate is far above the others' is globally
# unlike them: soft from camera shake, or exposed differently.
UNLIKE_FRAME_RATIO = 1.5


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
        return self.clean or self.action or self.masks


def run_motion_fusion(
    aligned: list[Path],
    out_clean: Path | None,
    out_action: Path | None,
    motion: MotionOptions,
    masks_dir: Path | None = None,
) -> str:
    """Fuse an aligned stack so that whatever moved is excluded, or kept per frame.

    The median of the stack is the reference: a pixel the subject occupies in
    fewer than half the frames is background there, so |frame - median|
    isolates the subject in each frame.

    Returns a one-line report for the transcript.
    """
    # Heavy image libraries are imported here, not at module import, so a Pano
    # run never pays for them.
    import numpy as np
    import tifffile
    from PIL import Image
    from scipy import ndimage as ndi

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

    # Photometry.  Match each frame per channel to the plate before looking for
    # differences, and keep the matched frames for the output.
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

    # Where the median has real background behind it.
    valid_stack = np.stack([valid for _, valid in parts])
    painted = np.any(valid_stack, axis=0)
    known = valid_stack.sum(axis=0) > n // 2

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
    # having moved: the stack's own disagreement there, not local texture.
    spread = np.empty(shape[:2], np.float32)
    for y0 in range(0, shape[0], 256):  # banded, so the stack copy stays small
        y1 = min(y0 + 256, shape[0])
        band = np.stack([d[y0:y1] for d in diffs], axis=0)
        spread[y0:y1] = np.median(band, axis=0)
        del band
    local = ndi.maximum_filter(spread, size=9).astype(np.float32)
    del spread

    # Two floors: an absolute one, and a relative one for flat bright areas.
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

    # The thin band only one or two frames painted: borrow a real pixel.
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
        # Compose by ownership rather than by order.
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
        # Keep the masks, plus a look-at-me version of each frame with the mask
        # painted over it in red, the way a mask reads in Lightroom.
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
#  Steps
# ---------------------------------------------------------------------------
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

    steps = [
        Step("Reading frames", [t["pto_gen"], "-o", img, *frames]),
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
                img,
            ],
        ),
        # Clean BEFORE optimising so outliers don't skew the geometry.
        Step(
            "Removing bad matches",
            [t["cpclean"], "--whole-pano-checking", f"--output={clean}", cps],
        ),
    ]

    opt = "y,p,r" if motion == "low" else "y,p,r,TrX,TrY,TrZ,v"
    steps.append(
        Step("Choosing alignment parameters", [t["pto_var"], "--opt", opt, "-o", s0, clean])
    )
    steps += [
        Step("Optimising alignment (pass 1)", [t["autooptimiser"], "-n", "-o", s1, s0]),
        Step("Refining matches", [t["cpclean"], f"--output={s1}", "--max-distance=3.3", s1]),
        Step("Optimising alignment (pass 2)", [t["autooptimiser"], "-n", "-o", s2, s1]),
    ]
    if photometric:
        steps.append(
            Step(
                "Matching exposure, vignetting and white balance",
                [t["autooptimiser"], "-m", "-o", s2, s2],
            )
        )

    steps += [
        Step(
            "Preparing output",
            [
                t["pano_modify"],
                "-o",
                s2,
                "--center",
                "--canvas=AUTO",
                "--crop=AUTO",
                "--projection=0",
                "--output-uncropped-tiff",
                "--output-type=REMAP",
                s2,
            ],
        ),
        Step(
            "Rendering aligned frames",
            [t["hugin_executor"], "--stitching", f"--prefix={aligned_dir / 'rendered_'}", s2],
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
                aligned_dir / "ais_",
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
                *frames,
            ],
        )
    ]


def output_steps(
    tools: dict,
    deliver: Path,
    work: Path,
    aligned_dir: Path,
    ref: Path,
    merge_types: list[str],
    motion: MotionOptions,
    animation: bool,
) -> list[Step]:
    """Combine / sharpen / (report) steps for one engine.

    Deliverables (JPEGs, GIF) go to `deliver`; everything else - merged TIFFs,
    masks - goes to `work`, which is swept unless asked for.
    """
    base = f"{ref.stem}_fuse"
    merged_dir = work / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
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

    def add_jpeg(tif: Path) -> None:
        """The sharpened JPEG for one merged product."""
        if not magick:
            return  # without ImageMagick the TIFF is the only artefact; keep it
        jpg = deliver / f"{tif.stem}.jpg"
        steps.append(
            Step(
                f"Sharpening {jpg.name}",
                cmd=[magick, tif, "-unsharp", "0x0.7+1.0+0", "-quality", "95", jpg],
            )
        )

    for name in merge_types:
        extra, label = MERGE_TYPES[name]
        tif = merged_dir / f"{base}_{name}.tif"

        def combine_step(tif=tif, extra=extra):
            return [stacker, *extra, f"--output={tif}", *aligned_frames()]

        steps.append(Step(f"Combining frames ({label})", cmd_fn=combine_step))
        add_jpeg(tif)

    # Whatever moves between frames: one detection pass feeds both products.
    if motion.enabled:
        clean_tif = merged_dir / f"{base}_clean.tif" if motion.clean else None
        action_tif = merged_dir / f"{base}_action.tif" if motion.action else None
        masks_dir = work / "masks" if motion.masks else None

        def motion_step(clean_tif=clean_tif, action_tif=action_tif, masks_dir=masks_dir):
            return run_motion_fusion(aligned_frames(), clean_tif, action_tif, motion, masks_dir)

        steps.append(Step("Detecting motion", run_fn=motion_step))
        for tif in (clean_tif, action_tif):
            if tif is not None:
                add_jpeg(tif)

    if magick and animation:
        gif = deliver / f"{base}.gif"

        def gif_step(gif=gif):
            return [
                magick,
                *aligned_frames(),
                "-resize",
                "1080x",
                "-delay",
                "30",
                "-loop",
                "0",
                gif,
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
            targets = [
                p for p in sorted(deliver.glob(f"{base}_*")) if p.suffix.lower() in (".jpg", ".tif")
            ]
            targets += [p for p in sorted(merged_dir.glob(f"{base}_*")) if p.suffix == ".tif"]
            argv = [exiftool, "-overwrite_original"]
            # exiftool needs "-charset filename=utf8" on Linux/macOS for accented
            # paths, but on Windows it uses wide filenames and that option makes
            # it fail to find the file, so omit it there.
            if os.name != "nt":
                argv += ["-charset", "filename=utf8"]
            # -Orientation#=1 forces the tag to "normal" (the "#" means numeric):
            # hugin_stacker writes TIFFs tagged "rotate 180", which flips them in
            # viewers even though the pixels are fine.
            argv += ["-TagsFromFile", ref, "-all:all", "-Orientation#=1"]
            argv += targets
            return argv

        steps.append(Step(f"Copying metadata from {ref.name}", cmd_fn=tag_step))

    return steps


def build(tools: dict, frames: list[Path], outdir: Path, args) -> RunPlan:
    """Group the steps by alignment engine ('all' runs both, each in its own folder)."""
    plan: dict[str, list[Step]] = {}
    scratch: dict[str, Path] = {}
    output_dir: dict[str, Path] = {}
    modes = ("manual", "ais") if args.aligner == "all" else (args.aligner,)
    for mode in modes:
        engine = mode if args.aligner == "all" else "fuse"
        deliver = outdir / mode if args.aligner == "all" else outdir
        work = scratch_dir(outdir, engine, args.scratch)
        aligned_dir = work / "aligned"
        # The output folders must exist before any tool writes into them.
        deliver.mkdir(parents=True, exist_ok=True)
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
            deliver,
            work,
            aligned_dir,
            frames[-1],
            args.merge_types,
            args.motion_options,
            args.do_animation,
        )
        plan[mode] = steps
        scratch[mode] = work
        output_dir[mode] = deliver
    return RunPlan(plan=plan, scratch=scratch, output_dir=output_dir)


def collect(runplan: RunPlan) -> list[tuple[str, Path]]:
    """The deliverables actually sitting in each engine's output folder."""
    results: list[tuple[str, Path]] = []
    for mode, folder in runplan.output_dir.items():
        for p in sorted(folder.iterdir()):
            if (
                p.is_file()
                and "_fuse" in p.name
                and p.suffix.lower() in {".jpg", ".tif", ".tiff", ".gif"}
            ):
                results.append((mode, p))
    return results


# ---------------------------------------------------------------------------
#  Options
# ---------------------------------------------------------------------------
class AllOutputsAction(argparse.Action):
    """--all switches on every output type at the point it appears.

    Options are applied in the order they are given, so "--all --no-median"
    produces everything except the median merge, while "--no-action --all"
    turns the action composite back on.
    """

    OUTPUTS = ("do_mean", "do_winsor", "do_median", "do_clean", "do_action", "do_animation")

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, default=False, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)
        for name in self.OUTPUTS:
            setattr(namespace, name, True)


OUTPUT_NAMES = ("do_mean", "do_winsor", "do_median", "do_clean", "do_action", "do_animation")


def add_arguments(sub) -> None:
    """Register Fuse's own options on its subcommand parser."""
    sub.add_argument(
        "--aligner",
        choices=["manual", "ais", "all"],
        default="ais",
        help="alignment engine (default: ais)",
    )
    sub.add_argument(
        "--motion",
        choices=["high", "low"],
        default="high",
        help="manual engine: 'low' for a steady tripod (default: high)",
    )
    sub.add_argument(
        "--photometric",
        "--no-photometric",
        dest="photometric",
        action=core.Toggle,
        off=["--no-photometric"],
        default=False,
        help="manual engine: also match exposure/vignetting/WB between frames",
    )
    sub.add_argument(
        "--all",
        dest="all_outputs",
        action=AllOutputsAction,
        help="produce every output type: mean, winsor, median, clean, action, animation",
    )
    for flag, dest, helptext in (
        ("--mean", "do_mean", "produce the mean merge (default: off)"),
        (
            "--winsor",
            "do_winsor",
            "produce the winsorised-mean merge (the default when nothing else is named)",
        ),
        ("--median", "do_median", "produce the median merge (default: off)"),
        ("--clean", "do_clean", "fuse with whatever moved between frames excluded (default: off)"),
        (
            "--action",
            "do_action",
            "fuse every detected moving instance back in, one per frame (default: off)",
        ),
    ):
        sub.add_argument(
            flag,
            f"--no-{flag[2:]}",
            dest=dest,
            action=core.Toggle,
            off=[f"--no-{flag[2:]}"],
            default=None,
            help=helptext,
        )
    sub.add_argument(
        "--animation",
        "--gif",
        "--no-animation",
        "--no-gif",
        dest="do_animation",
        action=core.Toggle,
        off=["--no-animation", "--no-gif"],
        default=None,
        help="build the animated GIF of the aligned frames (default: off)",
    )
    sub.add_argument(
        "--mask-z",
        type=float,
        metavar="Z",
        default=3.5,
        help="motion: how far from the stack median a pixel must sit to count (default: 3.5)",
    )
    sub.add_argument(
        "--mask-min-area",
        type=int,
        metavar="PX",
        default=0,
        help="motion: ignore blobs smaller than this many pixels (default: 0 = auto)",
    )
    sub.add_argument(
        "--mask-dilate",
        type=int,
        metavar="PX",
        default=12,
        help="motion: grow each mask by this radius, in pixels (default: 12)",
    )
    sub.add_argument(
        "--mask-feather",
        type=float,
        metavar="PX",
        default=4.0,
        help="motion: soften the pasted edge by this sigma, in pixels (default: 4)",
    )


def prepare_args(args) -> None:
    """Resolve Fuse's output selection after parsing."""
    if all(getattr(args, name) is None for name in OUTPUT_NAMES):
        args.do_winsor = True
    for name in OUTPUT_NAMES:
        if getattr(args, name) is None:
            setattr(args, name, False)
    if not any(getattr(args, name) for name in OUTPUT_NAMES):
        raise SystemExit(
            "No outputs selected - every output type was switched off.\n"
            "Name the ones you want, or use --all for everything."
        )
    args.motion_options = MotionOptions(
        clean=args.do_clean,
        action=args.do_action,
        masks=args.overlays_on,
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


def scratch_paths(outdir: Path, args) -> list[Path]:
    if args.aligner == "all":
        return [
            scratch_dir(outdir, "manual", args.scratch),
            scratch_dir(outdir, "ais", args.scratch),
        ]
    return [scratch_dir(outdir, "fuse", args.scratch)]


def keep(args, tools: dict) -> tuple[bool, set[str]]:
    """(keep everything, names of scratch subfolders to keep).

    Without ImageMagick the merged TIFFs are the only artefacts, so they are
    kept regardless.
    """
    keep_all = args.keep or args.develop
    names: set[str] = set()
    if args.keep_aligned:
        names.add("aligned")
    if args.keep_merged or not tools.get("magick"):
        names.add("merged")
    if args.keep_ptos:
        names.add("pto")
    if args.overlays_on:
        names.add("masks")
    return keep_all, names
