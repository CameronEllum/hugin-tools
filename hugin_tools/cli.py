"""Command-line interface: ``hugin fuse`` and ``hugin pano``.

One entry point, one set of shared options (input, output, scratch, keep,
overlays, logging); each subcommand adds only its own.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import core, fuse, pano, vision
from .core import UI

FUSE_DESCRIPTION = (
    "Fuse - align and fuse a stack of frames to reduce noise. "
    "Produces sharpened JPEGs (mean / winsor / median / motion-clean / motion-action), "
    "an optional animated GIF, and keeps every intermediate in <outdir>/.hugin_tools unless -k."
)

PANO_DESCRIPTION = (
    "Pano - classic Hugin panorama workflow. Numbered .pto projects are written under "
    "<outdir>/.hugin_tools/pano after every step and deleted unless -k."
)


def _common_parser() -> argparse.ArgumentParser:
    """The options both subcommands share, defined once."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "images",
        nargs="*",
        metavar="IMAGE",
        help="frames to process: image files, glob patterns, or a folder (default: this folder)",
    )
    common.add_argument(
        "-i",
        "--input",
        dest="source",
        metavar="INPUT",
        default=None,
        help="frames to process, as with IMAGE above (default: the current folder)",
    )
    common.add_argument(
        "-o", "--outdir", default=None, help="output folder (default: current folder)"
    )
    common.add_argument(
        "--scratch",
        default=None,
        metavar="DIR",
        help="folder for intermediates (default: <outdir>/.hugin_tools)",
    )
    common.add_argument(
        "-f", "--force", action="store_true", help="redo work even if its output already exists"
    )
    common.add_argument(
        "--overlays",
        "--no-overlays",
        dest="overlays",
        action=core.Toggle,
        off=["--no-overlays"],
        default=False,
        help="write the detection check sheets (mask / line overlays) and keep them"
        " (default: off; -d implies on)",
    )
    common.add_argument(
        "-k", "--keep", action="store_true", help="keep the scratch folder (every intermediate)"
    )
    common.add_argument(
        "--keep-aligned", action="store_true", help="keep the aligned / remapped frames"
    )
    common.add_argument(
        "--keep-merged",
        "--keep-merged-tifs",
        dest="keep_merged",
        action="store_true",
        help="keep the merged TIFFs",
    )
    common.add_argument(
        "--keep-ptos",
        "--keep-projects",
        dest="keep_ptos",
        action="store_true",
        help="keep the .pto project files",
    )
    common.add_argument(
        "--clean",
        "--no-clean",
        dest="clean",
        action=core.Toggle,
        off=["--no-clean"],
        default=True,
        help="remove wrong control points with cpclean, both before and after the fit"
        " (default: on; --no-clean keeps every control point cpfind found)",
    )
    common.add_argument(
        "--clean-sigma",
        type=core.positive_float,
        metavar="N",
        default=3.0,
        help="cleaning: remove a control point when its error is more than this many"
        " standard deviations above the mean (cpclean's own knob; default: 3)"
        " - ignored with --no-clean",
    )
    common.add_argument("--log", default=None, help="also write a transcript to this file")
    common.add_argument("--no-color", action="store_true", help="disable coloured output")
    common.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print the full command line and tool output of every step",
    )
    common.add_argument(
        "-d",
        "--develop",
        "--debug",
        action="store_true",
        help="keep every intermediate and overlay, and enable verbose logging",
    )
    common.add_argument(
        "--review",
        "--no-review",
        dest="review",
        action=core.Toggle,
        off=["--no-review"],
        default=False,
        help="review the result with a local vision model (default: off)"
        " - --review fails the run if the review cannot be produced;"
        " --no-review is the explicit off",
    )
    common.add_argument(
        "--vision-url",
        default=vision.DEFAULT_URL,
        metavar="URL",
        help="OpenAI-compatible endpoint of the local vision model (default: %(default)s)",
    )
    common.add_argument(
        "--vision-model",
        default=None,
        metavar="NAME",
        help="model to review with (default: the first loaded vision model at --vision-url)",
    )
    common.add_argument(
        "--vision-context",
        type=int,
        default=None,
        metavar="TOKENS",
        help="context window to size the images to (default: read from the server, else 8192)",
    )
    common.add_argument(
        "--vision-detail",
        choices=["low", "balanced", "high"],
        default="balanced",
        help="how much image detail to send (default: balanced)",
    )
    common.add_argument(
        "--vision-timeout",
        type=float,
        default=120.0,
        metavar="SECONDS",
        help="how long to wait for the model (default: %(default)s)",
    )
    common.add_argument(
        "--vision-max-tokens",
        type=int,
        default=512,
        metavar="N",
        help="cap on the reply length (default: %(default)s)",
    )
    return common


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="hugin_tools",
        description="hugin_tools - Hugin command-line workflows.",
        formatter_class=core._HELP_FORMATTER,
    )
    sub = ap.add_subparsers(dest="command", metavar="{fuse,pano}")
    sub.required = True

    common = _common_parser()
    fuse_parser = sub.add_parser(
        "fuse",
        parents=[common],
        help="align and fuse a frame stack",
        description=FUSE_DESCRIPTION,
        formatter_class=core._HELP_FORMATTER,
    )
    fuse.add_arguments(fuse_parser)

    pano_parser = sub.add_parser(
        "pano",
        parents=[common],
        help="stitch a panorama",
        description=PANO_DESCRIPTION,
        formatter_class=core._HELP_FORMATTER,
    )
    pano.add_arguments(pano_parser)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    core.setup_console()

    if args.log:
        args.log = str(Path(args.log).resolve())
    ui = UI(verbose=args.verbose, no_color=args.no_color, log_path=args.log)

    args.overlays_on = bool(args.overlays or args.develop)
    if args.develop:
        args.verbose = True
        ui.verbose = True

    args.vision = vision.Options(
        mode=args.review,
        url=args.vision_url,
        model=args.vision_model,
        context=args.vision_context,
        detail=args.vision_detail,
        timeout=args.vision_timeout,
        max_tokens=args.vision_max_tokens,
    )

    outdir = Path(args.outdir).resolve() if args.outdir else Path.cwd()
    specs = [*args.images, *([args.source] if args.source else [])] or [Path.cwd()]

    try:
        if args.command == "fuse":
            try:
                tools = fuse.resolve_tools()
                fuse.prepare_args(args)
            except SystemExit as exc:
                ui.print(f"[red]{core.escape(str(exc))}[/red]")
                return 1
            keep_all, keep_names = fuse.keep(args, tools)
            return core.run_cli(
                ui,
                title="Fuse",
                tools=tools,
                outdir=outdir,
                specs=specs,
                exts=fuse.FUSE_EXTS,
                min_frames=1,
                anchor_middle=False,
                is_output=fuse.is_our_output,
                output_label="Fuse",
                noun="frame",
                do_chdir=False,
                mark_anchor=False,
                size_check=(fuse.check_same_size if args.aligner in ("ais", "all") else None),
                frames_extra=lambda frames: fuse.frame_warnings(frames, args.aligner),
                build=lambda tools, frames, od: fuse.build(tools, frames, od, args),
                collect=fuse.collect,
                scratch_paths=fuse.scratch_paths(outdir, args),
                keep_all=keep_all,
                keep_names=keep_names,
            )

        try:
            tools = pano.resolve_tools()
            pano.prepare_args(args)
        except SystemExit as exc:
            ui.print(f"[red]{core.escape(str(exc))}[/red]")
            return 1
        keep_all, keep_names = pano.keep(args, tools)
        return core.run_cli(
            ui,
            title="Pano",
            tools=tools,
            outdir=outdir,
            specs=specs,
            exts=pano.PANO_EXTS,
            min_frames=2,
            anchor_middle=True,
            is_output=lambda p: pano.is_our_output(p, args.prefix),
            output_label="pano",
            noun="jpgs",
            do_chdir=True,
            mark_anchor=True,
            size_check=None,
            frames_extra=None,
            build=lambda tools, frames, od: pano.build(tools, frames, od, args),
            collect=lambda runplan: pano.collect(runplan, args.prefix),
            scratch_paths=pano.scratch_paths(outdir, args),
            keep_all=keep_all,
            keep_names=keep_names,
        )
    finally:
        ui.close()


if __name__ == "__main__":
    sys.exit(main())
