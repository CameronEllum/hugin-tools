"""Fuse-specific behaviour: options, frame geometry, build layout, keep policy."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import pytest

from hugin_tools import fuse


def make_args(argv, overlays_on=False):
    ap = argparse.ArgumentParser()
    fuse.add_arguments(ap)
    args = ap.parse_args(argv)
    # The common options normally come from the shared parent parser.
    args.scratch = None
    args.force = False
    args.develop = False
    args.keep = False
    args.keep_aligned = False
    args.keep_merged = False
    args.keep_ptos = False
    args.overlays_on = overlays_on
    fuse.prepare_args(args)
    return args


def fake_tools(with_magick=False):
    tools = {name: Path(name) for name in fuse.FUSE_REQUIRED}
    tools["magick"] = Path("magick") if with_magick else None
    tools["exiftool"] = None
    return tools


def png_bytes(w: int, h: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", w, h)


def test_default_output_is_winsor():
    assert make_args([]).merge_types == ["winsor"]


def test_naming_an_output_drops_the_default():
    args = make_args(["--clean"])
    assert args.do_clean is True
    assert args.merge_types == []
    assert args.motion_options.clean is True


def test_all_switches_everything_on():
    args = make_args(["--all"])
    assert args.merge_types == ["mean", "winsor", "median"]
    assert args.motion_options.clean and args.motion_options.action


def test_no_outputs_selected_errors():
    with pytest.raises(SystemExit):
        make_args(["--no-winsor"])


def test_overlays_enable_the_detection_pass():
    args = make_args(["--all"], overlays_on=True)
    assert args.motion_options.masks is True
    assert args.motion_options.enabled is True


def test_frame_size_reads_png(tmp_path):
    a = tmp_path / "a.png"
    a.write_bytes(png_bytes(10, 20))
    assert fuse.frame_size(a) == (10, 20)


def test_check_same_size(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    c = tmp_path / "c.png"
    a.write_bytes(png_bytes(10, 20))
    b.write_bytes(png_bytes(10, 20))
    c.write_bytes(png_bytes(30, 40))
    fuse.check_same_size([a, b])  # no raise
    with pytest.raises(SystemExit):
        fuse.check_same_size([a, c])


def test_frame_warnings_only_for_manual(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(png_bytes(10, 20))
    b.write_bytes(png_bytes(30, 40))
    assert fuse.frame_warnings([a, b], "ais") == []
    assert fuse.frame_warnings([a, b], "manual")


def test_build_places_intermediates_in_scratch(tmp_path):
    args = make_args(["--all"])
    frames = [tmp_path / "a.tif", tmp_path / "b.tif"]
    runplan = fuse.build(fake_tools(), frames, tmp_path, args)
    assert set(runplan.plan) == {"ais"}
    assert runplan.scratch["ais"] == tmp_path / ".hugin_tools" / "fuse"
    assert runplan.output_dir["ais"] == tmp_path
    assert (runplan.scratch["ais"] / "merged").is_dir()


def test_build_aligner_all_splits_engines(tmp_path):
    args = make_args(["--all", "--aligner", "all"])
    frames = [tmp_path / "a.tif", tmp_path / "b.tif"]
    runplan = fuse.build(fake_tools(), frames, tmp_path, args)
    assert set(runplan.plan) == {"manual", "ais"}
    assert runplan.output_dir["manual"] == tmp_path / "manual"
    assert runplan.output_dir["ais"] == tmp_path / "ais"
    assert runplan.scratch["ais"] == tmp_path / ".hugin_tools" / "ais"


def test_keep_without_magick_keeps_merged():
    keep_all, names = fuse.keep(make_args(["--all"]), fake_tools(with_magick=False))
    assert keep_all is False
    assert names == {"merged"}


def test_keep_with_overlays():
    args = make_args(["--all"], overlays_on=True)
    _, names = fuse.keep(args, fake_tools(with_magick=True))
    assert "masks" in names
    assert "merged" not in names


def test_keep_all():
    args = make_args(["--all"])
    args.keep = True
    keep_all, _ = fuse.keep(args, fake_tools())
    assert keep_all is True
