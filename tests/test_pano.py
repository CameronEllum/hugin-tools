"""Pano-specific behaviour: control-point parsing, options, build layout, keep."""

from __future__ import annotations

import argparse
from pathlib import Path

from hugintools import pano


def make_args(argv):
    ap = argparse.ArgumentParser()
    pano.add_arguments(ap)
    args = ap.parse_args(argv)
    args.scratch = None
    args.force = False
    args.develop = False
    args.keep = False
    args.keep_aligned = False
    args.keep_ptos = False
    args.overlays_on = False
    pano.prepare_args(args)
    return args


def fake_tools():
    tools = {name: Path(name) for name in pano.PANO_REQUIRED}
    tools["exiftool"] = None
    return tools


def test_parse_c_line_normal():
    d = pano._parse_c_line("c n0 N1 x10 y20 X30 Y40 t0")
    assert d is not None
    assert (d["n"], d["N"], d["x"], d["Y"]) == (0, 1, 10.0, 40.0)


def test_parse_c_line_skips_non_normal():
    assert pano._parse_c_line("c n0 N0 x1 y2 X3 Y4 t1") is None
    assert pano._parse_c_line('i n"a.jpg"') is None
    assert pano._parse_c_line("c n0 N1 x1 y2") is None


def test_lines_toggle_variants():
    base = make_args([])
    assert base.do_lines is True
    assert base.line_count == pano.LINE_COUNT_DEFAULT
    assert base.line_filter is False

    assert make_args(["--no-lines"]).do_lines is False
    assert make_args(["-l", "3"]).line_count == 3

    filtered = make_args(["-l", "filtered"])
    assert filtered.line_filter is True
    assert filtered.line_count == pano.LINE_COUNT_FILTERED
    assert make_args(["--line", "filtered"]).line_filter is True


def test_aligned_default_off():
    assert make_args([]).aligned_on is False
    assert make_args(["--aligned"]).aligned_on is True


def test_develop_implies_aligned():
    ap = argparse.ArgumentParser()
    pano.add_arguments(ap)
    args = ap.parse_args([])
    args.develop = True
    args.overlays_on = True
    pano.prepare_args(args)
    assert args.aligned_on is True


def test_build_places_intermediates_in_scratch(tmp_path):
    args = make_args([])
    frames = [tmp_path / n for n in ("a.jpg", "b.jpg", "c.jpg")]
    runplan = pano.build(fake_tools(), frames, tmp_path, args)
    assert set(runplan.plan) == {"pano"}
    assert runplan.scratch["pano"] == tmp_path / ".hugin_tools" / "pano"
    assert runplan.output_dir["pano"] == tmp_path
    assert (runplan.scratch["pano"] / "pto").is_dir()
    assert args.prefix == "c_equirectangular"


def test_build_respects_explicit_prefix(tmp_path):
    args = make_args(["--prefix", "myrow"])
    frames = [tmp_path / n for n in ("a.jpg", "b.jpg")]
    pano.build(fake_tools(), frames, tmp_path, args)
    assert args.prefix == "myrow"


def test_keep_names():
    args = make_args([])
    args.keep_ptos = True
    _, names = pano.keep(args, fake_tools())
    assert "pto" in names

    _, names = pano.keep(make_args(["--aligned"]), fake_tools())
    assert "aligned" in names

    args = make_args([])
    args.overlays_on = True
    _, names = pano.keep(args, fake_tools())
    assert "lines" in names


def test_keep_all():
    args = make_args([])
    args.keep = True
    keep_all, _ = pano.keep(args, fake_tools())
    assert keep_all is True
