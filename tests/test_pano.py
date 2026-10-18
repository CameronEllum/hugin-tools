"""Pano-specific behaviour: control-point parsing, line detection, options, build."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pytest

from hugin_tools import pano


def make_args(argv, clean=True, clean_sigma=3.0):
    ap = argparse.ArgumentParser()
    pano.add_arguments(ap)
    args = ap.parse_args(argv)
    # The common options normally come from the shared parent parser.
    args.scratch = None
    args.force = False
    args.develop = False
    args.keep = False
    args.keep_aligned = False
    args.keep_ptos = False
    args.overlays_on = False
    args.clean = clean
    args.clean_sigma = clean_sigma
    pano.prepare_args(args)
    return args


def fake_tools():
    tools = {name: Path(name) for name in pano.PANO_REQUIRED}
    tools["exiftool"] = None
    return tools


def commands(runplan):
    """The literal command lines a plan will run (closures are left alone)."""
    return [
        " ".join(str(a) for a in step.cmd)
        for steps in runplan.plan.values()
        for step in steps
        if step.cmd
    ]


def labels(runplan):
    return [step.label for steps in runplan.plan.values() for step in steps]


def build(tmp_path, argv=(), clean=True, clean_sigma=3.0):
    args = make_args(list(argv), clean=clean, clean_sigma=clean_sigma)
    frames = [tmp_path / n for n in ("a.jpg", "b.jpg")]
    return pano.build(fake_tools(), frames, tmp_path, args)


# ---------------------------------------------------------------------------
#  .pto parsing
# ---------------------------------------------------------------------------
def test_parse_c_line_normal():
    d = pano._parse_c_line("c n0 N1 x10 y20 X30 Y40 t0")
    assert d is not None
    assert (d["n"], d["N"], d["x"], d["Y"]) == (0, 1, 10.0, 40.0)


def test_parse_c_line_skips_non_normal():
    assert pano._parse_c_line("c n0 N0 x1 y2 X3 Y4 t1") is None
    assert pano._parse_c_line('i n"a.jpg"') is None
    assert pano._parse_c_line("c n0 N1 x1 y2") is None


def test_pto_lines_reads_our_format(tmp_path):
    pto = tmp_path / "lines.pto"
    pto.write_text(
        "# hugin project file\np f2 w300 h150 v360\n"
        'i w300 h300 f0 v50 Ra0 Rb0 Rc0 Rd0 Re0 r0 p0 y0 n"a.jpg"\n'
        "c n0 N1 x10 y20 X30 Y40 t0\n"
        "c n0 N0 x100.5 y5.25 X101.5 Y200.75 t1\n",
        encoding="utf-8",
    )
    lines = pano.pto_lines(pto)
    assert lines == [(0, 100.5, 5.25, 101.5, 200.75, "vertical")]


# ---------------------------------------------------------------------------
#  The line detector
# ---------------------------------------------------------------------------
def test_segment_angle_conventions():
    assert pano.segment_angle_deg(0.0, 0.0, 0.0, 10.0) == 0.0
    assert pano.segment_angle_deg(0.0, 0.0, 10.0, 0.0) == 90.0
    assert pano.segment_angle_deg(10.0, 0.0, 0.0, 0.0) == 90.0
    assert pano.segment_angle_deg(0.0, 0.0, 10.0, 100.0) == pytest.approx(5.71, abs=0.01)
    assert pano.segment_angle_deg(0.0, 0.0, -10.0, 100.0) == pytest.approx(-5.71, abs=0.01)


def test_detector_separates_verticals_from_horizontals():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    picture = np.zeros((300, 300), np.uint8)
    picture[20:280, 60:63] = 255  # vertical bars
    picture[20:280, 130:133] = 255
    picture[20:280, 200:203] = 255
    picture[150:153, 20:280] = 255  # and a horizontal one

    segments = pano.segments_from_array(picture, 20.0)
    assert segments
    kept, _note = pano.select_vertical_segments(segments)
    assert kept
    # Only the verticals survive, and they come out vertical.
    for x1, _y1, x2, _y2, _length, _score in kept:
        assert abs(x1 - x2) < 5.0
    assert max(abs(pano.segment_angle_deg(*s[:4])) for s in kept) < 5.0


def test_detect_lines_writes_t1_points(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    image = pytest.importorskip("PIL.Image")

    picture = np.zeros((300, 300), np.uint8)
    picture[:, 60:63] = 255
    picture[:, 130:133] = 255
    picture[:, 200:203] = 255
    image.fromarray(picture).save(tmp_path / "bars.png")

    src = tmp_path / "in.pto"
    src.write_text(
        "# hugin project file\np f2 w300 h150 v360\n"
        'i w300 h300 f0 v50 Ra0 Rb0 Rc0 Rd0 Re0 r0 p0 y0 n"bars.png"\n',
        encoding="utf-8",
    )
    dst = tmp_path / "out.pto"
    report = pano.detect_lines(src, dst, min_fraction=0.1)
    assert "written as vertical lines" in report

    lines = pano.pto_lines(dst)
    assert lines, report
    for image_index, x1, _y1, x2, _y2, kind in lines:
        assert image_index == 0
        assert kind == "vertical"
        assert abs(x1 - x2) < 5.0


def test_line_min_length_is_honoured():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    picture = np.zeros((300, 300), np.uint8)
    picture[:, 150:153] = 255  # full height
    picture[120:220, 60:63] = 255  # only a third of it
    everything = pano.segments_from_array(picture, 0.0)
    only_long = pano.segments_from_array(picture, 250.0)
    assert len(only_long) < len(everything)
    assert only_long
    assert min(segment[4] for segment in only_long) >= 250.0


# ---------------------------------------------------------------------------
#  The vanishing point
# ---------------------------------------------------------------------------
def segments_through(point, angles, start=600.0, end=1400.0):
    """Vertical-ish segments lying on lines that all pass through `point`."""
    px, py = point
    segments = []
    for angle in angles:
        dx, dy = math.sin(math.radians(angle)), math.cos(math.radians(angle))
        segments.append(
            (
                px + start * dx,
                py + start * dy,
                px + end * dx,
                py + end * dy,
                end - start,
                0.0,
            )
        )
    return segments


def test_vanishing_point_at_infinity():
    pytest.importorskip("numpy")
    segments = [(x, 0.0, x, 500.0, 500.0, 0.0) for x in (100.0, 220.0, 320.0, 700.0, 900.0)]
    fit = pano.vertical_vp(segments)
    assert fit is not None
    v, residuals = fit
    assert abs(v[2]) < 1e-6  # a level camera puts the point at infinity
    assert max(residuals) < 0.01
    kept, _note = pano.select_vertical_segments(segments)
    assert len(kept) == len(segments)


def test_vanishing_point_matches_ground_truth():
    pytest.importorskip("numpy")
    point = (500.0, -2500.0)
    segments = segments_through(point, [-30, -20, -10, 0, 10, 20, 30])
    fit = pano.vertical_vp(segments)
    assert fit is not None
    v, residuals = fit
    assert abs(v[2]) > 1e-6
    assert v[0] / v[2] == pytest.approx(point[0], abs=1.0)
    assert v[1] / v[2] == pytest.approx(point[1], abs=1.0)
    assert max(residuals) < 0.05


def test_leaning_lines_are_rejected():
    pytest.importorskip("numpy")
    point = (500.0, -2500.0)
    true_verticals = segments_through(point, [-18, -14, -10, -6, -2, 2, 6, 10, 14, 18])
    leaning = [
        (1400.0, 0.0, 1400.0, 1200.0, 1200.0, 0.0),  # misses the point entirely
        (1600.0, 400.0, 1620.0, 1400.0, 1000.0, 0.0),
    ]
    kept, note = pano.select_vertical_segments([*true_verticals, *leaning])
    assert len(kept) == len(true_verticals)
    assert "10 of 12" in note


def test_horizontals_never_reach_the_fit():
    pytest.importorskip("numpy")
    segments = [
        (10.0, 10.0, 10.0, 200.0, 190.0, 0.0),  # vertical
        (10.0, 10.0, 200.0, 10.0, 190.0, 0.0),  # horizontal
        (10.0, 10.0, 150.0, 90.0, 160.0, 0.0),  # 45 degrees off
    ]
    assert pano.vertical_candidates(segments) == [segments[0]]


def test_plane_normals_are_unit_vectors():
    # Regression: leaving the normals only partly normalised makes a distant
    # vanishing point look artificially good.
    np = pytest.importorskip("numpy")
    segments = [
        (10.0, 0.0, 12.0, 400.0, 400.0, 0.0),
        (300.0, 200.0, 305.0, 900.0, 700.0, 0.0),
        (10.0, 10.0, 200.0, 10.0, 190.0, 0.0),
    ]
    norms = np.linalg.norm(pano._line_normals(segments), axis=1)
    assert np.allclose(norms, 1.0)


def test_long_lines_decide_the_vanishing_point():
    pytest.importorskip("numpy")
    truth = (500.0, -2500.0)
    long_ones = segments_through(truth, [-9, -3, 3, 9], start=500.0, end=2500.0)
    # A crowd of short segments through a different point must not outvote them.
    crowd = segments_through((900.0, -1500.0), [a - 30 for a in range(-20, 21)], 60.0, 100.0)
    kept, note = pano.select_vertical_segments([*long_ones, *crowd])
    assert len(kept) == len(long_ones)
    assert "agree with the vanishing point" in note


def test_a_gently_leaning_line_is_dropped():
    pytest.importorskip("numpy")
    point = (500.0, -2500.0)
    true_verticals = segments_through(point, [-12, -8, -4, 0, 4, 8, 12])
    # A segment whose direction misses the vanishing point ray by 3 degrees.
    length = 400.0
    leaning = (
        500.0 - 0.5 * length * math.sin(math.radians(3)),
        -1000.0 + 0.5 * length * math.cos(math.radians(3)),
        500.0 + 0.5 * length * math.sin(math.radians(3)),
        -1000.0 - 0.5 * length * math.cos(math.radians(3)),
        length,
        0.0,
    )
    kept, note = pano.select_vertical_segments([*true_verticals, leaning])
    assert len(kept) == len(true_verticals)
    assert "7 of 8" in note


# ---------------------------------------------------------------------------
#  Options
# ---------------------------------------------------------------------------
def test_lines_toggle_variants():
    base = make_args([])
    assert base.do_lines is True
    assert base.line_count is None  # no cap by default
    assert base.line_min_length == pano.LINE_MIN_FRACTION

    assert make_args(["--no-lines"]).do_lines is False
    assert make_args(["-l"]).line_count is None
    assert make_args(["-l", "3"]).line_count == 3
    assert make_args(["--line", "3"]).line_count == 3
    assert make_args(["--line-min-length", "0.02"]).line_min_length == 0.02
    with pytest.raises(SystemExit):
        make_args(["-l", "0"])


def test_filtered_lines_are_refused():
    with pytest.raises(SystemExit):
        make_args(["-l", "filtered"])


def test_straighten_defaults_off():
    assert make_args([]).straighten is False
    assert make_args(["--straighten"]).straighten is True


def test_clean_distance_defaults_off():
    assert make_args([]).clean_distance is False
    assert make_args(["--clean-distance"]).clean_distance is True


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


# ---------------------------------------------------------------------------
#  The plan
# ---------------------------------------------------------------------------
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


def test_build_detects_lines_without_linefind(tmp_path):
    plan = build(tmp_path)
    assert "Finding vertical lines" in labels(plan)
    assert not any("linefind" in c for c in commands(plan))
    assert "cpclean" in " ".join(c for c in commands(plan) if "cpclean" in c)


def test_build_skips_detection_with_no_lines(tmp_path):
    plan = build(tmp_path, ["--no-lines"])
    assert "Finding vertical lines" not in labels(plan)


def test_build_cleans_before_and_after_the_fit(tmp_path):
    joined = commands(build(tmp_path))
    assert any("--whole-pano-checking" in c and "--max-distance=3" in c for c in joined)
    assert any("--check-line-cp" in c and "--max-distance=3" in c for c in joined)


def test_clean_sigma_reaches_cpclean(tmp_path):
    plan = build(tmp_path, clean_sigma=4.5)
    cpclean = [c for c in commands(plan) if "cpclean" in c]
    assert cpclean
    assert all("--max-distance=4.5" in c for c in cpclean)


def test_no_clean_skips_every_cleaning_step(tmp_path):
    plan = build(tmp_path, ["--clean-distance"], clean=False)
    assert not any("cpclean" in c for c in commands(plan))
    assert "Removing bad matches" not in labels(plan)
    assert "Removing points with error above threshold" not in labels(plan)


def test_clean_distance_adds_the_residual_step(tmp_path):
    assert "Removing points with error above threshold" not in labels(build(tmp_path))
    plan = build(tmp_path, ["--clean-distance"])
    assert "Removing points with error above threshold" in labels(plan)


def test_line_verticality_is_reported(tmp_path):
    assert "Checking line verticality" in labels(build(tmp_path))
    assert "Checking line verticality" not in labels(build(tmp_path, ["--no-lines"]))


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
