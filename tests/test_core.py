"""Shared-core behaviour: frame collection, pathlib helpers, scratch, run_plan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from hugin_tools import core


def test_collect_frames_folder(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"x")
    (tmp_path / "note.txt").write_text("n")
    frames, skipped, unmatched = core.collect_frames([str(tmp_path)], exts={".jpg"})
    assert [p.name for p in frames] == ["a.jpg", "b.jpg"]
    assert skipped == []
    assert unmatched == []


def test_collect_frames_glob(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    frames, _, _ = core.collect_frames([str(tmp_path / "*.jpg")], exts={".jpg"})
    assert [p.name for p in frames] == ["a.jpg"]


def test_collect_frames_unmatched(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "a.jpg").write_bytes(b"x")
    frames, _, unmatched = core.collect_frames(
        [str(frames_dir), str(tmp_path / "nope.jpg")], exts={".jpg"}
    )
    assert [p.name for p in frames] == ["a.jpg"]
    assert unmatched == [str(tmp_path / "nope.jpg")]


def test_collect_frames_min_frames(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    with pytest.raises(SystemExit):
        core.collect_frames([str(tmp_path)], exts={".jpg"}, min_frames=2)


def test_collect_frames_anchor_middle(tmp_path):
    for name in "abcdef":
        (tmp_path / f"{name}.jpg").write_bytes(b"x")
    frames, _, _ = core.collect_frames(
        [str(tmp_path)], exts={".jpg"}, min_frames=2, anchor_middle=True
    )
    assert frames[0].name == "d.jpg"


def test_collect_frames_skips_own_outputs(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "a_pano.jpg").write_bytes(b"x")

    def is_out(p: Path) -> bool:
        return p.stem.endswith("_pano")

    frames, skipped, _ = core.collect_frames([str(tmp_path)], exts={".jpg"}, is_output=is_out)
    assert [p.name for p in frames] == ["a.jpg"]
    assert [p.name for p in skipped] == ["a_pano.jpg"]


def test_glob_paths(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    assert core.glob_paths(str(tmp_path / "*.jpg")) == [tmp_path / "a.jpg"]


def test_glob_paths_no_match(tmp_path):
    assert core.glob_paths(str(tmp_path / "*.jpg")) == []


def test_relpath(tmp_path):
    a = tmp_path / "x" / "y"
    b = tmp_path / "x"
    assert core.relpath(a, b) == Path("y")
    assert core.relpath(b, a) == Path("..")


def test_remove_tree(tmp_path):
    d = tmp_path / "d"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "f").write_text("x")
    (d / "top").write_text("y")
    core.remove_tree(d)
    assert not d.exists()


def test_remove_tree_missing_is_noop(tmp_path):
    core.remove_tree(tmp_path / "nope")  # must not raise


def test_sweep_scratch_keeps_named(tmp_path):
    s = tmp_path / core.SCRATCH_NAME / "fuse"
    (s / "aligned").mkdir(parents=True)
    (s / "aligned" / "a.tif").write_text("x")
    (s / "merged").mkdir()
    (s / "merged" / "m.tif").write_text("x")
    core.sweep_scratch(s, keep_all=False, keep_names={"aligned"})
    assert (s / "aligned").exists()
    assert not (s / "merged").exists()


def test_sweep_scratch_removes_empty_scratch(tmp_path):
    s = tmp_path / core.SCRATCH_NAME / "fuse"
    (s / "merged").mkdir(parents=True)
    core.sweep_scratch(s, keep_all=False, keep_names=set())
    assert not s.exists()
    # Removing the now-empty hidden root is run_cli's job, not sweep_scratch's.
    assert (tmp_path / core.SCRATCH_NAME).exists()


def test_sweep_scratch_keep_all(tmp_path):
    s = tmp_path / "s"
    (s / "aligned").mkdir(parents=True)
    core.sweep_scratch(s, keep_all=True, keep_names=set())
    assert s.exists()


def test_scratch_dir_default_and_override(tmp_path):
    assert core.scratch_dir(tmp_path, "fuse") == tmp_path / core.SCRATCH_NAME / "fuse"
    assert core.scratch_dir(tmp_path, "pano", "D:/work") == Path("D:/work/pano")


def test_toggle():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--lines",
        "--no-lines",
        dest="do_lines",
        action=core.Toggle,
        off=["--no-lines"],
        default=None,
    )
    assert p.parse_args([]).do_lines is None
    assert p.parse_args(["--lines"]).do_lines is True
    assert p.parse_args(["--no-lines"]).do_lines is False


def test_fmt_duration():
    assert core.fmt_duration(12) == "12s"
    assert core.fmt_duration(83) == "1m 23s"
    assert core.fmt_duration(7440) == "2h 04m"


def test_display_dir(tmp_path):
    assert core.display_dir(Path(".")) == "/"
    assert core.display_dir(Path("out")) == "/out"
    # An absolute path is shown verbatim.
    assert core.display_dir(tmp_path) == tmp_path.as_posix()


def test_run_plan_runs_commands(tmp_path):
    ui = core.UI()
    marker = tmp_path / "out.txt"
    code = f"open(r'{marker}', 'w').write('hi')"
    plan = {"main": [core.Step("Doing a thing", cmd=[sys.executable, "-c", code])]}
    assert core.run_plan(ui, plan) == 0
    assert marker.read_text() == "hi"


def test_run_plan_run_fn_and_on_done():
    ui = core.UI()
    calls: list[str] = []
    plan = {
        "main": [
            core.Step("In process", run_fn=lambda: (calls.append("fn"), "report")[1]),
            core.Step("Note only", note="hidden", quiet=True),
            core.Step("Action", on_done=lambda: (calls.append("done"), "did it")[1]),
        ]
    }
    assert core.run_plan(ui, plan) == 0
    assert calls == ["fn", "done"]


def test_run_plan_failure_counts():
    ui = core.UI()
    plan = {"main": [core.Step("Failing", cmd=[sys.executable, "-c", "raise SystemExit(3)"])]}
    assert core.run_plan(ui, plan) == 1


def test_run_plan_action_only_error():
    ui = core.UI()

    def boom() -> str:
        raise RuntimeError("nope")

    plan = {"main": [core.Step("Bad action", on_done=boom)]}
    assert core.run_plan(ui, plan) == 1
