"""The shared command line: one entry point, two subcommands, shared options."""

from __future__ import annotations

import pytest

from hugin_tools import cli


def parse(argv):
    return cli._parser().parse_args(argv)


def test_cleaning_is_shared_and_on_by_default():
    assert parse(["pano"]).clean is True
    assert parse(["fuse"]).clean is True
    assert parse(["fuse", "--no-clean"]).clean is False
    assert parse(["pano", "--no-clean"]).clean is False


def test_clean_sigma_is_shared():
    assert parse(["pano"]).clean_sigma == 3.0
    assert parse(["fuse"]).clean_sigma == 3.0
    assert parse(["fuse", "--clean-sigma", "1.5"]).clean_sigma == 1.5
    assert parse(["pano", "--clean-sigma", "4"]).clean_sigma == 4.0


def test_clean_sigma_must_be_positive():
    with pytest.raises(SystemExit):
        parse(["pano", "--clean-sigma", "0"])
    with pytest.raises(SystemExit):
        parse(["pano", "--clean-sigma", "nope"])


def test_fuse_motion_outputs_are_named_for_motion():
    args = parse(["fuse", "--motion-clean"])
    assert args.do_clean is True
    assert parse(["fuse", "--motion-action"]).do_action is True
    assert parse(["fuse"]).do_clean is None  # decides later, in prepare_args


def test_no_subcommand_is_an_error():
    with pytest.raises(SystemExit):
        parse([])
