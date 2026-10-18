"""The optional review pass: sizing, discovery, the prompt, and the step."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from hugin_tools import cli, fuse, vision


def make_image(path: Path, width: int, height: int) -> Path:
    image = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.new("RGB", (width, height), (30, 40, 50)).save(path)
    return path


# ---------------------------------------------------------------------------
#  Sizing
# ---------------------------------------------------------------------------
def test_scaled_size_fits_the_long_side(tmp_path):
    picture = make_image(tmp_path / "wide.png", 4000, 1000)
    assert vision.scaled_size(picture, 800) == (800, 200)


def test_scaled_size_keeps_small_images(tmp_path):
    picture = make_image(tmp_path / "small.png", 100, 50)
    assert vision.scaled_size(picture, 800) == (100, 50)


def test_estimate_tokens_tracks_area(tmp_path):
    small = make_image(tmp_path / "s.png", 560, 560)
    large = make_image(tmp_path / "l.png", 1120, 1120)
    assert vision.estimate_tokens(large, 2000) == pytest.approx(
        4 * vision.estimate_tokens(small, 2000), rel=0.1
    )


def test_plan_drops_sheets_before_the_output(tmp_path):
    stitch = make_image(tmp_path / "stitch.png", 4000, 2000)
    sheets = [make_image(tmp_path / f"sheet{i}.png", 1200, 1600) for i in range(6)]
    items = [vision.Artifact(stitch, "final stitch")]
    items += [vision.Artifact(s, f"sheet {i}", sheet=True) for i, s in enumerate(sheets)]

    # A tiny context cannot hold everything, so the sheets go first.
    _ss, _hs, kept = vision.plan(
        items, context=2048, detail=vision.DETAIL["balanced"], max_tokens=256
    )
    assert kept[0].path == stitch
    assert all(a.sheet for a in kept[1:])
    assert len(kept) < len(items)


def test_plan_shrinks_before_dropping(tmp_path):
    stitch = make_image(tmp_path / "stitch.png", 7283, 3084)
    item = [vision.Artifact(stitch, "final stitch")]
    stitch_side, _sheet_side, kept = vision.plan(
        item, context=2048, detail=vision.DETAIL["high"], max_tokens=512
    )
    assert kept == item
    assert stitch_side < vision.DETAIL["high"].stitch  # shrank to fit
    assert stitch_side >= vision.DETAIL["high"].floor


def test_plan_never_below_the_floor(tmp_path):
    stitch = make_image(tmp_path / "stitch.png", 8000, 6000)
    item = [vision.Artifact(stitch, "final stitch")]
    stitch_side, _sheet_side, kept = vision.plan(
        item, context=512, detail=vision.DETAIL["balanced"], max_tokens=256
    )
    assert kept == item  # the output is never dropped
    assert stitch_side == vision.DETAIL["balanced"].floor


# ---------------------------------------------------------------------------
#  Discovery
# ---------------------------------------------------------------------------
def lmstudio_payload(context=8192, state="loaded"):
    return {
        "data": [
            {"id": "text-only", "type": "llm", "state": "loaded"},
            {
                "id": "qwen/qwen3-vl-8b",
                "type": "vlm",
                "state": state,
                "loaded_context_length": context,
                "max_context_length": 262144,
            },
        ]
    }


def test_discover_reads_loaded_context(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "_get_json", lambda url, timeout: lmstudio_payload(8192))
    model, context = vision.discover("http://localhost:1234/v1", None, None)
    assert model == "qwen/qwen3-vl-8b"
    assert context == 8192  # the loaded window, not the 262144 maximum


def test_discover_explicit_context_wins(monkeypatch):
    monkeypatch.setattr(vision, "_get_json", lambda url, timeout: lmstudio_payload(8192))
    _model, context = vision.discover("http://localhost:1234/v1", None, 4096)
    assert context == 4096


def test_discover_falls_back_when_unreachable(monkeypatch):
    def boom(url, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(vision, "_get_json", boom)
    model, context = vision.discover("http://localhost:1234/v1", "my-vlm", None)
    assert model == "my-vlm"  # trust the name the user gave
    assert context == vision.DEFAULT_CONTEXT


def two_vlm_payload():
    return {
        "data": [
            {
                "id": "qwen3.6-35b-a3b",
                "type": "vlm",
                "state": "loaded",
                "loaded_context_length": 8192,
            },
            {
                "id": "qwen/qwen3-vl-8b",
                "type": "vlm",
                "state": "loaded",
                "loaded_context_length": 8192,
            },
        ]
    }


def test_discover_prefers_a_vision_named_model(monkeypatch):
    # A big reasoning model loaded next to a small VLM must not win by default.
    monkeypatch.setattr(vision, "_get_json", lambda url, timeout: two_vlm_payload())
    model, _context = vision.discover("http://localhost:1234/v1", None, None)
    assert model == "qwen/qwen3-vl-8b"


def test_discover_named_model_still_wins(monkeypatch):
    monkeypatch.setattr(vision, "_get_json", lambda url, timeout: two_vlm_payload())
    model, _context = vision.discover("http://localhost:1234/v1", "qwen3.6-35b-a3b", None)
    assert model == "qwen3.6-35b-a3b"  # an explicit choice is honoured


def test_discover_openai_listing(monkeypatch):
    def only_models(url, timeout):
        if url.endswith("/api/v0/models"):
            raise OSError("not LM Studio")
        return {"data": [{"id": "llava:latest"}]}

    monkeypatch.setattr(vision, "_get_json", only_models)
    model, context = vision.discover("http://localhost:11434/v1", None, None)
    assert model == "llava:latest"
    assert context == vision.DEFAULT_CONTEXT


# ---------------------------------------------------------------------------
#  The prompt
# ---------------------------------------------------------------------------
ALL_QUESTIONS = (
    vision.PANO_STITCH_QUESTION,
    vision.PANO_SHEET_QUESTION,
    vision.FUSE_ACTION_QUESTION,
    vision.FUSE_CLEAN_QUESTION,
    vision.FUSE_FRAME_QUESTION,
    vision.FUSE_MASK_QUESTION,
    vision.PANO_CLOSING,
    vision.FUSE_CLOSING,
)


def test_questions_never_carry_a_canned_answer():
    # Regression: telling the model the exact words for "all is well" made it
    # say them about a blatantly broken panorama.
    for text in ALL_QUESTIONS:
        assert "no problems found" not in text.lower()
        assert "say exactly" not in text.lower()
    assert "VERDICT: ok" in vision.PANO_CLOSING
    assert "VERDICT: ok" in vision.FUSE_CLOSING


def test_each_fuse_mode_has_its_own_question():
    # An action composite means the subject repeats on purpose; a clean plate
    # means it should not; they cannot share one question.
    questions = {
        vision.FUSE_ACTION_QUESTION,
        vision.FUSE_CLEAN_QUESTION,
        vision.FUSE_FRAME_QUESTION,
    }
    assert len(questions) == 3
    assert "on purpose" in vision.FUSE_ACTION_QUESTION
    assert "should be gone" in vision.FUSE_CLEAN_QUESTION


def test_fuse_artifacts_carry_mode_questions(tmp_path):
    deliver = tmp_path
    work = tmp_path / ".hugin_tools" / "fuse"
    (work / "masks").mkdir(parents=True)
    (deliver / "shot_fuse_action.jpg").write_bytes(b"")
    (deliver / "shot_fuse_winsor.jpg").write_bytes(b"")
    (work / "masks" / "frame0_overlay.jpg").write_bytes(b"")

    items = fuse.review_artifacts(deliver, work, "shot_fuse")
    by_name = {a.path.name: a for a in items}
    assert items[0].path.name == "shot_fuse_action.jpg"  # motion output judged first
    assert by_name["shot_fuse_action.jpg"].question == vision.FUSE_ACTION_QUESTION
    assert by_name["shot_fuse_winsor.jpg"].question == vision.FUSE_FRAME_QUESTION
    assert by_name["frame0_overlay.jpg"].question == vision.FUSE_MASK_QUESTION
    assert by_name["frame0_overlay.jpg"].sheet is True


def test_build_messages_labels_and_questions_every_image(tmp_path):
    stitch = make_image(tmp_path / "stitch.png", 800, 400)
    sheet = make_image(tmp_path / "sheet.png", 400, 800)
    items = [
        vision.Artifact(stitch, "final stitch: stitch.png", question="judge the seam"),
        vision.Artifact(sheet, "line check sheet: sheet.png", question="judge the red", sheet=True),
    ]
    messages = vision.build_messages(items, "intro text", vision.PANO_CLOSING, 800, 400)
    assert messages[0]["role"] == "system"
    content = messages[1]["content"]
    texts = [part["text"] for part in content if part["type"] == "text"]
    images = [part for part in content if part["type"] == "image_url"]
    assert "intro text" in texts
    assert "final stitch: stitch.png" in texts
    assert "judge the seam" in texts  # each image keeps its own question
    assert "judge the red" in texts
    assert texts[-1] == vision.PANO_CLOSING
    assert len(images) == 2
    assert all(img["image_url"]["url"].startswith("data:image/jpeg;base64,") for img in images)


def test_chat_reports_http_error(monkeypatch):
    def http_error(url, timeout):
        raise urllib.error.HTTPError(url, 400, "Bad Request", {}, None)

    monkeypatch.setattr(vision.urllib.request, "urlopen", http_error)
    with pytest.raises(vision.ReviewError):
        vision.chat("http://x/v1", "m", [], 5.0, 100)


def test_chat_rejects_a_reasoning_only_reply(monkeypatch):
    # Regression: a reasoning model spent its whole budget thinking, returned an
    # empty content, and the run said nothing at all about the review.
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            body = {"choices": [{"message": {"content": "", "reasoning_content": "hmm"}}]}
            return json.dumps(body).encode("utf-8")

    monkeypatch.setattr(vision.urllib.request, "urlopen", lambda url, timeout: FakeResponse())
    with pytest.raises(vision.ReviewError) as exc:
        vision.chat("http://x/v1", "m", [], 5.0, 10)
    assert "reasoning" in str(exc.value)


def test_chat_asks_for_no_thinking_first(monkeypatch):
    # Thinking is switched off up front so a reasoning model does not burn the
    # whole reply (and the timeout) before producing a verdict.
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "VERDICT: ok"}}]}).encode()

    def fake_urlopen(request, timeout):
        calls.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr(vision.urllib.request, "urlopen", fake_urlopen)
    assert vision.chat("http://x/v1", "m", [], 5.0, 100) == "VERDICT: ok"
    assert len(calls) == 1
    assert calls[0]["reasoning_effort"] == "none"


def test_chat_falls_back_when_the_parameter_is_rejected(monkeypatch):
    # A server that does not know reasoning_effort must still get a review.
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "VERDICT: ok"}}]}).encode()

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        calls.append(payload)
        if "reasoning_effort" in payload:
            raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, None)
        return FakeResponse()

    monkeypatch.setattr(vision.urllib.request, "urlopen", fake_urlopen)
    assert vision.chat("http://x/v1", "m", [], 5.0, 100) == "VERDICT: ok"
    assert len(calls) == 2
    assert "reasoning_effort" not in calls[1]


def test_chat_turns_a_timeout_into_a_review_error(monkeypatch):
    def slow(url, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(vision.urllib.request, "urlopen", slow)
    with pytest.raises(vision.ReviewError) as exc:
        vision.chat("http://x/v1", "m", [], 5.0, 100)
    assert "did not answer" in str(exc.value)


# ---------------------------------------------------------------------------
#  The step: best-effort vs mandatory
# ---------------------------------------------------------------------------
def a_step(mode, tmp_path, monkeypatch):
    stitch = make_image(tmp_path / "out.png", 400, 300)
    options = vision.Options(mode=mode)
    step = vision.review_step(
        artifacts=lambda: [vision.Artifact(stitch, "out")],
        intro=lambda: "intro",
        closing=vision.PANO_CLOSING,
        options=options,
    )
    return step


def test_review_step_absent_when_disabled(tmp_path):
    assert (
        vision.review_step(
            artifacts=lambda: [],
            intro=lambda: "",
            closing="q",
            options=vision.Options(mode=False),
        )
        is None
    )
    assert (
        vision.review_step(artifacts=lambda: [], intro=lambda: "", closing="q", options=None)
        is None
    )


def test_mandatory_review_still_fails_on_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vision, "run_review", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timed out"))
    )
    step = a_step(True, tmp_path, monkeypatch)
    with pytest.raises(TimeoutError):
        step.run_fn()


def test_mandatory_review_raises_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vision, "run_review", lambda *a, **k: (_ for _ in ()).throw(vision.ReviewError("no model"))
    )
    step = a_step(True, tmp_path, monkeypatch)
    assert step is not None
    with pytest.raises(vision.ReviewError):
        step.run_fn()


def test_review_step_returns_the_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "run_review", lambda *a, **k: "VERDICT: ok")
    step = a_step(True, tmp_path, monkeypatch)
    assert step.run_fn() == "VERDICT: ok"


def test_suspect_verdict_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "run_review", lambda *a, **k: "VERDICT: suspect\nbad seam")
    step = a_step(True, tmp_path, monkeypatch)
    assert step.run_fn().startswith("REVIEW FLAGGED PROBLEMS")


# ---------------------------------------------------------------------------
#  Command line
# ---------------------------------------------------------------------------
def parse(argv):
    return cli._parser().parse_args(argv)


def test_review_is_off_by_default():
    args = parse(["pano"])
    assert args.review is False  # no review unless --review is given
    assert args.vision_model is None
    assert args.vision_url == vision.DEFAULT_URL
    assert args.vision_detail == "balanced"


def test_review_is_opt_in():
    assert parse(["pano", "--review"]).review is True
    assert parse(["pano", "--no-review"]).review is False
    assert parse(["fuse", "--review"]).review is True
    assert parse(["fuse", "--no-review"]).review is False
