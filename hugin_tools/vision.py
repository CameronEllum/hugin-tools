"""Optional review pass: hand a run's own artefacts to a local vision model.

The two workflows already produce small decision previews - Fuse's motion-mask
overlays and Pano's line overlays - plus a finished image.  A local
vision-language model behind an OpenAI-compatible endpoint (LM Studio, Ollama,
llama.cpp, vLLM) can look at those and say, in words, whether the result looks
right; the numeric checks (residuals, line tilts, mask coverage) stay the precise
layer underneath.

The images are shrunk to fit the server's *loaded* context window, which is
usually far smaller than the model's maximum, and the token cost is estimated
before anything is sent.  Pillow is imported only when a review actually runs,
so an ordinary Fuse or Pano run pays nothing.

Review runs only when --review is given, and is then mandatory: a review that
cannot be produced fails the run rather than being silently skipped.  Without it
nothing is sent, so an ordinary Fuse or Pano run pays nothing.
"""

from __future__ import annotations

import base64
import io
import json
import math
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .core import Step

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
DEFAULT_URL = "http://localhost:1234/v1"  # LM Studio; Ollama is http://localhost:11434/v1
DEFAULT_CONTEXT = 8192  # a safe assumption when the server will not say
PATCH = 28  # a Qwen-VL packs an image into roughly 28x28 patches
PROMPT_OVERHEAD = 400  # rough token cost of the instructions
MARGIN = 0.15  # the estimator is approximate, so leave headroom


@dataclass(frozen=True)
class Options:
    """How (and whether) to review, resolved from the command line."""

    mode: bool = False  # False = never, True = required (fail the run if it cannot run)
    url: str = DEFAULT_URL
    model: str | None = None
    context: int | None = None
    detail: str = "balanced"
    timeout: float = 120.0
    max_tokens: int = 1024  # reasoning models need room before they draft the verdict


@dataclass(frozen=True)
class Detail:
    """Image sizes for one --vision-detail setting, in pixels of the long side."""

    stitch: int  # the finished output
    sheet: int  # a detection check sheet
    floor: int  # never shrink below this


DETAIL = {
    "low": Detail(768, 640, 512),
    "balanced": Detail(1280, 1024, 640),
    "high": Detail(1600, 1280, 768),
}


@dataclass
class Artifact:
    """One image for the model, with the question that applies to it."""

    path: Path
    label: str
    question: str | None = None
    sheet: bool = False  # sheets are dropped first when the budget is tight


class ReviewError(RuntimeError):
    """A review could not be produced (no model, unreachable, bad reply)."""


# ---------------------------------------------------------------------------
#  The prompt.  There is deliberately no canned answer to echo: telling a small
#  model the exact words for "all is well" makes it say those words about a
#  plainly broken image.  Ask an open question and let it answer.
# ---------------------------------------------------------------------------
SYSTEM = (
    "You review the output of an automated Hugin panorama / frame-stack tool. "
    "You are shown the tool's own detection check sheets and its final output. "
    "Answer only from what you can see. Be specific and terse; do not restate "
    "these instructions."
)

PANO_STITCH_QUESTION = (
    "The final stitch. Look for duplicated or vertically shifted regions, "
    "ghosting, visible seams, and blown or banded exposure blends."
)

PANO_SHEET_QUESTION = (
    "A line check sheet: red = a vertical line the tool kept, blue = horizontal, "
    "orange = dropped. Check whether any red line sits where there is no real "
    "vertical structure (trees, grass or water)."
)

PANO_CLOSING = (
    "Give one overall 'VERDICT: ok' or 'VERDICT: suspect', then up to three "
    "findings, each naming the file and a concrete hugin_tools option "
    "(-l/--lines, --line-min-length, --clean-sigma, --clean-distance). If you "
    "see nothing wrong, say so plainly."
)

# Each fused output is judged differently, so each carries its own question: an
# action composite is *meant* to show the subject several times, a clean plate is
# meant not to, and a plain frame is meant to be neither.
FUSE_ACTION_QUESTION = (
    "The action composite: the subject appears once per frame on purpose, so the "
    "repetition is correct - do not call it a fault. Judge only alignment, halo or "
    "colour mismatch where the instances meet, and the background behind them."
)

FUSE_CLEAN_QUESTION = (
    "The clean plate: the moving subject should be gone. Check that none of it "
    "remains and that where it was is not smeared or softened."
)

FUSE_FRAME_QUESTION = (
    "A plain fused frame. Check for ghosting along edges, a frame that stayed "
    "soft, and noise or blotches left behind."
)

FUSE_MASK_QUESTION = (
    "A motion check sheet: red marks what the tool treated as moving. Check the "
    "red covers real movement rather than a whole frame, and is not missing."
)

FUSE_CLOSING = (
    "Give one overall 'VERDICT: ok' or 'VERDICT: suspect', then up to three "
    "findings, each naming the file and a concrete hugin_tools option "
    "(--aligner, --mask-z, --mask-dilate, --clean-sigma). If you see nothing "
    "wrong, say so plainly."
)


# ---------------------------------------------------------------------------
#  Sizing: keep the request inside the server's loaded context
# ---------------------------------------------------------------------------
_ORIGINAL_SIZES: dict[str, tuple[int, int]] = {}


def _original_size(path: Path) -> tuple[int, int]:
    key = str(path)
    size = _ORIGINAL_SIZES.get(key)
    if size is None:
        from PIL import Image  # only a review needs Pillow

        with Image.open(path) as opened:
            size = opened.size
        _ORIGINAL_SIZES[key] = size
    return size


def scaled_size(path: Path, long_side: int) -> tuple[int, int]:
    """The size an image will have once shrunk to fit `long_side`."""
    width, height = _original_size(path)
    scale = min(1.0, long_side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def estimate_tokens(path: Path, long_side: int) -> int:
    """Rough vision-token cost of one image at the given long side."""
    width, height = scaled_size(path, long_side)
    return math.ceil(width / PATCH) * math.ceil(height / PATCH) + 4


def plan(
    items: list[Artifact], context: int, detail: Detail, max_tokens: int
) -> tuple[int, int, list[Artifact]]:
    """(stitch side, sheet side, kept artefacts) that fit in the context.

    Shrinks both kinds of image together until the estimate fits, and if that is
    still not enough at the floor, drops check sheets from the end - the finished
    output is the one thing never dropped.
    """
    budget = max(256, int((context - max_tokens - PROMPT_OVERHEAD) * (1.0 - MARGIN)))

    def cost(stitch_side: int, sheet_side: int, seq: list[Artifact]) -> int:
        return sum(estimate_tokens(a.path, sheet_side if a.sheet else stitch_side) for a in seq)

    kept = list(items)
    stitch_side, sheet_side = detail.stitch, detail.sheet
    while cost(stitch_side, sheet_side, kept) > budget and (
        stitch_side > detail.floor or sheet_side > detail.floor
    ):
        stitch_side = max(detail.floor, stitch_side - 64)
        sheet_side = max(detail.floor, sheet_side - 64)
    while len(kept) > 1 and cost(stitch_side, sheet_side, kept) > budget:
        for i in range(len(kept) - 1, -1, -1):
            if kept[i].sheet:  # sheets go first, in reverse
                kept.pop(i)
                break
        else:
            break
    return stitch_side, sheet_side, kept


# ---------------------------------------------------------------------------
#  Talking to the server
# ---------------------------------------------------------------------------
def _get_json(url: str, timeout: float):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _root(url: str) -> str:
    """The server root behind an OpenAI-style /v1 base URL."""
    root = url.rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def _prefer_vision(models: list[dict]) -> list[dict]:
    """Put models whose name looks like a vision model first.

    A local server often has a big general/reasoning model loaded alongside a
    small VLM; choosing blindly can pick the one that will spend its whole reply
    thinking and return nothing.
    """

    def key(entry: dict) -> int:
        name = str(entry.get("id", "")).lower()
        return 0 if any(tag in name for tag in ("vl", "vision", "llava")) else 1

    return sorted(models, key=key)


def discover(
    url: str, model: str | None, context: int | None, timeout: float = 5.0
) -> tuple[str | None, int]:
    """(model id or None, context length), asking the server where it can.

    LM Studio's /api/v0/models reports the *loaded* context length, which is
    what the request has to fit - the model's maximum is often far larger.
    """
    ctx = int(context) if context else DEFAULT_CONTEXT
    try:
        data = _get_json(_root(url) + "/api/v0/models", timeout)
    except Exception:  # noqa: BLE001 - any failure just means "use the fallback"
        data = None
    if data:
        models = data.get("data", [])
        vision = [m for m in models if str(m.get("type", "")).lower() in ("vlm", "vision")]
        loaded = _prefer_vision([m for m in vision if m.get("state") == "loaded"] or vision)
        chosen = next((m for m in vision if model and m.get("id") == model), None)
        if chosen is None:
            chosen = loaded[0] if loaded else None
        if chosen is not None:
            if not context:
                ctx = int(chosen.get("loaded_context_length") or DEFAULT_CONTEXT)
            return chosen.get("id", model), ctx
    if not model:  # an OpenAI-style listing can still name something to try
        try:
            listing = _get_json(url.rstrip("/") + "/models", timeout)
            ids = [m.get("id") for m in listing.get("data", []) if m.get("id")]
            model = ids[0] if ids else None
        except Exception:  # noqa: BLE001
            pass
    return model, ctx


def data_url(path: Path, long_side: int, quality: int = 85) -> str:
    """A JPEG data URL for one image, shrunk to `long_side`."""
    from PIL import Image

    with Image.open(path) as opened:
        picture = opened.convert("RGB")
        if max(picture.size) > long_side:
            picture.thumbnail((long_side, long_side))
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_messages(
    items: list[Artifact], intro: str, closing: str, stitch_side: int, sheet_side: int
) -> list[dict]:
    """Context, then each image labelled with its own question, then the closing."""
    content: list[dict] = [{"type": "text", "text": intro}]
    for artifact in items:
        long_side = sheet_side if artifact.sheet else stitch_side
        content.append({"type": "text", "text": f"[{artifact.label}]"})
        content.append(
            {"type": "image_url", "image_url": {"url": data_url(artifact.path, long_side)}}
        )
        if artifact.question:
            content.append({"type": "text", "text": artifact.question})
    content.append({"type": "text", "text": closing})
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]


def _complete(
    url: str,
    model: str,
    messages: list[dict],
    timeout: float,
    max_tokens: int,
    reasoning_effort: str | None = None,
) -> tuple[str, bool]:
    """One request. Returns (text, reasoning_only). Raises ReviewError on transport."""
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    request = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:  # a real reply, just not the one we want
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise ReviewError(f"the model server replied HTTP {exc.code}: {detail}") from None
    except TimeoutError:
        raise ReviewError(
            f"the model did not answer within {timeout:g}s - raise --vision-timeout"
        ) from None
    except OSError as exc:  # URLError, connection reset, ...
        raise ReviewError(f"cannot reach {url} ({getattr(exc, 'reason', exc)})") from None
    try:
        body = json.loads(raw.decode("utf-8"))
    except ValueError:
        raise ReviewError("the model server returned a non-JSON reply") from None
    try:
        message = body["choices"][0]["message"]
    except KeyError, IndexError, TypeError:
        raise ReviewError("the model server returned an unexpected response") from None
    text = (message.get("content") or "").strip()
    reasoning_only = not text and bool((message.get("reasoning_content") or "").strip())
    return text, reasoning_only


def chat(url: str, model: str, messages: list[dict], timeout: float, max_tokens: int) -> str:
    """A completion from the local server, or ReviewError.

    Thinking is switched off up front: a reasoning model otherwise spends its
    whole reply reasoning and returns no verdict at all.  A server that does not
    accept the parameter is retried without it.
    """
    try:
        text, reasoning_only = _complete(url, model, messages, timeout, max_tokens, "none")
    except ReviewError:
        text, reasoning_only = _complete(url, model, messages, timeout, max_tokens)
    if text:
        return text
    if reasoning_only:
        raise ReviewError(
            "the model used its whole reply for reasoning and returned no verdict"
            " - name a non-reasoning --vision-model, or raise --vision-max-tokens"
        )
    raise ReviewError("the model returned an empty reply")


def run_review(items: list[Artifact], intro: str, closing: str, options: Options) -> str:
    """Size the request to the context, send it, return the verdict text."""
    model, context = discover(options.url, options.model, options.context, options.timeout)
    if not model:
        raise ReviewError(
            f"no vision model found at {options.url} - load one, or pass --vision-model"
        )
    stitch_side, sheet_side, kept = plan(items, context, DETAIL[options.detail], options.max_tokens)
    messages = build_messages(kept, intro, closing, stitch_side, sheet_side)
    return chat(options.url, model, messages, options.timeout, options.max_tokens)


# ---------------------------------------------------------------------------
#  The plan step
# ---------------------------------------------------------------------------
def review_step(
    *,
    artifacts: Callable[[], list[Artifact]],
    intro: Callable[[], str],
    closing: str,
    options: Options | None,
) -> Step | None:
    """A step that reviews the run, or None unless --review was given.

    `artifacts` and `intro` are callables: neither the outputs nor the report they
    quote exist until the earlier steps have run.  A review is only ever run when
    asked for, and then it is mandatory: any failure propagates and fails the run
    rather than being silently skipped.
    """
    if options is None or not options.mode:
        return None

    def run() -> str:
        items = [a for a in artifacts() if a.path.exists()]
        if not items:
            raise ReviewError("no outputs were produced to review")
        verdict = run_review(items, intro(), closing, options)
        if "verdict: suspect" in verdict.lower():
            return "REVIEW FLAGGED PROBLEMS\n" + verdict
        return verdict

    return Step("Reviewing result", run_fn=run)
