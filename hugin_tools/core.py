"""Shared plumbing for the hugin-tools bundle.

Both tools (Fuse and Pano) are thin front ends over the Hugin command-line
tools.  Everything they have in common lives here: the console/UI, external
tool discovery, frame collection, the ordered-step plan and its executor, the
scratch directory, and a few pathlib helpers.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

try:
    from rich.console import Console
    from rich.markup import escape
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich_argparse import RawDescriptionRichHelpFormatter

    HAVE_RICH = True
    _HELP_FORMATTER = RawDescriptionRichHelpFormatter
except ImportError:  # keep working (plainer) if rich is somehow unavailable
    HAVE_RICH = False
    _HELP_FORMATTER = argparse.RawDescriptionHelpFormatter

    def escape(text: str) -> str:  # type: ignore[misc]
        return text


# ---------------------------------------------------------------------------
#  Console setup
# ---------------------------------------------------------------------------
def setup_console() -> None:
    """Force UTF-8 output (so the marks below render) and enable ANSI on Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError, ValueError:
            pass

    if os.name == "nt":
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_MARKUP = re.compile(r"\[/?[a-zA-Z0-9 #=/_.,'-]+\]")


def _plain(text: str) -> str:
    """Strip rich markup and ANSI codes (for the plain-text log)."""
    return _MARKUP.sub("", _ANSI.sub("", text))


# ---------------------------------------------------------------------------
#  A small presenter: rich when available, plain text otherwise.
# ---------------------------------------------------------------------------
class UI:
    """All user-facing output goes through here, so it stays consistent."""

    def __init__(self, verbose: bool = False, no_color: bool = False, log_path: str | None = None):
        self.verbose = verbose
        self._fh = None
        if log_path:
            try:
                self._fh = open(log_path, "w", encoding="utf-8")
            except OSError as exc:
                print(f"Warning: cannot write log {log_path}: {exc}", file=sys.stderr)
        self._plain: list[str] = []
        self.console = (
            Console(no_color=no_color, highlight=False, emoji=True) if HAVE_RICH else None
        )

    def _log(self, text: str) -> None:
        if self._fh:
            self._fh.write(_plain(text) + "\n")
            self._fh.flush()

    def say(self, markup: str, console=None) -> None:
        """Print a line, optionally through a live progress console."""
        if console is not None:
            console.print(markup)
        elif self.console is not None:
            self.console.print(markup)
        else:
            print(_plain(markup), flush=True)
        self._log(markup)

    def print(self, markup: str = "") -> None:
        self.say(markup)

    def dim(self, text: str, console=None) -> None:
        """Print tool output / command lines, de-emphasised."""
        for line in text.splitlines() or [""]:
            if console is not None or self.console is not None:
                (console or self.console).print(f"[grey50]    {escape(line)}[/grey50]")
            else:
                print(f"    {line}", flush=True)
        self._log("\n".join("    " + ln for ln in (text.splitlines() or [""])))

    def cmd(self, argv: list[str], console=None) -> None:
        self.say(f"[grey50]    $ {escape(' '.join(argv))}[/grey50]", console)

    def header(
        self,
        title: str,
        count: int,
        source: Path | str,
        frames: list[Path],
        extra: list[str] | None = None,
        anchor: Path | None = None,
    ) -> None:
        """Program name, inputs, frame names (sorted, anchor marked), then a
        Progress: heading."""
        noun = "frame" if count == 1 else "frames"
        self.print(f"📷 [bold]{title}[/bold]  ·  {count} {noun}")
        self.print("")
        self.print(f"📁 [grey50]{escape(str(source))}[/grey50]")
        for p in sorted(frames, key=lambda p: p.name):
            mark = " (anchor)" if anchor is not None and p == anchor else ""
            self.print(f"  📄 [grey50]{escape(p.name)}{mark}[/grey50]")
        for line in extra or []:
            self.print(line)
        self.print("")
        self.print("▶️ [bold]Progress:[/bold]")

    def summary(self, results: list[tuple[str, Path]]) -> None:
        """An unboxed, grouped listing - no table borders, just aligned text."""
        if not results:
            return
        self.print("")
        self.print("📦 [bold]Outputs[/bold]")
        groups: dict[str, list[Path]] = {}
        for mode, path in results:
            groups.setdefault(mode, []).append(path)
        for mode, paths in groups.items():
            self.print("")
            self.print(f"📂 [bold cyan]{mode.upper()}[/bold cyan]")
            width = max(len(p.name) for p in paths)
            for p in paths:
                try:
                    size = f"{p.stat().st_size / 1_048_576:.1f} MB"
                except OSError:
                    size = ""
                pad = " " * (width - len(p.name))
                self.say(f"    📄 {escape(p.name)}{pad}  [grey50]{size:>8}[/grey50]")
        self.print("")

    def close(self) -> None:
        if self._fh:
            self._fh.write("\n".join(self._plain) + "\n")
            self._fh.close()
            self._fh = None


# ---------------------------------------------------------------------------
#  Tool discovery
# ---------------------------------------------------------------------------
def find_tool(name: str) -> Path | None:
    """Return the full path to a program, or None if not found."""
    exe = name + (".exe" if os.name == "nt" else "")
    found = shutil.which(name)
    if found:
        return Path(found)

    extra_dirs: list[Path] = []
    if os.name == "nt":
        pf = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        extra_dirs += [pf / "Hugin" / "bin", Path(r"C:\Program Files\Hugin\bin")]
        extra_dirs += list(pf.glob("ImageMagick-*"))
    else:
        extra_dirs += [
            Path(d) for d in ("/usr/bin", "/usr/local/bin", "/snap/bin", "/opt/homebrew/bin")
        ]

    for d in extra_dirs:
        cand = d / exe
        if cand.exists():
            return cand
    return None


def resolve_tools(required: Iterable[str], optional: Iterable[str] = ()) -> dict[str, Path | None]:
    """Find every external program we need. Raise if a required one is missing."""
    required = list(required)
    tools = {name: find_tool(name) for name in (*required, *optional)}

    missing = sorted(n for n in required if not tools.get(n))
    if missing:
        raise SystemExit(
            "Missing required tool(s): "
            + ", ".join(missing)
            + "\nInstall Hugin (and its command-line tools) and make sure they are on PATH."
        )
    return tools


# ---------------------------------------------------------------------------
#  Source frames
# ---------------------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}


def is_image(path: Path, exts: set[str] = IMAGE_EXTS) -> bool:
    return path.is_file() and path.suffix.lower() in exts


def has_magic(text: str) -> bool:
    """True if a shell pattern contains wildcard characters."""
    return any(ch in text for ch in "*?[")


def glob_paths(pattern: str) -> list[Path]:
    """Expand a glob pattern to paths, using Path.glob (recursive, ** aware).

    cmd.exe hands wildcards through unexpanded, so the pattern arrives intact and
    is expanded here; the longest magic-free prefix becomes the glob anchor.
    """
    path = Path(pattern)
    anchor_parts: list[str] = []
    for part in path.parts:
        if has_magic(part):
            break
        anchor_parts.append(part)
    if len(anchor_parts) == len(path.parts):  # no magic at all
        return [path] if path.exists() else []
    anchor = Path(*anchor_parts) if anchor_parts else Path(".")
    relative = str(Path(*path.parts[len(anchor_parts) :]))
    return sorted(anchor.glob(relative))


def collect_frames(
    specs: list[str],
    *,
    exts: set[str] = IMAGE_EXTS,
    min_frames: int = 1,
    anchor_middle: bool = False,
    is_output: Callable[[Path], bool] | None = None,
    noun: str = "frames",
) -> tuple[list[Path], list[Path], list[str]]:
    """Frames to process, from folders, files or glob patterns.

    Whatever the shell hands over: a folder is read as before, a pattern the
    shell did not expand is expanded here (cmd.exe leaves wildcards alone, unlike
    PowerShell), and explicit files are taken as given.  Sorted by name; when
    anchor_middle is set the middle frame is rotated to the front (Pano's anchor).

    Returns (frames, skipped, unmatched) - skipped are the tool's own previous
    outputs, unmatched are names that matched nothing or are not images, so the
    caller can say so rather than quietly use fewer frames.
    """
    frames: list[Path] = []
    skipped: list[Path] = []
    unmatched: list[str] = []
    for spec in specs:
        text = str(spec)
        path = Path(text)
        if path.is_dir():
            found = [p for p in sorted(path.iterdir(), key=lambda p: p.name) if is_image(p, exts)]
        elif has_magic(text):
            found = [p for p in glob_paths(text) if is_image(p, exts)]
        elif is_image(path, exts):
            found = [path]
        else:
            found = []
        if not found:
            unmatched.append(text)
        for p in found:
            (skipped if (is_output and is_output(p)) else frames).append(p.resolve())

    frames = sorted(dict.fromkeys(frames), key=lambda p: p.name)
    if not frames:
        raise SystemExit(
            "No source frames found in: "
            + ", ".join(str(s) for s in specs)
            + f"\nName image files, a glob pattern, or a folder holding them"
            f" ({', '.join(sorted(exts))})."
        )
    if len(frames) < min_frames:
        raise SystemExit(
            f"Need at least {min_frames} {noun}, found {len(frames)} in: "
            + ", ".join(str(s) for s in specs)
        )
    if anchor_middle:
        mid = len(frames) // 2
        frames = [frames[mid]] + frames[:mid] + frames[mid + 1 :]
    return frames, skipped, unmatched


# ---------------------------------------------------------------------------
#  The plan: an ordered list of steps, built up-front so the progress bar can
#  show real progress, then executed one at a time.
# ---------------------------------------------------------------------------
@dataclass
class Step:
    label: str
    cmd: list | None = None
    cmd_fn: Callable[[], list] | None = None  # built when the step runs
    run_fn: Callable[[], str | None] | None = None  # in-process Python step
    note: str | None = None  # informational step: no command, just a message
    on_done: Callable[[], str | None] | None = None  # may return a message to print
    outputs: list[Path] = field(default_factory=list)  # files this step writes
    quiet: bool = False  # run silently: no transcript line unless it fails


# A leading icon for each primary task's line (first prefix match wins, so the
# more specific labels must come first - "Skipping alignment" before "Skipping").
STEP_ICONS: tuple[tuple[str, str], ...] = (
    ("Reading frames", "📖"),
    ("Finding matching features", "🔍"),
    ("Finding vertical lines", "📏"),
    ("Setting anchor", "⚓"),
    ("Removing bad matches", "🧹"),
    ("Removing points", "🧹"),
    ("Cleaning control points", "🧹"),
    ("Checking line verticality", "📏"),
    ("Choosing alignment parameters", "🧭"),
    ("Optimising alignment", "🎯"),
    ("Optimising positions", "🎯"),
    ("Refining matches", "🎯"),
    ("Matching exposure", "🌗"),
    ("Preparing output", "🧭"),
    ("Preparing aligned frames", "🧭"),
    ("Drawing line overlays", "🖍️"),
    ("Rendering aligned frames", "🖼️"),
    ("Aligning frames", "🎯"),
    ("Skipping alignment", "⏩"),
    ("Combining frames", "🧩"),
    ("Detecting motion", "🏃"),
    ("Sharpening", "✨"),
    ("Building animation", "🎞️"),
    ("Stitching", "🧵"),
    ("Calculating output size", "📐"),
    ("Centering", "🧭"),
    ("Straightening", "↔️"),
    ("Skipping", "⏩"),
    ("Reviewing result", "🔎"),
    ("Checking", "🔍"),
    ("ImageMagick not found", "⚠️"),
    ("Copying metadata", "🏷️"),
    ("Removing aligned frames", "🧹"),
    ("Merged TIFF", "📦"),
)


def step_icon(label: str) -> str:
    for prefix, icon in STEP_ICONS:
        if label.startswith(prefix):
            return icon
    return "•"


def _failure_tail(lines: list[str], keep: int = 6) -> str:
    """The last few lines a tool printed, for a failure message.

    A bare exit code says nothing about why a command failed, and these tools
    put the reason on their last lines.
    """
    tail = [line.strip() for line in lines if line.strip()][-keep:]
    if not tail:
        return ""
    return "\n" + "\n".join(f"    {line}" for line in tail)


def execute_step(ui: UI, step: Step, console=None) -> tuple[float, str | None]:
    """Run one step's command or in-process callable.

    Returns (elapsed seconds, optional report line).  Raises on failure.
    """
    start = time.time()
    if step.run_fn is not None:  # in-process step: no subprocess to supervise
        report = step.run_fn()
        return time.time() - start, report

    argv = step.cmd_fn() if step.cmd_fn is not None else step.cmd
    # Paths travel as Paths everywhere else; stringify only at this boundary.
    argv = [str(a) for a in argv]
    if ui.verbose:
        ui.cmd(argv, console)

    buf: list[str] = []
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip()
            buf.append(line)
            if ui.verbose:
                ui.dim(line, console)
        if proc.wait() != 0:
            raise RuntimeError(f"{step.label} failed (exit {proc.returncode}){_failure_tail(buf)}")
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        raise
    finally:
        if proc.stdout:
            proc.stdout.close()
    return time.time() - start, None


def run_plan(ui: UI, plan: dict[str, list[Step]]) -> int:
    """Execute every step. Returns the number of engines that failed.

    The progress bar is transient: it shows live progress while steps run and
    disappears when done, leaving the plain line-by-line transcript behind.
    """
    total = sum(len(steps) for steps in plan.values())
    progress = None
    if ui.console is not None:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=ui.console,
            transient=True,
        )

    failures = 0
    task = None

    def run_steps(console) -> None:
        nonlocal failures
        for mode, steps in plan.items():
            err = None
            done = 0
            for step in steps:
                if progress is not None:
                    progress.update(task, description=step.label)
                action_only = step.cmd is None and step.cmd_fn is None and step.run_fn is None
                if action_only:
                    try:
                        msg = step.on_done() if step.on_done else None
                    except Exception as exc:  # noqa: BLE001 - report and continue
                        err = str(exc)
                        break
                    if not step.quiet:
                        tag = "".join(f" [yellow]{escape(f'[{p.name}]')}[/]" for p in step.outputs)
                        ui.say(
                            f"  {step_icon(step.label)}"
                            f" [grey50]{escape(msg or step.note or step.label)}{tag}[/grey50]",
                            console,
                        )
                    if progress is not None:
                        progress.advance(task)
                    continue
                try:
                    took, report = execute_step(ui, step, console)
                except Exception as exc:  # noqa: BLE001 - report and continue
                    err = str(exc)
                    break
                tag = "".join(f" [yellow]{escape(f'[{p.name}]')}[/]" for p in step.outputs)
                ui.say(
                    f"  {step_icon(step.label)} {escape(step.label)}"
                    f" [grey50]({took:.1f}s){tag}[/grey50]",
                    console,
                )
                if report:
                    ui.dim(report, console)
                if step.on_done:
                    extra = step.on_done()
                    if extra:
                        ui.say(
                            f"  {step_icon(step.label)} [grey50]{escape(extra)}[/grey50]", console
                        )
                if progress is not None:
                    progress.advance(task)
                    done += 1
            if err is not None:
                failures += 1
                if progress is not None:
                    progress.advance(task, len(steps) - done - 1)
                ui.say(f"  ❌ {escape(err)} - skipping the rest of {mode}", console)

    if progress is not None:
        with progress:
            task = progress.add_task("Starting", total=total)
            run_steps(progress.console)
    else:
        run_steps(None)
    return failures


# ---------------------------------------------------------------------------
#  Options
# ---------------------------------------------------------------------------
class Toggle(argparse.Action):
    """A boolean flag with explicit on/off option names, shown on one help line.

    Example: Toggle("--lines", "--no-lines") sets the value True for --lines and
    False for --no-lines.
    """

    def __init__(self, option_strings, dest, off=(), default=None, help=None, **kwargs):
        self._off = set(off)
        super().__init__(option_strings, dest, nargs=0, default=default, help=help, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, option_string not in self._off)


def positive_float(text: str) -> float:
    """Argparse type: a float greater than zero (so argparse reports it nicely)."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def fmt_duration(seconds: float) -> str:
    """Format seconds as 12s, 1m 23s or 2h 04m."""
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def display_dir(path: Path) -> str:
    """Render an output folder so it reads as a directory: absolute paths as-is,
    relative ones prefixed with '/' ('.' becomes '/')."""
    s = path.as_posix()
    if path.is_absolute():
        return s
    if s.startswith("./"):
        s = s[2:]
    return "/" if s == "." else f"/{s}"


# ---------------------------------------------------------------------------
#  pathlib helpers (no os.path for path arithmetic)
# ---------------------------------------------------------------------------
def relpath(path: Path | str, start: Path | str) -> Path:
    """A pathlib replacement for os.path.relpath, returning a Path.

    Raises ValueError when the two share no root (different drives on Windows),
    so callers can fall back to the absolute path.
    """
    path = Path(path)
    start = Path(start)
    if not path.is_absolute():
        path = path.absolute()
    if not start.is_absolute():
        start = start.absolute()
    for depth, base in enumerate((start, *start.parents)):
        try:
            rel = path.relative_to(base)
        except ValueError:
            continue
        return rel if depth == 0 else Path(*([".."] * depth)) / rel
    raise ValueError(f"{path} has no relative form from {start}")


def copy_file(src: Path | str, dst: Path | str) -> None:
    """Copy a (small text) file, creating the destination folder if needed."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(Path(src).read_bytes())


def remove_tree(path: Path | str) -> None:
    """Remove a file or directory tree (no shutil.rmtree in pathlib)."""
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        for child in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if child.is_dir() and not child.is_symlink():
                child.rmdir()
            else:
                child.unlink(missing_ok=True)
        path.rmdir()
    else:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
#  Scratch directory: every intermediate lives here and is swept unless kept
# ---------------------------------------------------------------------------
SCRATCH_NAME = ".hugin_tools"


def scratch_dir(outdir: Path, engine: str, override: str | None = None) -> Path:
    """The working folder for one engine: <outdir>/.hugin_tools/<engine>."""
    root = Path(override) if override else Path(outdir) / SCRATCH_NAME
    return root / engine


def sweep_scratch(scratch: Path, keep_all: bool, keep_names: set[str]) -> None:
    """Delete a scratch tree, keeping only the named subfolders when asked."""
    if not scratch.exists():
        return
    if not keep_all:
        for child in list(scratch.iterdir()):
            if child.is_dir() and child.name in keep_names:
                continue
            remove_tree(child)
    if scratch.is_dir() and not any(scratch.iterdir()):
        scratch.rmdir()


@dataclass
class RunPlan:
    """A tool's steps, plus where its scratch and deliverables live."""

    plan: dict[str, list[Step]]
    scratch: dict[str, Path]
    output_dir: dict[str, Path]


# ---------------------------------------------------------------------------
#  The shared run: collect frames, build, execute, report, sweep
# ---------------------------------------------------------------------------
def run_cli(
    ui: UI,
    *,
    title: str,
    tools: dict,
    outdir: Path,
    specs: list[str],
    exts: set[str],
    min_frames: int,
    anchor_middle: bool,
    is_output: Callable[[Path], bool] | None,
    output_label: str,
    noun: str,
    do_chdir: bool,
    mark_anchor: bool,
    size_check: Callable[[list[Path]], None] | None,
    frames_extra: Callable[[list[Path]], list[str]] | None,
    build: Callable[[dict, list[Path], Path], RunPlan],
    collect: Callable[[RunPlan], list[tuple[str, Path]]],
    scratch_paths: list[Path],
    keep_all: bool,
    keep_names: set[str],
) -> int:
    """The common main(): parse-free, so both tools share one implementation."""
    runplan: RunPlan | None = None
    run_start = time.time()
    try:
        frames, skipped, unmatched = collect_frames(
            specs,
            exts=exts,
            min_frames=min_frames,
            anchor_middle=anchor_middle,
            is_output=is_output,
            noun=noun,
        )

        # Run from inside the frames' folder so bare image names in project files
        # always resolve, wherever we were invoked from.
        if do_chdir:
            parents = {p.parent for p in frames}
            if len(parents) == 1:
                os.chdir(parents.pop())

        extra: list[str] = []
        if skipped:
            names = ", ".join(p.name for p in skipped[:5])
            more = f" (+{len(skipped) - 5} more)" if len(skipped) > 5 else ""
            extra.append(
                f"[yellow]Ignored {len(skipped)} previous {output_label} output(s): {names}"
                f"{more}[/yellow]"
            )
        if unmatched:
            names = ", ".join(unmatched[:5])
            more = f" (+{len(unmatched) - 5} more)" if len(unmatched) > 5 else ""
            extra.append(f"[yellow]No image matched: {names}{more}[/yellow]")
        if size_check is not None:
            size_check(frames)
        if frames_extra is not None:
            extra.extend(frames_extra(frames))

        where = str(specs[0]) if len(specs) == 1 else " ".join(str(s) for s in specs)
        ui.header(
            title, len(frames), where, frames, extra, anchor=frames[0] if mark_anchor else None
        )

        runplan = build(tools, frames, outdir)
        failures = run_plan(ui, runplan.plan)

        results = collect(runplan)
        took = fmt_duration(time.time() - run_start)
        ui.print("")
        ui.summary(results)
        if failures:
            ui.print(f"[red]❌ Finished with {failures} failure(s) in {took}.[/red]")
            return 1
        ui.print(f"🎉 [green]Done in {took}.[/green] Output in: {display_dir(outdir)}")
        return 0
    except KeyboardInterrupt:
        ui.print("")
        ui.print("⏹️ [red]Interrupted.[/red]")
        return 130
    finally:
        for scratch in scratch_paths:
            sweep_scratch(scratch, keep_all, keep_names)
        for root in {p.parent for p in scratch_paths}:
            if root.is_dir() and not any(root.iterdir()):
                root.rmdir()
        ui.close()
