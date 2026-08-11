"""AgenticSpliceAI Lab entry point and process control.

Usage:
    conda run -n agentic-spliceai python -m server.bio.app            # start (foreground)
    conda run -n agentic-spliceai python -m server.bio.app restart    # stop the old one, start
    conda run -n agentic-spliceai python -m server.bio.app stop
    conda run -n agentic-spliceai python -m server.bio.app status

`start` is the default, so the bare command is unchanged.

Why this exists rather than `kill $(lsof -ti:8005)`: uvicorn's reloader runs a
**parent and a spawned worker on the same socket**, so `lsof` returns two PIDs
and killing the worker alone just makes the parent respawn it. Stopping cleanly
means killing the reloader parent and waiting for the port to actually free.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time

import uvicorn

from . import config
from .bio_service import app  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

#: Command-line markers identifying our own processes. A listener is only killed
#: when one of these appears in its command line or a *python* ancestor's, so an
#: unrelated service holding the port is reported instead of killed.
_MARKERS = ("-m server.bio.app", "server/bio/app.py", "server.bio.bio_service")

_RELOAD_DIRS = ["server/bio"]


def _run(cmd: list[str]) -> str:
    """Run a command, returning stdout ('' on any failure)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _listeners(port: int) -> list[int]:
    """PIDs listening on `port`."""
    out = _run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"])
    return sorted({int(p) for p in out.split() if p.isdigit()})


def _ppid(pid: int) -> int | None:
    out = _run(["ps", "-o", "ppid=", "-p", str(pid)]).strip()
    return int(out) if out.isdigit() else None


def _command(pid: int) -> str:
    return _run(["ps", "-o", "command=", "-p", str(pid)]).strip()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_python(pid: int) -> bool:
    """Whether `pid`'s executable is a python interpreter."""
    return "python" in _run(["ps", "-o", "comm=", "-p", str(pid)]).strip().rsplit("/", 1)[-1]


def _is_ours(pid: int) -> bool:
    """Whether `pid` belongs to this service, following the parent chain.

    The reload worker's command line is a `multiprocessing.spawn` bootstrap
    naming no module, so identity has to come from an ancestor.

    Every step must itself be a **python** process. Without that, the walk
    escapes into whatever shell launched us, and a shell's command line
    routinely contains our own markers — a `bash -c` running a script that
    merely mentions ``-m server.bio.app`` matched, so an unrelated
    ``python -m http.server 8005`` was classified as ours and killed. Stopping
    at the first non-python ancestor keeps the walk inside our process tree.
    The loop is bounded so a cycle or a PID-1 ancestor cannot hang it.
    """
    seen: set[int] = set()
    cur: int | None = pid
    for _ in range(8):
        if cur is None or cur <= 1 or cur in seen or not _is_python(cur):
            return False
        seen.add(cur)
        cmd = _command(cur)
        if any(marker in cmd for marker in _MARKERS):
            return True
        cur = _ppid(cur)
    return False


def _roots(pids: list[int]) -> list[int]:
    """Reduce a PID set to those with no parent inside it.

    Killing a reload worker on its own accomplishes nothing: the parent
    respawns it. Signalling the roots takes the children with them.
    """
    s = set(pids)
    return [p for p in pids if _ppid(p) not in s]


def stop(port: int, timeout: float = 10.0) -> bool:
    """Stop the service on `port`. True if the port is free afterwards."""
    pids = _listeners(port)
    if not pids:
        print(f"Nothing listening on port {port}.")
        return True

    foreign = [p for p in pids if not _is_ours(p)]
    if foreign:
        for p in foreign:
            print(f"Port {port} is held by an unrelated process: {p} {_command(p)}")
        print("Refusing to kill it. Stop it yourself, or use a different port.")
        return False

    targets = _roots(pids)
    print(f"Stopping {', '.join(map(str, targets))} on port {port}...")
    for p in targets:
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass

    # Wait on the PORT, not on the PIDs: the parent can exit before its worker
    # releases the socket, and a freed PID says nothing about the listener.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _listeners(port):
            print("Stopped.")
            return True
        time.sleep(0.25)

    survivors = [p for p in _listeners(port) if _alive(p)]
    print(f"Still up after {timeout:.0f}s; sending SIGKILL to {survivors}.")
    for p in survivors:
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(1.0)

    if _listeners(port):
        print(f"Port {port} is still held. Investigate with: lsof -nP -iTCP:{port} -sTCP:LISTEN")
        return False
    print("Stopped (forced).")
    return True


def status(port: int) -> bool:
    """Report what holds `port`. True if our service is running."""
    pids = _listeners(port)
    if not pids:
        print(f"Not running (nothing on port {port}).")
        return False

    ours = [p for p in pids if _is_ours(p)]
    for p in pids:
        role = "ours" if p in ours else "FOREIGN"
        print(f"  pid {p:<8} [{role}] ppid={_ppid(p)}  {_command(p)[:88]}")

    if not ours:
        print(f"Port {port} is held by something else.")
        return False
    shape = "reloader + worker" if len(pids) == 2 else f"{len(pids)} process(es)"
    print(f"Running on http://{config.HOST}:{port}  ({shape})")
    return True


def start(port: int) -> None:
    """Run the server in the foreground with auto-reload."""
    if _listeners(port):
        print(f"Port {port} is already in use. Use `restart`, or `status` to see what holds it.")
        raise SystemExit(1)
    uvicorn.run(
        "server.bio.bio_service:app",
        host=config.HOST,
        port=port,
        reload=True,
        reload_dirs=_RELOAD_DIRS,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m server.bio.app",
        description="Run and manage the AgenticSpliceAI Lab server.",
    )
    parser.add_argument(
        "action", nargs="?", default="start",
        choices=("start", "stop", "restart", "status"),
        help="start (default) runs in the foreground with auto-reload",
    )
    parser.add_argument("--port", type=int, default=config.PORT,
                        help=f"port to use (default: {config.PORT})")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="seconds to wait for a graceful stop before SIGKILL")
    args = parser.parse_args(argv)

    if args.action == "status":
        return 0 if status(args.port) else 1
    if args.action == "stop":
        return 0 if stop(args.port, args.timeout) else 1
    if args.action == "restart" and not stop(args.port, args.timeout):
        return 1

    start(args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
