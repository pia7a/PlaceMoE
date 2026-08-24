"""Supervise one asynchronous planner and stop it when training exits.

The runtime normally reaps the complete planner process group. Cluster job
cancellation can terminate a training rank before Python cleanup runs, so this
small Linux supervisor also requests a parent-death signal from the kernel and
forwards it to the planner session.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
from types import FrameType


_PR_SET_PDEATHSIG = 1
_PARENT_PID_ENV = "PLACEMOE_SUPERVISOR_PARENT_PID"
_child: subprocess.Popen[bytes] | None = None


def _set_parent_death_signal() -> None:
    if not sys.platform.startswith("linux"):
        return
    expected_parent_pid = int(os.environ.pop(_PARENT_PID_ENV, os.getppid()))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def _terminate_session(signum: int, _frame: FrameType | None) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if _child is not None and _child.poll() is None:
        try:
            os.killpg(os.getpgrp(), signum)
        except ProcessLookupError:
            pass
        try:
            _child.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgrp(), signal.SIGKILL)
            except ProcessLookupError:
                pass
    os._exit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    global _child
    command = list(sys.argv[1:] if argv is None else argv)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SystemExit("planner supervisor requires a command after '--'")

    signal.signal(signal.SIGINT, _terminate_session)
    signal.signal(signal.SIGTERM, _terminate_session)
    _set_parent_death_signal()
    _child = subprocess.Popen(command)
    return int(_child.wait())


if __name__ == "__main__":
    raise SystemExit(main())
