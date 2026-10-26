"""Explicit ART 0.5.17 teardown, including orphan-prone vLLM descendants.

ART's mp_actors.Proxy.close() terminates only its direct child. Snapshot the
owned tree BEFORE closing the proxy; afterwards descendants may have PPID=1.
Never use pkill, GPU-wide process lists, or kill the caller's process group.
"""

import asyncio
import functools
import logging
import os
import sys
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field

import psutil

logger = logging.getLogger(__name__)


@dataclass
class _Lifecycle:
    monitors: set[asyncio.Task] = field(default_factory=set)
    processes: dict[tuple[int, float], psutil.Process] = field(default_factory=dict)
    capture_errors: list[str] = field(default_factory=list)


def _state(backend) -> _Lifecycle:
    state = vars(backend).get("_agenticrl_lifecycle")
    if state is None:
        state = _Lifecycle()
        backend._agenticrl_lifecycle = state
    return state


def _capture_owned_processes(backend) -> None:
    state = _state(backend)
    # These fields are pinned-version adapters, not process-name guesses.
    # vars() avoids triggering mp_actors.Proxy.__getattr__ / remote RPCs.
    for service in vars(backend).get("_services", {}).values():
        for key in ("_process", "_vllm_process"):
            handle = vars(service).get(key)
            pid = getattr(handle, "pid", None)
            if pid is None or pid == os.getpid():
                continue
            # Consult the original OS child handle before trusting its PID:
            # a dead mp.Process/Popen may otherwise name a reused PID.
            if getattr(handle, "exitcode", None) is not None:
                continue
            poll = getattr(handle, "poll", None)
            if poll is not None and poll() is not None:
                continue
            try:
                root = psutil.Process(pid)
                # Only a backend's direct child establishes ownership. A PID
                # from a remote/shared service is never authority to kill it.
                if root.ppid() != os.getpid():
                    continue
                for process in [root, *root.children(recursive=True)]:
                    identity = (process.pid, process.create_time())
                    state.processes[identity] = process
            except psutil.NoSuchProcess:
                pass
            except psutil.Error as exc:
                state.capture_errors.append(f"Cannot inspect backend PID {pid}: {exc}")


def install_monitor_workaround(backend_class) -> None:
    """Keep the existing health-check tolerance, without nested global patches.

    Track only this backend's monitors so notebook/other asyncio tasks are not
    cancelled. Capture children before ART's done callback can pop a service.
    CancelledError is deliberately NOT suppressed.
    """
    original = backend_class._monitor_openai_server
    if vars(original).get("_agenticrl_wrapped", False):
        return

    @functools.wraps(original)
    async def safe_monitor(self, *args, **kwargs):
        state = _state(self)
        task = asyncio.current_task()
        if task is not None:
            state.monitors.add(task)
        _capture_owned_processes(self)
        try:
            await original(self, *args, **kwargs)
        except Exception as exc:
            logger.warning(
                "[lifecycle] suppressed _monitor_openai_server crash: %s: %s",
                type(exc).__name__, exc, exc_info=True,
            )
        finally:
            _capture_owned_processes(self)
            if task is not None:
                state.monitors.discard(task)

    safe_monitor.__dict__["_agenticrl_wrapped"] = True
    backend_class._monitor_openai_server = safe_monitor


def _live(process: psutil.Process) -> bool:
    try:
        # is_running() checks cached creation time: never signal a reused PID.
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _reap_owned_processes(state: _Lifecycle, terminate_timeout: float, kill_timeout: float) -> None:
    processes = [p for p in reversed(list(state.processes.values())) if _live(p)]
    for process in processes:
        with suppress(psutil.NoSuchProcess):
            process.terminate()
    _, alive = psutil.wait_procs(processes, timeout=terminate_timeout)
    alive = [p for p in alive if _live(p)]
    for process in alive:
        with suppress(psutil.NoSuchProcess):
            process.kill()
    _, alive = psutil.wait_procs(alive, timeout=kill_timeout)
    survivors = [p.pid for p in alive if _live(p)]
    if survivors:
        raise RuntimeError(f"Backend processes still alive after teardown: {survivors}")
    if state.capture_errors:
        raise RuntimeError("; ".join(state.capture_errors))


async def _shutdown(backend, *, terminate_timeout: float, kill_timeout: float) -> None:
    state = _state(backend)
    _capture_owned_processes(backend)
    print(f"[lifecycle] Closing ART backend; tracking {len(state.processes)} owned process(es)", flush=True)
    try:
        # ART 0.5.17 close() has bounded proxy joins; call it while the loop is
        # still alive, before asyncio.run() cancels monitors and pops services.
        await backend.close()
    finally:
        monitors = list(state.monitors)
        for task in monitors:
            task.cancel()
        try:
            if monitors:
                await asyncio.gather(*monitors, return_exceptions=True)
        finally:
            # Even failed backend.close() must not strand known descendants.
            await asyncio.to_thread(
                _reap_owned_processes, state, terminate_timeout, kill_timeout,
            )
    print("[lifecycle] Backend cleanup complete", flush=True)


@asynccontextmanager
async def managed_backend(backend, *, terminate_timeout: float = 5.0, kill_timeout: float = 2.0):
    """Close on success, registration/training errors, and task cancellation.

    Cleanup failure fails a successful command, but does not overwrite an
    original training/evaluation exception. Does not touch checkpoints.
    """
    _state(backend)
    error = None
    try:
        yield backend
    except BaseException as exc:
        error = exc
        raise
    finally:
        cleanup = asyncio.create_task(_shutdown(
            backend, terminate_timeout=terminate_timeout, kill_timeout=kill_timeout,
        ))
        try:
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                # A second interrupt should not cancel process-tree cleanup.
                await cleanup
                raise
        except Exception as exc:
            if error is None:
                raise
            error.add_note(f"ART backend cleanup also failed: {exc}")
            print(f"[lifecycle] cleanup failed: {exc}", file=sys.stderr, flush=True)
