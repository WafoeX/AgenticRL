"""CPU-only lifecycle regressions, including REAL orphan-prone subprocesses."""

import asyncio
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import psutil
import pytest

from src import backend_lifecycle as lifecycle
from tests import test_phase1_integration as _phase1

entrypoints = _phase1.entrypoints
write_scenarios = _phase1.write_scenarios


@pytest.mark.parametrize("entry", ["train", "eval", "both"])
def test_success_closes_backend_and_only_then_reports_complete(entrypoints, tmp_path, capsys, entry):
    path = write_scenarios(tmp_path)
    output = tmp_path / "report.json"
    if entry == "train":
        asyncio.run(entrypoints.train.train(
            num_steps=2, rollouts_per_step=2, num_generations=2,
            train_scenarios_path=path, val_scenarios_path=path,
        ))
        assert entrypoints.backend.train.await_count == 2
    else:
        asyncio.run(entrypoints.evaluate.main(
            path, 2, mode="both" if entry == "both" else "sample", output_path=str(output),
        ))
        assert json.loads(output.read_text())["mode"] == ("both" if entry == "both" else "sample")
    entrypoints.backend.close.assert_awaited_once()
    log = capsys.readouterr().out
    assert log.index("Backend cleanup complete") < log.index("Backend resources released")
    if entry != "train":
        assert log.index("Detailed results saved") < log.index("Closing ART backend")


@pytest.mark.parametrize("entry", ["train", "eval"])
@pytest.mark.parametrize("fault", ["register", "get_step", "work"])
def test_entrypoint_errors_still_close(entrypoints, tmp_path, entry, fault, monkeypatch):
    error = RuntimeError(f"{entry} {fault} failed")
    if fault == "work":
        if entry == "train":
            entrypoints.backend.train.side_effect = error
        else:
            monkeypatch.setattr(entrypoints.evaluate, "_evaluate_mode", AsyncMock(side_effect=error))
    else:
        getattr(entrypoints.model, fault).side_effect = error
    path = write_scenarios(tmp_path)
    with pytest.raises(RuntimeError) as caught:
        if entry == "train":
            asyncio.run(entrypoints.train.train(
                num_steps=1, rollouts_per_step=2, num_generations=2,
                train_scenarios_path=path, val_scenarios_path=path,
            ))
        else:
            asyncio.run(entrypoints.evaluate.main(path, 1, output_path=str(tmp_path / "out.json")))
    assert caught.value is error
    entrypoints.backend.close.assert_awaited_once()


def test_output_write_failure_still_closes(entrypoints, tmp_path):
    with pytest.raises(IsADirectoryError):
        asyncio.run(entrypoints.evaluate.main(write_scenarios(tmp_path), 1, output_path=str(tmp_path)))
    entrypoints.backend.close.assert_awaited_once()


def test_cleanup_failure_leaves_saved_result_but_does_not_claim_success(entrypoints, tmp_path, capsys):
    entrypoints.backend.close.side_effect = RuntimeError("close failed")
    output = tmp_path / "out.json"
    with pytest.raises(RuntimeError, match="close failed"):
        asyncio.run(entrypoints.evaluate.main(write_scenarios(tmp_path), 1, output_path=str(output)))
    assert json.loads(output.read_text())["total_rollouts"] == 1
    log = capsys.readouterr().out
    assert "Detailed results saved" in log
    assert "Evaluation complete!" not in log


def test_cleanup_failure_does_not_mask_primary_failure(entrypoints, tmp_path):
    primary = ValueError("registration failed")
    entrypoints.model.register.side_effect = primary
    entrypoints.backend.close.side_effect = RuntimeError("close failed")
    with pytest.raises(ValueError) as caught:
        asyncio.run(entrypoints.evaluate.main(write_scenarios(tmp_path), 1))
    assert caught.value is primary
    assert any("close failed" in note for note in primary.__notes__)


def test_monitor_patch_is_idempotent_and_cancellation_propagates():
    class Backend:
        _monitor_openai_server = AsyncMock(side_effect=asyncio.CancelledError)

    lifecycle.install_monitor_workaround(Backend)
    wrapped = Backend._monitor_openai_server
    lifecycle.install_monitor_workaround(Backend)
    assert Backend._monitor_openai_server is wrapped
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(Backend()._monitor_openai_server())


def test_monitor_failure_remains_logged_and_suppressed(caplog):
    class Backend:
        _monitor_openai_server = AsyncMock(side_effect=RuntimeError("health check failed"))

    lifecycle.install_monitor_workaround(Backend)
    asyncio.run(Backend()._monitor_openai_server())
    assert "health check failed" in caplog.text


def test_cancellation_cleans_backend_and_only_its_monitors():
    async def run():
        entered = asyncio.Event()
        monitoring = asyncio.Event()
        unrelated = asyncio.create_task(asyncio.Event().wait())
        events = []

        class Backend:
            async def _monitor_openai_server(self):
                monitoring.set()
                await asyncio.Event().wait()

            async def close(self):
                events.append("close")

        lifecycle.install_monitor_workaround(Backend)
        backend = Backend()
        monitor = None

        async def work():
            nonlocal monitor
            async with lifecycle.managed_backend(backend):
                monitor = asyncio.create_task(backend._monitor_openai_server())
                await monitoring.wait()
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(work())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert events == ["close"]
        assert monitor is not None and monitor.cancelled()
        assert not unrelated.done()  # e.g. Colab's kernel work is never cancelled.
        unrelated.cancel()
        await asyncio.gather(unrelated, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("dead_handle", ["mp", "popen"])
def test_dead_handle_cannot_authorize_a_reused_pid(monkeypatch, dead_handle):
    handle = SimpleNamespace(pid=12345)
    if dead_handle == "mp":
        handle.exitcode = 0
    else:
        handle.poll = lambda: 0
    backend = SimpleNamespace(_services={"model": SimpleNamespace(_process=handle)})
    process = MagicMock()
    monkeypatch.setattr(lifecycle.psutil, "Process", process)
    lifecycle._capture_owned_processes(backend)
    process.assert_not_called()
    assert not lifecycle._state(backend).processes


def test_non_child_root_cannot_authorize_signals(monkeypatch):
    backend = SimpleNamespace(_services={"model": SimpleNamespace(_process=SimpleNamespace(pid=12345))})
    root = MagicMock()
    root.ppid.return_value = os.getpid() + 1
    monkeypatch.setattr(lifecycle.psutil, "Process", lambda pid: root)
    lifecycle._capture_owned_processes(backend)
    root.children.assert_not_called()
    assert not lifecycle._state(backend).processes


def test_reused_pid_snapshot_is_not_signalled(monkeypatch):
    process = MagicMock()
    process.is_running.return_value = False  # psutil detected different creation time.
    state = lifecycle._Lifecycle(processes={(12345, 1.0): process})
    lifecycle._reap_owned_processes(state, 0.01, 0.01)
    process.terminate.assert_not_called()
    process.kill.assert_not_called()


@pytest.mark.skipif(os.name != "posix", reason="Colab/Linux and local macOS process-tree check")
@pytest.mark.parametrize("ignore_term", [False, True])
@pytest.mark.parametrize("close_fails", [False, True])
def test_real_orphan_tree_reaped_without_touching_sibling(tmp_path, ignore_term, close_fails):
    child_code = (
        "import signal,time; "
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_term else "")
        + "print('ready',flush=True); time.sleep(60)"
    )
    parent_code = (
        "import subprocess,sys,time; "
        f"child=subprocess.Popen([sys.executable,'-u','-c',{child_code!r}],stdout=subprocess.PIPE,text=True); "
        "assert child.stdout.readline().strip()=='ready'; "
        "print(child.pid,flush=True); time.sleep(60)"
    )
    parent = subprocess.Popen(
        [sys.executable, "-u", "-c", parent_code], stdout=subprocess.PIPE, text=True,
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    child = None
    try:
        assert parent.stdout is not None
        child = psutil.Process(int(parent.stdout.readline()))
        # Force psutil to cache creation time before parent termination/reparenting.
        child.create_time()
        checkpoint = tmp_path / "checkpoint"
        checkpoint.write_bytes(b"already-saved-model")

        async def close():
            if close_fails:
                raise RuntimeError("backend close failed")
            # Exactly the orphan-prone shape of ART Proxy.close(): parent ONLY.
            parent.terminate()
            parent.wait(timeout=3)

        backend = SimpleNamespace(
            _services={"model": SimpleNamespace(_process=parent)}, close=close,
        )

        async def run():
            async with lifecycle.managed_backend(backend, terminate_timeout=0.3, kill_timeout=1.0):
                assert child.is_running()

        if close_fails:
            with pytest.raises(RuntimeError, match="backend close failed"):
                asyncio.run(run())
        else:
            asyncio.run(run())
        assert not lifecycle._live(child)
        assert parent.poll() is not None
        assert unrelated.poll() is None
        assert checkpoint.read_bytes() == b"already-saved-model"
    finally:
        if child is not None and lifecycle._live(child):
            child.kill()
        for process in (parent, unrelated):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)
        if parent.stdout is not None:
            parent.stdout.close()
