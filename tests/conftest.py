import os
import multiprocessing

import pytest


def pytest_configure(config):
    # Make mining fast in CI/tests, cap runtime
    os.environ.setdefault("QUIP_MINING_NUM_READS", "16")
    os.environ.setdefault("QUIP_MINING_NUM_SWEEPS", "64")
    os.environ.setdefault("QUIP_MINING_TIMEOUT_SEC", "2.0")
    os.environ.setdefault("QUIP_TEST_FAST", "1")
    os.environ.setdefault("QUIP_TEST_REF_TIMEOUT", "3.0")

    # On Linux (and Py 3.12+ generally), the default fork start method
    # combined with any threaded parent — e.g. the QueueListener that
    # Node() spins up for multiprocess logging — triggers deadlock-risk
    # warnings and leaves zombie children at interpreter shutdown.
    # Spawn costs a few ms at process start but doesn't deadlock.
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set earlier in the session by another import; leave it.
        pass

    # Register custom timeout marker to avoid warnings
    config.addinivalue_line(
        "markers", "timeout(seconds): override default test timeout"
    )


@pytest.fixture(scope="session", autouse=True)
def _reap_child_processes():
    """Safety net: terminate any multiprocessing children still alive at
    session end so pytest's process can actually exit. Individual tests
    should still clean up after themselves — this only catches leaks."""
    yield
    for child in multiprocessing.active_children():
        try:
            child.terminate()
            child.join(timeout=1.0)
        except Exception:
            pass



# Global per-test timeout without requiring external plugins.
# Default can be overridden via environment variable PYTEST_DEFAULT_TIMEOUT (seconds),
# or per-test via @pytest.mark.timeout(<seconds>) or @pytest.mark.timeout(seconds=<seconds>).
import signal


def _get_timeout_seconds(item: pytest.Item) -> int:
    try:
        default_timeout = int(float(os.getenv("PYTEST_DEFAULT_TIMEOUT", "30")))
    except Exception:
        default_timeout = 30
    m = item.get_closest_marker("timeout")
    if m:
        # Accept both positional and keyword forms
        if m.args and len(m.args) >= 1:
            try:
                return int(m.args[0])
            except Exception:
                pass
        seconds = m.kwargs.get("seconds") if hasattr(m, "kwargs") else None
        if seconds is not None:
            try:
                return int(seconds)
            except Exception:
                pass
    return default_timeout


def pytest_runtest_setup(item: pytest.Item):
    # Only enforce on POSIX where SIGALRM exists
    if not hasattr(signal, "SIGALRM"):
        return
    timeout = _get_timeout_seconds(item)

    def _handler(signum, frame):
        raise TimeoutError(f"Test timed out after {timeout}s: {item.nodeid}")

    # Store prior handler so we can restore it in teardown
    item._old_alarm_handler = signal.signal(signal.SIGALRM, _handler)
    item._timeout_seconds = timeout
    signal.alarm(timeout)


def pytest_runtest_teardown(item: pytest.Item):
    if not hasattr(signal, "SIGALRM"):
        return
    # Cancel alarm and restore previous handler
    signal.alarm(0)
    old = getattr(item, "_old_alarm_handler", None)
    if old is not None:
        try:
            signal.signal(signal.SIGALRM, old)
        except Exception:
            pass


# --- TEMP DIAGNOSTIC: attribute leaked ProcessPoolExecutors to tests -------
# concurrent.futures.process._threads_wakeups holds one entry per live
# executor manager thread; the thread removes itself on termination. A
# count that rises across a test and never drains is an executor whose
# manager thread survives to interpreter shutdown, where _python_exit
# joins it and hangs. Reported so CI names the offending test(s).
import concurrent.futures.process as _cf_process

_executor_count_before: dict = {}


def _live_executor_count() -> int:
    try:
        return len(_cf_process._threads_wakeups)
    except Exception:
        return -1


def pytest_runtest_logstart(nodeid, location):
    _executor_count_before[nodeid] = _live_executor_count()


def pytest_runtest_logfinish(nodeid, location):
    before = _executor_count_before.pop(nodeid, None)
    after = _live_executor_count()
    if before is not None and after > before:
        import sys
        sys.stderr.write(
            f"\n!!! EXECUTOR-LEAK {nodeid}: _threads_wakeups {before} -> {after}\n"
        )
        sys.stderr.flush()


def pytest_sessionfinish(session, exitstatus):
    import sys
    n = _live_executor_count()
    sys.stderr.write(f"\n!!! SESSION-END live ProcessPoolExecutors: {n}\n")
    sys.stderr.flush()
