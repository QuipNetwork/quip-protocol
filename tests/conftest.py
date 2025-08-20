import os

def pytest_configure(config):
    # Make mining fast in CI/tests, cap runtime
    os.environ.setdefault("QUIP_MINING_NUM_READS", "16")
    os.environ.setdefault("QUIP_MINING_NUM_SWEEPS", "64")
    os.environ.setdefault("QUIP_MINING_TIMEOUT_SEC", "2.0")
    os.environ.setdefault("QUIP_TEST_FAST", "1")
    os.environ.setdefault("QUIP_TEST_REF_TIMEOUT", "3.0")

