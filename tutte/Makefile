REPO_ROOT := $(shell dirname $(CURDIR))

.PHONY: setup test test-tutte benchmark benchmark-paths visualize clean

# ===== Setup =====

setup:
	uv sync --extra dev

setup-viz:
	uv sync --extra dev --extra viz

# ===== Tests =====

test:
	uv run pytest tests/ -v

test-tutte:
	uv run pytest tests/tutte/ -v

# ===== Benchmark =====

benchmark:
	uv run python -m pytest tests/tutte/test_benchmark.py::test_known_families -v -s 2>&1 | tee benchmark_output.log

benchmark-paths:
	uv run python -m pytest tests/tutte/test_benchmark.py::test_path_graphs -v -s 2>&1 | tee benchmark_paths_output.log

# ===== Visualizer =====

visualize:
	PYTHONPATH=$(REPO_ROOT) uv run python scripts/visualize_tutte.py

# ===== Clean =====

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
