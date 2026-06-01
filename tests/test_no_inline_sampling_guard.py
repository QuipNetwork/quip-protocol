# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors
"""The inline-sampling guard must stay green.

Proves no ``STREAMING_PUMP``/``DRIVER_OWNS_FEEDER``/``_sample`` symbols have
crept back into source after the unified-streaming-stack flip.
"""
from tools.lint_no_inline_sampling import main


def test_no_inline_sampling_symbols_in_source():
    assert main() == 0
