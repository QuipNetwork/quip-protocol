#!/usr/bin/env python3
"""
Shared helpers for mining-visualisation tools.

Used by visualize_mining_performance.py and visualize_comparative_performance.py.
"""

import logging
from typing import Dict, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

# Standard color scheme for miner types
MINER_COLORS: Dict[str, str] = {
    'CPU': '#e74c3c',
    'GPU': '#3498db',
    'QPU': '#2ecc71',
}


def normalize_miner_type(miner_type: str) -> str:
    """Normalize miner type to CPU/GPU/QPU for display."""
    miner_type = miner_type.lower()
    if miner_type == 'cuda':
        return 'GPU'
    return miner_type.upper()


def get_device_counts(df: pd.DataFrame) -> Dict[str, int]:
    """Count actual devices per miner type from the data.

    For GPU/CUDA: count unique (miner_machine, process) pairs = number of GPUs.
    For CPU: count unique (miner_machine, process) pairs = number of CPU workers.
    For QPU: count unique miner_machine = number of QPUs.

    Args:
        df: DataFrame with mining data.

    Returns:
        Dict mapping display type (CPU/GPU/QPU) to device count.
    """
    counts: Dict[str, int] = {}

    for miner_type, miner_df in df.groupby('miner_type'):
        display_type = normalize_miner_type(miner_type)

        if display_type == 'QPU':
            count = miner_df['miner_machine'].nunique()
        else:
            count = miner_df.groupby(['miner_machine', 'process']).ngroups

        counts[display_type] = count

    return counts


def load_mining_csv(
    filepath: str,
    required_columns: Optional[Set[str]] = None,
    parse_dates: bool = False,
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    """Load mining data CSV into a DataFrame.

    Args:
        filepath: Path to the CSV file.
        required_columns: Set of column names that must be present.  When
            ``None`` no column validation is performed.
        parse_dates: When ``True``, attempt to parse ``start_time`` and
            ``end_time`` columns as datetime values.
        encoding: Optional encoding passed to ``pd.read_csv``.

    Returns:
        Loaded DataFrame.

    Raises:
        ValueError: If ``required_columns`` is given and any are absent.
    """
    read_kwargs: dict = {}
    if encoding is not None:
        read_kwargs['encoding'] = encoding

    df = pd.read_csv(filepath, **read_kwargs)

    if required_columns is not None:
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV missing required columns: {', '.join(sorted(missing))}"
            )

    if parse_dates:
        for col in ('start_time', 'end_time'):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

    return df
