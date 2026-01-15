#!/usr/bin/env python3
"""Parse threshold finder log and extract summary statistics.

Numeric parsing errors are treated as hard failures (ValueError should crash).
"""
import re
import statistics
import sys
from pathlib import Path
from typing import Any

# Number of successful samples required by threshold finder (default fallback)
REQUIRED_SAMPLES = 35

# Compiled regex patterns - key off text, not emojis (more robust to encoding issues)
SAMPLE_PATTERN = re.compile(r'Sample (\d+)/(\d+): ([\d.]+)s, GSE=([-\d.]+)')
TOO_FAST_PATTERN = re.compile(r'Too fast: ([\d.]+)s < \d+s, GSE=([-\d.]+)')
TOO_SLOW_PATTERN = re.compile(r'Too slow \((timeout|slow)\): ([\d.]+)s > \d+s(?:, GSE=([-\d.]+))?')
ADJUSTMENT_PATTERN = re.compile(r'Adjusting difficulty (HARDER|EASIER): ([-\d.]+)')

# Config patterns (allow floats for flexibility)
TARGET_TIME_PATTERN = re.compile(r'Target time: ([\d.]+) minutes')
TOLERANCE_PATTERN = re.compile(r'Tolerance: ±([\d.]+)%')
TARGET_RANGE_PATTERN = re.compile(r'Target range: \[([\d.]+)s, ([\d.]+)s\]')

def parse_log(log_file: str) -> dict[str, Any]:
    """Parse log file and extract statistics."""
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract configuration using compiled patterns
    target_match = TARGET_TIME_PATTERN.search(content)
    tolerance_match = TOLERANCE_PATTERN.search(content)
    target_range_match = TARGET_RANGE_PATTERN.search(content)

    target_time = float(target_match.group(1)) if target_match else None
    tolerance = float(tolerance_match.group(1)) if tolerance_match else None
    if target_range_match:
        lower_bound = float(target_range_match.group(1))
        upper_bound = float(target_range_match.group(2))
    else:
        lower_bound = upper_bound = None

    # Extract all attempts using compiled patterns (no emoji dependency)
    in_range = []
    below_range = []
    above_range = []
    all_gse = []
    required_samples = REQUIRED_SAMPLES  # Default fallback

    # Sample pattern captures: (sample_num, total_required, time, gse)
    for match in SAMPLE_PATTERN.finditer(content):
        # Extract required_samples from first match (e.g., "Sample 7/50" -> 50)
        required_samples = int(match.group(2))
        time_val = float(match.group(3))
        gse_val = float(match.group(4))
        in_range.append(time_val)
        all_gse.append(gse_val)

    # Too fast pattern
    for match in TOO_FAST_PATTERN.finditer(content):
        time_val = float(match.group(1))
        gse_val = float(match.group(2))
        below_range.append(time_val)
        all_gse.append(gse_val)

    # Too slow pattern (timeout or slow)
    for match in TOO_SLOW_PATTERN.finditer(content):
        time_val = float(match.group(2))
        above_range.append(time_val)
        if match.group(3):  # GSE might not be present for timeouts
            gse_val = float(match.group(3))
            all_gse.append(gse_val)

    # Difficulty adjustments
    adjustments = []
    for match in ADJUSTMENT_PATTERN.finditer(content):
        direction = match.group(1)
        new_difficulty = float(match.group(2))
        adjustments.append((direction, new_difficulty))
    
    # Calculate statistics
    total_attempts = len(in_range) + len(below_range) + len(above_range)

    return {
        'config': {
            'target_time_min': target_time,
            'tolerance_pct': tolerance,
            'lower_bound_s': lower_bound,
            'upper_bound_s': upper_bound,
        },
        'attempts': {
            'total': total_attempts,
            'in_range': len(in_range),
            'below_range': len(below_range),
            'above_range': len(above_range),
        },
        'times': {
            'in_range': in_range,
            'below_range': below_range,
            'above_range': above_range,
        },
        'gse': all_gse,
        'adjustments': adjustments,
        'required_samples': required_samples,
    }

def print_summary(stats: dict[str, Any]) -> None:
    """Print formatted summary."""
    print("=" * 60)
    print("THRESHOLD FINDER LOG SUMMARY")
    print("=" * 60)

    cfg = stats['config']
    print(f"\n📋 Configuration:")
    if cfg['target_time_min'] is not None and cfg['tolerance_pct'] is not None:
        # Format tolerance as int if whole number, else as float
        tol = cfg['tolerance_pct']
        tol_str = f"{int(tol)}" if tol == int(tol) else f"{tol:.1f}"
        print(f"   Target: {cfg['target_time_min']:.1f} minutes (±{tol_str}%)")
    else:
        print("   Target: Not found in log")
    if cfg['lower_bound_s'] is not None and cfg['upper_bound_s'] is not None:
        # Format bounds as int if whole numbers
        lb = cfg['lower_bound_s']
        ub = cfg['upper_bound_s']
        lb_str = f"{int(lb)}" if lb == int(lb) else f"{lb:.1f}"
        ub_str = f"{int(ub)}" if ub == int(ub) else f"{ub:.1f}"
        print(f"   Range: [{lb_str}s, {ub_str}s]")
    else:
        print("   Range: Not found in log")

    # Use dynamic required_samples from parsed log
    required = stats.get('required_samples', REQUIRED_SAMPLES)

    att = stats['attempts']
    print(f"\n📊 Progress:")
    print(f"   Total attempts: {att['total']}")

    # Warn if no attempts found
    if att['total'] == 0:
        print("   ⚠️  No attempts found - log may be incomplete or format unrecognized")
        return

    print(f"   ✅ In range: {att['in_range']} (need {required})")
    print(f"   ⬇️  Too fast: {att['below_range']}")
    print(f"   ⬆️  Too slow: {att['above_range']}")

    if att['in_range'] > 0:
        pct_complete = (att['in_range'] / required) * 100
        print(f"   Progress: {pct_complete:.1f}%")
    
    # GSE statistics
    if stats['gse']:
        gse = stats['gse']
        print(f"\n🎯 GSE Statistics (n={len(gse)}):")
        print(f"   Min: {min(gse):.1f}")
        print(f"   Max: {max(gse):.1f}")
        print(f"   Mean: {statistics.mean(gse):.1f}")
        print(f"   Median: {statistics.median(gse):.1f}")
        if len(gse) > 1:
            print(f"   StdDev: {statistics.stdev(gse):.1f}")
    
    # Difficulty adjustments
    if stats['adjustments']:
        print(f"\n🔧 Difficulty Adjustments ({len(stats['adjustments'])}):")
        for i, (direction, difficulty) in enumerate(stats['adjustments'][-5:], 1):
            arrow = "↑" if direction == "EASIER" else "↓"
            print(f"   {arrow} {direction}: {difficulty:.1f}")
    
    # Time distribution
    times = stats['times']
    if times['in_range']:
        print(f"\n⏱️  In-Range Times (n={len(times['in_range'])}):")
        print(f"   Mean: {statistics.mean(times['in_range']):.1f}s")
        print(f"   Median: {statistics.median(times['in_range']):.1f}s")
        if len(times['in_range']) > 1:
            print(f"   StdDev: {statistics.stdev(times['in_range']):.1f}s")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python parse_threshold_log.py <log_file>")
        sys.exit(1)

    log_file = sys.argv[1]
    if not Path(log_file).exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)

    try:
        stats = parse_log(log_file)
        print_summary(stats)
    except (OSError, UnicodeDecodeError) as e:
        print(f"Error reading log file: {e}")
        sys.exit(1)
