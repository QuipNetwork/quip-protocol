#!/usr/bin/env python3
"""
Process mining logs into structured CSV for analysis.

Parses mining.log files from directories organized by miner type (cpu, cuda, qpu)
and generates a CSV with one row per mining attempt.

Usage:
    python tools/process_mining_comparison.py 90m_test_results -o mining_data.csv
"""

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Regex patterns for log parsing
PATTERNS = {
    # Mining start: "2025-11-29 18:35:46,502 - miner.rate-test-cpu-24 - INFO - Mining block 1..."
    # Also matches Gibbs format: "miner.rate-test-cuda-gibbs-0"
    'mining_start': re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - '
        r'miner\.rate-test-([\w-]+?)-(\d+) - INFO - Mining block (\d+)'
    ),

    # Mining attempt: "[rate-test-cpu-15] Mining attempt - Energy: -14928, Valid: 94 ..."
    # Also matches: "[rate-test-cuda-gibbs-0] Mining attempt ..."
    'mining_attempt': re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - '
        r'shared\.quantum_proof_of_work - INFO - '
        r'\[rate-test-([\w-]+?)-(\d+)\] Mining attempt - '
        r'Energy: (-?\d+(?:\.\d+)?), Valid: (\d+) '
        r'\(best 5 diversity: ([\d.]+)\) '
        r'\(requirements: energy<=(-?\d+(?:\.\d+)?)'
    ),

    # Block found: "[rate-test-cuda-1] Block 1 found! ..."
    # Also matches: "[rate-test-cuda-gibbs-0] Block 1 found!"
    'block_found': re.compile(
        r'\[rate-test-([\w-]+?)-(\d+)\] Block (\d+) found!'
    ),

    # Model detection patterns
    'cpu_model': re.compile(r'CPU Model:\s*(.+)'),
    'gpu_model': re.compile(r'\|\s+\d+\s+(NVIDIA[^|]+?)\s+(?:On|Off)\s+\|'),
    'qpu_topology': re.compile(r'Topology:\s*(\S+)'),
}


def parse_timestamp(ts: str) -> datetime:
    """Parse log timestamp to datetime."""
    return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S,%f')


def detect_miner_type(dirname: str) -> str:
    """Detect miner type from directory name."""
    dirname_lower = dirname.lower()
    if 'cpu' in dirname_lower:
        return 'cpu'
    elif 'cuda' in dirname_lower:
        return 'cuda'
    elif 'qpu' in dirname_lower:
        return 'qpu'
    return 'unknown'


def detect_model(log_content: str, miner_type: str) -> str:
    """Extract hardware model from log header."""
    if miner_type == 'cpu':
        match = PATTERNS['cpu_model'].search(log_content)
        return match.group(1).strip() if match else 'unknown'
    elif miner_type == 'cuda':
        match = PATTERNS['gpu_model'].search(log_content)
        return match.group(1).strip() if match else 'h100'
    elif miner_type == 'qpu':
        match = PATTERNS['qpu_topology'].search(log_content)
        return match.group(1) if match else 'Advantage2_system1.13'
    return 'unknown'


def parse_log_file(filepath: Path, miner_machine: str, miner_type: str) -> List[dict]:
    """
    Parse a single mining.log file into attempt records.

    Args:
        filepath: Path to the mining.log file
        miner_machine: Full pod/machine name
        miner_type: Type of miner (cpu, cuda, qpu)

    Returns:
        List of attempt records as dicts
    """
    records = []

    # Read entire file for model detection (first ~5000 chars should be enough)
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        header_content = f.read(5000)

    model = detect_model(header_content, miner_type)

    # Track active mining starts per process: process_id -> (start_time, block_num)
    active_mining: Dict[str, Tuple[datetime, int]] = {}

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Check for mining start - only set if we don't have a start time yet
            # (i.e., first attempt after a block was found or start of log)
            start_match = PATTERNS['mining_start'].search(line)
            if start_match:
                timestamp_str, log_miner_type, process_id, block_num = start_match.groups()
                # Only set start time if we don't already have one for this process
                if process_id not in active_mining:
                    start_time = parse_timestamp(timestamp_str)
                    active_mining[process_id] = (start_time, int(block_num))
                continue

            # Check for block found (reset timer for this process)
            found_match = PATTERNS['block_found'].search(line)
            if found_match:
                log_miner_type, process_id, block_num = found_match.groups()
                # Clear start time - next "Mining block X" will set new start
                if process_id in active_mining:
                    del active_mining[process_id]
                continue

            # Check for mining attempt
            attempt_match = PATTERNS['mining_attempt'].search(line)
            if attempt_match:
                (timestamp_str, log_miner_type, process_id,
                 energy, valid, diversity, threshold) = attempt_match.groups()

                end_time = parse_timestamp(timestamp_str)

                # Get start time from active mining
                if process_id in active_mining:
                    start_time, block_num = active_mining[process_id]
                    time_to_solution = (end_time - start_time).total_seconds()
                else:
                    # No matching start found, use end_time as approximate
                    start_time = end_time
                    block_num = 0
                    time_to_solution = 0.0

                record = {
                    'miner_machine': miner_machine,
                    'miner_type': miner_type,
                    'model': model,
                    'process': int(process_id),
                    'block_num': block_num,
                    'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'time_to_solution': round(time_to_solution, 3),
                    'energy': float(energy),
                    'valid': int(valid),
                    'diversity': float(diversity),
                    'threshold': float(threshold),
                }
                records.append(record)

    return records


def find_log_files(input_dir: Path) -> List[Tuple[Path, str, str]]:
    """
    Find all mining.log files in the input directory.

    Returns:
        List of (filepath, miner_machine, miner_type) tuples
    """
    log_files = []

    # Check for miner type subdirectories (cpu, cuda, qpu)
    for type_dir in input_dir.iterdir():
        if not type_dir.is_dir():
            continue

        miner_type = detect_miner_type(type_dir.name)
        if miner_type == 'unknown':
            continue

        # Look for miner machine directories
        for machine_dir in type_dir.iterdir():
            if not machine_dir.is_dir():
                continue

            miner_machine = machine_dir.name

            # Check for mining.log directly in machine dir (QPU case)
            direct_log = machine_dir / 'mining.log'
            if direct_log.exists():
                log_files.append((direct_log, miner_machine, miner_type))
                continue

            # Check for timestamp subdirectories (CPU/CUDA case)
            for subdir in machine_dir.iterdir():
                if not subdir.is_dir():
                    continue

                log_path = subdir / 'mining.log'
                if log_path.exists():
                    log_files.append((log_path, miner_machine, miner_type))

    return log_files


# Known slow CPU models (can be excluded via --exclude-slow-cpus)
SLOW_CPU_MODELS = [
    'Intel Xeon E3-12xx v2 (Ivy Bridge, IBRS)',
    'Intel(R) Xeon(R) CPU E5-2650L v2 @ 1.70GHz',
]


def main():
    parser = argparse.ArgumentParser(
        description='Process mining logs into structured CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/process_mining_comparison.py 90m_test_results -o mining_data.csv
    python tools/process_mining_comparison.py 90m_test_results --exclude-slow-cpus
    python tools/process_mining_comparison.py 90m_test_results --exclude-cpu-models "Intel Xeon E3-12xx"
        """
    )
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Directory containing mining logs (e.g., 90m_test_results)'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('mining_data.csv'),
        help='Output CSV file (default: mining_data.csv)'
    )
    parser.add_argument(
        '--exclude-slow-cpus',
        action='store_true',
        help='Exclude known slow CPU models (Intel Xeon E3-12xx, E5-2650L)'
    )
    parser.add_argument(
        '--exclude-cpu-models',
        type=str,
        nargs='+',
        default=[],
        help='Exclude CPU models containing these substrings (case-insensitive)'
    )
    parser.add_argument(
        '--list-cpu-models',
        action='store_true',
        help='List all CPU models found and exit'
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory '{args.input_dir}' not found")
        return 1

    print(f"Scanning {args.input_dir} for mining logs...")

    # Find all log files
    log_files = find_log_files(args.input_dir)

    if not log_files:
        print(f"Error: No mining.log files found in {args.input_dir}")
        return 1

    print(f"Found {len(log_files)} log files")

    # Parse all log files
    all_records = []
    stats = {'cpu': 0, 'cuda': 0, 'qpu': 0}

    for log_path, miner_machine, miner_type in log_files:
        print(f"  Parsing {log_path.relative_to(args.input_dir)} ({miner_type})...")
        records = parse_log_file(log_path, miner_machine, miner_type)
        all_records.extend(records)
        stats[miner_type] += len(records)
        print(f"    -> {len(records)} attempts")

    if not all_records:
        print("Error: No mining attempts found in log files")
        return 1

    # List CPU models if requested
    if args.list_cpu_models:
        cpu_models = set(r['model'] for r in all_records if r['miner_type'] == 'cpu')
        print("\nCPU models found:")
        for model in sorted(cpu_models):
            count = sum(1 for r in all_records if r['model'] == model)
            print(f"  {model}: {count:,} records")
        return 0

    # Build exclusion list
    exclude_patterns = list(args.exclude_cpu_models)
    if args.exclude_slow_cpus:
        exclude_patterns.extend(SLOW_CPU_MODELS)

    # Filter out excluded CPU models
    if exclude_patterns:
        original_count = len(all_records)
        filtered_records = []
        excluded_models = set()

        for record in all_records:
            if record['miner_type'] == 'cpu':
                model = record['model']
                should_exclude = any(
                    pattern.lower() in model.lower()
                    for pattern in exclude_patterns
                )
                if should_exclude:
                    excluded_models.add(model)
                    continue
            filtered_records.append(record)

        all_records = filtered_records

        if excluded_models:
            print(f"\nExcluded {original_count - len(all_records):,} records from CPU models:")
            for model in sorted(excluded_models):
                print(f"  - {model}")

    # Sort by miner_machine, process, start_time
    all_records.sort(key=lambda r: (r['miner_machine'], r['process'], r['start_time']))

    # Normalize start_time and end_time to be offsets from each machine's test start
    # (tests ran at different times on different machines)
    print("\nNormalizing timestamps to relative offsets per machine...")
    machine_start_times = {}
    for record in all_records:
        machine = record['miner_machine']
        start_dt = datetime.strptime(record['start_time'], '%Y-%m-%d %H:%M:%S.%f')
        if machine not in machine_start_times or start_dt < machine_start_times[machine]:
            machine_start_times[machine] = start_dt

    for record in all_records:
        machine = record['miner_machine']
        base_time = machine_start_times[machine]
        start_dt = datetime.strptime(record['start_time'], '%Y-%m-%d %H:%M:%S.%f')
        end_dt = datetime.strptime(record['end_time'], '%Y-%m-%d %H:%M:%S.%f')
        # Keep absolute times and add relative offsets
        record['absolute_start_time'] = record['start_time']
        record['absolute_end_time'] = record['end_time']
        # Relative offsets (seconds from machine's test start)
        record['start_time'] = round((start_dt - base_time).total_seconds(), 3)
        record['end_time'] = round((end_dt - base_time).total_seconds(), 3)

    # Write CSV
    fieldnames = [
        'miner_machine', 'miner_type', 'model', 'process', 'block_num',
        'start_time', 'end_time', 'absolute_start_time', 'absolute_end_time',
        'time_to_solution', 'energy', 'valid', 'diversity', 'threshold'
    ]

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nWrote {len(all_records)} records to {args.output}")
    print(f"\nSummary by miner type:")
    for miner_type, count in stats.items():
        if count > 0:
            print(f"  {miner_type.upper()}: {count:,} attempts")

    # Additional stats
    successful = sum(1 for r in all_records if r['valid'] > 0)
    print(f"\nSuccess rate: {successful:,} / {len(all_records):,} ({100*successful/len(all_records):.1f}%)")

    # Energy stats
    energies = [r['energy'] for r in all_records]
    print(f"Energy range: {min(energies):.0f} to {max(energies):.0f}")

    # TTS stats for successful attempts
    tts_values = [r['time_to_solution'] for r in all_records if r['valid'] > 0 and r['time_to_solution'] > 0]
    if tts_values:
        import statistics
        print(f"TTS (successful): mean={statistics.mean(tts_values):.1f}s, median={statistics.median(tts_values):.1f}s")

    return 0


if __name__ == '__main__':
    sys.exit(main())
