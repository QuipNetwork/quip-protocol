#!/usr/bin/env python3
"""
Test script for the adapt_parameters function in CPU/sa_miner.py
Tests the full range of energy requirements from -15700 to -14200
"""

import sys
import os

# Add the project root to the path so we can import the function
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from CPU.sa_miner import SimulatedAnnealingMiner

adapt_parameters = SimulatedAnnealingMiner.adapt_parameters


def test_parameter_constraints():
    """Test that parameters meet expected constraints"""
    
    print("\n" + "=" * 80)
    print("Parameter constraint tests:")
    print("=" * 80)
    
    # Test various difficulty settings
    test_difficulties = [
        (-15700, 0.25, 10),   # Very hard
        (-15650, 0.30, 15),   # Hard+
        (-15600, 0.35, 20),   # Hard
        (-15550, 0.40, 22),   # Hard-
        (-15500, 0.46, 25),   # Knee point
        (-15000, 0.46, 25),   # Medium
        (-14500, 0.60, 50),   # Easier
        (-14200, 0.80, 100),  # Easiest
    ]
    
    print(f"{'Energy':>8} | {'Diversity':>9} | {'Min_Sol':>7} | {'Sweeps':>6} | {'Reads':>5} | {'Reads/Sol':>9}")
    print("-" * 80)
    
    all_valid = True
    for energy, diversity, min_sol in test_difficulties:
        params = adapt_parameters(energy, diversity, min_sol)
        sweeps = params['num_sweeps']
        reads = params['num_reads']
        reads_per_sol = reads / min_sol if min_sol > 0 else 0
        
        print(f"{energy:>8.0f} | {diversity:>9.2f} | {min_sol:>7d} | {sweeps:>6d} | {reads:>5d} | {reads_per_sol:>9.1f}")
        
        # Validate constraints against the live SA bounds
        # (ADAPT_MIN_SWEEPS=64, ADAPT_MAX_SWEEPS=4096).
        if sweeps < 64 or sweeps > 4096:
            print(f"  ERROR: sweeps {sweeps} out of range [64, 4096]")
            all_valid = False
        
        if reads < min_sol * 4:
            print(f"  ERROR: reads {reads} less than min_sol * 4 = {min_sol * 4}")
            all_valid = False
    
    print("\n" + "=" * 80)
    if all_valid:
        print("✅ All parameter constraints satisfied!")
    else:
        print("❌ Some parameter constraints violated!")

    assert all_valid, "Some parameter constraints violated"


def test_monotonicity():
    """Test that sweeps decrease monotonically as energy increases (gets easier)"""
    
    print("\n" + "=" * 80)
    print("Monotonicity test (sweeps should decrease as energy increases):")
    print("=" * 80)
    
    # Dense testing in critical range
    energies = list(range(-15700, -15480, 20)) + list(range(-15500, -14150, 100))
    prev_sweeps = None
    monotonic = True
    
    print(f"{'Energy':>8} | {'Sweeps':>6} | {'Delta':>6} | {'Status'}")
    print("-" * 40)
    
    for energy in energies:
        params = adapt_parameters(energy, 0.46, 25)
        sweeps = params['num_sweeps']
        
        if prev_sweeps is not None:
            delta = sweeps - prev_sweeps
            status = "✅" if delta <= 0 else "❌"
            if delta > 0:
                monotonic = False
        else:
            delta = 0
            status = "—"
        
        print(f"{energy:>8.0f} | {sweeps:>6d} | {delta:>6d} | {status}")
        prev_sweeps = sweeps
    
    print("\n" + "=" * 80)
    if monotonic:
        print("✅ Monotonicity property satisfied!")
    else:
        print("❌ Monotonicity property violated!")

    assert monotonic, "Monotonicity property violated"