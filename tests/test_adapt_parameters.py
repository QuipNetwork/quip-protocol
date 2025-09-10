#!/usr/bin/env python3
"""
Test script for the adapt_parameters function in CPU/sa_miner.py
Tests the full range of energy requirements from -15700 to -14200
"""

import sys
import os
import math

# Add the project root to the path so we can import the function
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from CPU.sa_miner import adapt_parameters


def test_adapt_parameters_range():
    """Test adapt_parameters across the full energy range"""
    
    print("Testing adapt_parameters function across energy range [-15700, -14200]")
    print("=" * 80)
    
    # Test boundary conditions
    test_cases = [
        # Energy, expected behavior
        (-15700, "hardest - should use max_sweeps (8192)"),
        (-15701, "beyond hardest - should use max_sweeps (8192)"),
        (-15650, "between hardest and knee - should interpolate"),
        (-15600, "between hardest and knee - should interpolate"),
        (-15550, "between hardest and knee - should interpolate"),
        (-15520, "near knee - should interpolate"),
        (-15500, "knee point - should use knee_sweeps (2048)"),
        (-15000, "mid-range - should interpolate (~400-500)"),
        (-14500, "easier - should interpolate (~100-200)"),
        (-14200, "easiest - should use min_sweeps (32)"),
        (-14100, "beyond easiest - should use min_sweeps (32)"),
    ]
    
    print(f"{'Energy':>8} | {'Sweeps':>6} | {'Reads':>5} | {'Description'}")
    print("-" * 80)
    
    for energy, description in test_cases:
        params = adapt_parameters(energy, 0.46, 25)
        sweeps = params['num_sweeps']
        reads = params['num_reads']
        print(f"{energy:>8.0f} | {sweeps:>6d} | {reads:>5d} | {description}")
    
    print("\n" + "=" * 80)
    print("Detailed energy sweep test (focus on -15700 to -15500 range):")
    print(f"{'Energy':>8} | {'Sweeps':>6} | {'Log2(Sweeps)':>11} | {'Fraction':>8}")
    print("-" * 50)
    
    # Test a range of energies to verify smooth interpolation
    min_observed_energy = -15700
    knee_energy = -15500
    max_observed_energy = -14200
    
    # More granular testing in the critical range between hardest and knee
    test_energies = []
    
    # Dense sampling between -15700 and -15500 (hardest to knee)
    test_energies.extend(range(-15700, -15500, 25))
    
    # Regular sampling for the rest
    test_energies.extend(range(-15500, -14150, 100))
    
    for energy in sorted(test_energies):
        params = adapt_parameters(energy, 0.46, 25)
        sweeps = params['num_sweeps']
        log2_sweeps = math.log2(sweeps) if sweeps > 0 else 0
        
        # Calculate fraction based on which range we're in
        if energy <= min_observed_energy:
            fraction_str = "hardest"
        elif energy >= max_observed_energy:
            fraction_str = "easiest"
        elif energy <= knee_energy:
            # Between hardest and knee - interpolating between max_sweeps and knee_sweeps
            fraction = (energy - min_observed_energy) / (knee_energy - min_observed_energy)
            fraction_str = f"{fraction:.3f}*"
        else:
            # Between knee and easiest - interpolating between knee_sweeps and min_sweeps
            fraction = (energy - knee_energy) / (max_observed_energy - knee_energy)
            fraction_str = f"{fraction:.3f}"
            
        print(f"{energy:>8.0f} | {sweeps:>6d} | {log2_sweeps:>11.2f} | {fraction_str:>8}")


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
        
        # Validate constraints
        if sweeps < 32 or sweeps > 8192:
            print(f"  ERROR: sweeps {sweeps} out of range [32, 8192]")
            all_valid = False
        
        if reads < min_sol * 4:
            print(f"  ERROR: reads {reads} less than min_sol * 4 = {min_sol * 4}")
            all_valid = False
    
    print("\n" + "=" * 80)
    if all_valid:
        print("✅ All parameter constraints satisfied!")
    else:
        print("❌ Some parameter constraints violated!")
    
    return all_valid


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
    
    return monotonic


if __name__ == "__main__":
    print("Adaptive Parameters Test Suite")
    print("=" * 80)
    
    try:
        # Run all tests
        test_adapt_parameters_range()
        constraints_ok = test_parameter_constraints()
        monotonic_ok = test_monotonicity()
        
        print("\n" + "=" * 80)
        print("SUMMARY:")
        print(f"Parameter constraints: {'✅ PASS' if constraints_ok else '❌ FAIL'}")
        print(f"Monotonicity:         {'✅ PASS' if monotonic_ok else '❌ FAIL'}")
        
        if constraints_ok and monotonic_ok:
            print("\n🎉 All tests PASSED!")
            sys.exit(0)
        else:
            print("\n💥 Some tests FAILED!")
            sys.exit(1)
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)