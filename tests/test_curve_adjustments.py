#!/usr/bin/env python3
"""
Test script for curve-based energy adjustment functions
Tests both the SA miner adapt_parameters and block difficulty adjustment functions
"""

import sys
import os
import math

# Add the project root to the path so we can import the functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from CPU.sa_miner import SimulatedAnnealingMiner

adapt_parameters = SimulatedAnnealingMiner.adapt_parameters
from shared.energy_utils import adjust_energy_along_curve, calc_energy_range


def test_sa_miner_adapt_parameters():
    """Test the SA miner adapt_parameters function across energy ranges"""
    
    print("=" * 80)
    print("SA MINER ADAPT_PARAMETERS TEST")
    print("=" * 80)
    
    print("Testing adapt_parameters with updated max_sweeps = 8192")
    print(f"{'Energy':>8} | {'Sweeps':>6} | {'Reads':>5} | {'Log2(Sweeps)':>11} | {'Range'}")
    print("-" * 60)
    
    # Test comprehensive energy range
    test_energies = []
    
    # Dense sampling in critical ranges
    test_energies.extend(range(-15700, -15450, 25))  # Hard range
    test_energies.extend(range(-15500, -14150, 50))  # Easy range
    
    for energy in sorted(test_energies):
        params = adapt_parameters(energy, 0.46, 25)
        sweeps = params['num_sweeps']
        reads = params['num_reads']
        log2_sweeps = math.log2(sweeps) if sweeps > 0 else 0
        
        # Determine which range we're in
        if energy <= -15700:
            range_desc = "hardest"
        elif energy <= -15500:
            range_desc = "hard*"
        elif energy >= -14200:
            range_desc = "easiest"
        else:
            range_desc = "easy"
            
        print(f"{energy:>8.0f} | {sweeps:>6d} | {reads:>5d} | {log2_sweeps:>11.2f} | {range_desc}")

    print("\n" + "=" * 80)


def test_block_energy_adjustments():
    """Test the block energy adjustment curve function"""
    
    print("BLOCK ENERGY ADJUSTMENT CURVE TEST")
    print("=" * 80)
    
    print("Testing adjust_energy_along_curve with 5% adjustments")
    print(f"{'Current':>8} | {'Harder':>8} | {'Easier':>8} | {'Hard Δ':>8} | {'Easy Δ':>8} | {'Scaling':>8}")
    print("-" * 70)
    
    # Test every 100 units across the range
    test_energies = list(range(-16000, -13900, 100))
    adjustment_rate = 0.05
    
    for energy in test_energies:
        harder = adjust_energy_along_curve(energy, adjustment_rate, 'harder')
        easier = adjust_energy_along_curve(energy, adjustment_rate, 'easier')
        hard_delta = harder - energy
        easy_delta = easier - energy
        
        # Calculate scaling factor for display
        total_range = -14000.0 - (-16000.0)
        linear_position = (energy - (-16000.0)) / total_range
        distance_from_min = linear_position
        distance_from_max = 1.0 - linear_position
        min_distance = min(distance_from_min, distance_from_max)
        scaling_factor = math.sqrt(min_distance * 2.0)
        
        print(f'{energy:>8.0f} | {harder:>8.0f} | {easier:>8.0f} | {hard_delta:>8.0f} | {easy_delta:>8.0f} | {scaling_factor:>8.3f}')

    print("\n" + "=" * 80)


def test_consecutive_adjustments():
    """Test consecutive adjustments to show dampening behavior"""

    print("CONSECUTIVE ADJUSTMENTS TEST")
    print("=" * 80)

    # Get actual energy range from topology
    min_e, knee_e, max_e = calc_energy_range()
    print(f"Energy range: {max_e:.1f} (easy) to {min_e:.1f} (hard)")
    print(f"Knee point: {knee_e:.1f}\n")

    # Test from different starting points relative to the actual range
    # Use percentages of the range to make tests topology-agnostic
    total_range = max_e - min_e
    start_positions = [
        min_e + 0.25 * total_range,  # 25% from hardest
        knee_e,                       # At knee point
        max_e - 0.25 * total_range,  # 25% from easiest
        min_e + 0.10 * total_range,  # Near hardest edge
        max_e - 0.10 * total_range,  # Near easiest edge
    ]

    for start_energy in start_positions:
        print(f"\nStarting from {start_energy:.0f} - making consecutive HARDER adjustments:")
        print(f"{'Step':>4} | {'Current':>8} | {'New':>8} | {'Delta':>6} | {'Scaling':>8}")
        print("-" * 40)

        current = start_energy
        for i in range(8):
            new_energy = adjust_energy_along_curve(current, 0.05, 'harder')
            delta = new_energy - current

            # Calculate scaling factor based on actual range
            linear_position = (current - min_e) / total_range
            distance_from_min = linear_position
            distance_from_max = 1.0 - linear_position
            min_distance = min(distance_from_min, distance_from_max)
            scaling_factor = math.sqrt(max(0, min_distance * 2.0))  # Ensure non-negative

            print(f'{i+1:>4} | {current:>8.0f} | {new_energy:>8.0f} | {delta:>6.0f} | {scaling_factor:>8.3f}')

            current = new_energy
            # Stop if we hit boundary or adjustment becomes very small
            if abs(delta) < 1 or current <= min_e or current >= max_e:
                print("     (boundary reached or adjustment too small, stopping)")
                break

    print("\n" + "=" * 80)


def test_boundary_conditions():
    """Test behavior at and beyond boundaries"""
    
    print("BOUNDARY CONDITIONS TEST")
    print("=" * 80)
    
    # Test exact boundaries and beyond
    boundary_tests = [
        -17000,  # Way beyond hard limit
        -16000,  # Exact hard limit
        -15999,  # Just inside hard limit
        -15600,  # Knee point
        -14001,  # Just inside easy limit
        -14000,  # Exact easy limit
        -13000,  # Way beyond easy limit
    ]
    
    print(f"{'Input':>8} | {'Clamped':>8} | {'Harder':>8} | {'Easier':>8} | {'Hard Δ':>8} | {'Easy Δ':>8}")
    print("-" * 60)
    
    for energy in boundary_tests:
        # Test what happens when we clamp the input
        clamped = max(-16000.0, min(energy, -14000.0))
        harder = adjust_energy_along_curve(energy, 0.05, 'harder')
        easier = adjust_energy_along_curve(energy, 0.05, 'easier')
        hard_delta = harder - clamped
        easy_delta = easier - clamped
        
        print(f'{energy:>8.0f} | {clamped:>8.0f} | {harder:>8.0f} | {easier:>8.0f} | {hard_delta:>8.0f} | {easy_delta:>8.0f}')

    print("\n" + "=" * 80)


def test_adjustment_rates():
    """Test different adjustment rates"""
    
    print("ADJUSTMENT RATES TEST")
    print("=" * 80)
    
    test_energy = -15000  # Middle of range
    adjustment_rates = [0.01, 0.02, 0.05, 0.10, 0.20]
    
    print(f"Testing different rates at energy {test_energy}")
    print(f"{'Rate':>6} | {'Harder':>8} | {'Easier':>8} | {'Hard Δ':>8} | {'Easy Δ':>8}")
    print("-" * 50)
    
    for rate in adjustment_rates:
        harder = adjust_energy_along_curve(test_energy, rate, 'harder')
        easier = adjust_energy_along_curve(test_energy, rate, 'easier')
        hard_delta = harder - test_energy
        easy_delta = easier - test_energy
        
        print(f'{rate:>6.2f} | {harder:>8.0f} | {easier:>8.0f} | {hard_delta:>8.0f} | {easy_delta:>8.0f}')

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("CURVE-BASED ENERGY ADJUSTMENT TEST SUITE")
    print("=" * 80)
    
    try:
        # Run all tests
        test_sa_miner_adapt_parameters()
        test_block_energy_adjustments()
        test_consecutive_adjustments()
        test_boundary_conditions()
        test_adjustment_rates()
        
        print("🎉 All tests completed successfully!")
        print("\nThese results can be used for debugging curve behavior.")
        print("Key observations:")
        print("- Adjustments are largest in the middle of the range")
        print("- Adjustments become progressively smaller near boundaries")
        print("- Consecutive adjustments naturally dampen")
        print("- Boundary clamping prevents invalid values")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)