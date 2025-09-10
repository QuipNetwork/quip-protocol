"""Energy adjustment utilities for quantum blockchain."""

import math


def adjust_energy_along_curve(current_energy: float, adjustment_rate: float, direction: str) -> float:
    """Adjust energy along a curve that compresses adjustments near boundaries.
    
    Uses a sqrt-based curve from min_energy (-16000) to max_energy (-14000) with knee at -15600.
    Adjustments become smaller as we approach the extremes, larger near the knee point.
    Beyond observed limits, returns a simple linear adjustment that calling functions can handle.
    
    Args:
        current_energy: Current energy value
        adjustment_rate: Percentage to move (e.g., 0.05 for 5%)
        direction: 'harder' (more negative) or 'easier' (less negative)
    
    Returns:
        New energy value after curve-based adjustment
    """
    min_energy = -16000.0  # Hardest (approximate, not hard limit)
    knee_energy = -15600.0  # Knee point
    max_energy = -14000.0  # Easiest (approximate, not hard limit)
    
    # Convert energy to normalized position [0, 1] for observed range
    total_range = max_energy - min_energy
    
    # Handle out-of-range values with linear adjustment
    if current_energy < min_energy or current_energy > max_energy:
        linear_adjustment = total_range * adjustment_rate
        if direction == 'harder':
            return current_energy - linear_adjustment
        else:  # easier
            return current_energy + linear_adjustment
    
    # Normalize current position [0, 1]
    normalized_pos = (current_energy - min_energy) / total_range
    
    # Create curve using sqrt function
    # At position 0 (min_energy): curve_factor ≈ 0.1 (small adjustments)
    # At position 0.3 (knee): curve_factor ≈ 1.0 (full adjustments)  
    # At position 1 (max_energy): curve_factor ≈ 0.1 (small adjustments)
    
    knee_pos = (knee_energy - min_energy) / total_range  # ≈ 0.3
    
    if normalized_pos <= knee_pos:
        # Left side: increase from 0.1 to 1.0
        progress = normalized_pos / knee_pos
        curve_factor = 0.1 + 0.9 * math.sqrt(progress)
    else:
        # Right side: decrease from 1.0 to 0.1
        progress = (normalized_pos - knee_pos) / (1.0 - knee_pos)
        curve_factor = 1.0 - 0.9 * math.sqrt(progress)
    
    # Apply curved adjustment
    curved_adjustment = total_range * adjustment_rate * curve_factor
    
    if direction == 'harder':
        return current_energy - curved_adjustment
    else:  # easier
        return current_energy + curved_adjustment
