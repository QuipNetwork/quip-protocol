from shared.block_requirements import BlockRequirements, calculate_requirements_decay


def test_decay_zero_steps_no_change():
    """Test that requirements are unchanged when no decay is applied."""
    req = BlockRequirements(difficulty_energy=-100.0, min_diversity=0.30, min_solutions=20, timeout_to_difficulty_adjustment_decay=10)

    # Verify requirements unchanged when no decay steps
    cur_req = req.to_json()
    assert cur_req['difficulty_energy'] == -100.0
    assert cur_req['min_diversity'] == 0.30
    assert cur_req['min_solutions'] == 20


def test_decay_one_step_applied():
    """Test that one decay step eases requirements in the expected direction."""
    req = BlockRequirements(difficulty_energy=-100.0, min_diversity=0.50, min_solutions=20, timeout_to_difficulty_adjustment_decay=10)
    original_energy = req.difficulty_energy
    original_diversity = req.min_diversity
    original_solutions = req.min_solutions

    # Build decayed requirements
    cur_req = req.to_json()
    cur_req = calculate_requirements_decay(cur_req)

    # Verify decay moves requirements in "easier" direction
    # Energy gets less negative (easier)
    assert cur_req['difficulty_energy'] > original_energy
    # Diversity requirement drops (easier)
    assert cur_req['min_diversity'] <= original_diversity
    # Solutions requirement drops (easier)
    assert cur_req['min_solutions'] <= original_solutions


def test_decay_multiple_steps_flooring():
    """Test that multiple decay steps respect the blockchain-defined floors."""
    from shared.energy_utils import DEFAULT_DIVERSITY_RANGE, DEFAULT_SOLUTIONS_RANGE

    req = BlockRequirements(difficulty_energy=-100.0, min_diversity=0.35, min_solutions=10, timeout_to_difficulty_adjustment_decay=5)

    # Apply many steps to hit the floors
    cur_req = req.to_json()
    for _ in range(20):
        cur_req = calculate_requirements_decay(cur_req)

    # Verify floors are respected (from energy_utils defaults)
    # DEFAULT_DIVERSITY_RANGE = (0.3, 0.3) - minimum is 0.3
    # DEFAULT_SOLUTIONS_RANGE = (5, 5) - minimum is 5
    assert cur_req['min_diversity'] >= DEFAULT_DIVERSITY_RANGE[0]
    assert cur_req['min_solutions'] >= DEFAULT_SOLUTIONS_RANGE[0]

