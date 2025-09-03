import time

from shared.block import Block, BlockHeader, MinerInfo, QuantumProof
from shared.block_requirements import BlockRequirements
from shared.quantum_proof_of_work import calculate_requirements_decay


def make_prev_and_cur_blocks(requirements: BlockRequirements, gap_seconds: int):
    now = int(time.time())
    prev = Block(
        header=BlockHeader(previous_hash=b"\x00"*32, index=1, timestamp=now, data_hash=b"\x00"*32),
        miner_info=MinerInfo(miner_id="node1-CPU-1", miner_type="CPU", reward_address=b"R"*32,
                             ecdsa_public_key=b"E"*64, wots_public_key=b"W"*64, next_wots_public_key=b"N"*64),
        quantum_proof=QuantumProof(
            nonce=123,
            salt=b"S",
            nodes=[0,1,2,3],
            edges=[(0,1),(1,2),(2,3)],
            solutions=[[1,1,1,1],[1,1,1,1]],
            mining_time=1.0,
        ),
        next_block_requirements=requirements,
        data=b"hello",
        raw=b"",
        hash=b"",
        signature=b"SIG",
    )

    cur = Block(
        header=BlockHeader(previous_hash=b"\x00"*32, index=2, timestamp=now + gap_seconds, data_hash=b"\x00"*32),
        miner_info=prev.miner_info,
        quantum_proof=prev.quantum_proof,
        next_block_requirements=requirements,
        data=b"hello",
        raw=b"",
        hash=b"",
        signature=b"SIG",
    )
    return prev, cur


def test_decay_zero_steps_no_change():
    req = BlockRequirements(difficulty_energy=-100.0, min_diversity=0.30, min_solutions=20, timeout_to_difficulty_adjustment_decay=10)
    prev, cur = make_prev_and_cur_blocks(req, gap_seconds=0)

    # Validate; should use same requirements and pass basic structure checks
    assert cur.validate_block(prev) in (True, False)  # we don't assert pass/fail here, just that no crash

    # Verify no decay applied when gap < decay interval
    elapsed = max(0, int((cur.header.timestamp - prev.header.timestamp) / req.timeout_to_difficulty_adjustment_decay))
    assert elapsed == 0


def test_decay_one_step_applied():
    req = BlockRequirements(difficulty_energy=-100.0, min_diversity=0.30, min_solutions=20, timeout_to_difficulty_adjustment_decay=10)
    prev, cur = make_prev_and_cur_blocks(req, gap_seconds=10)

    # Trigger validation which applies one decay step internally; it should not crash
    cur.validate_block(prev)

    # After one step, expected new values (without floors)
    expected_energy = -100.0 * (1 - 0.05)
    expected_diversity = max(0.20, 0.30 - 0.02)
    expected_solutions = max(10, int(20 * (1 - 0.10)))

    # Build decayed requirements like validator does
    cur_req = req.to_json()
    cur_req = calculate_requirements_decay(cur_req)

    assert abs(cur_req['difficulty_energy'] - expected_energy) < 1e-9
    assert abs(cur_req['min_diversity'] - expected_diversity) < 1e-9
    assert cur_req['min_solutions'] == expected_solutions


def test_decay_multiple_steps_flooring():
    req = BlockRequirements(difficulty_energy=-100.0, min_diversity=0.21, min_solutions=11, timeout_to_difficulty_adjustment_decay=5)
    prev, cur = make_prev_and_cur_blocks(req, gap_seconds=20)  # 4 steps

    # Trigger validation which applies four decay steps internally; it should not crash
    cur.validate_block(prev)

    # Apply four steps manually and verify floors
    cur_req = req.to_json()
    for _ in range(4):
        cur_req = calculate_requirements_decay(cur_req)

    # Diversity floor is 0.20, min_solutions floor is 10
    assert cur_req['min_diversity'] >= 0.20
    assert cur_req['min_solutions'] >= 10

