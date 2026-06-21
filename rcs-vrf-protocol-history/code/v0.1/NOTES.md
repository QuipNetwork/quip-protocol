# v0.1 — Two-sided F_XEB check

**Single substantive change vs v0:** F_XEB acceptance check became two-sided.

## Files in this version

- `rcs_vrf.py` — protocol with two-sided F_XEB acceptance check
- `spoofing_experiment.py` — extended to demonstrate upper-threshold attacks (cherry-picking heavy outputs)

## What was new here

Acceptance changed from `F_XEB ≥ χ` (one-sided, Liu's prescription) to `χ_low ≤ F_XEB ≤ χ_high` (two-sided, our addition).

## Why

Defence-in-depth against adversaries outside Liu's compute-rational model — specifically, attackers who score anomalously high via pure heavy-output generation.

## Honest framing

This is **our addition**, not Liu's prescription. Liu's adversary model doesn't require it (a rational adversary wouldn't waste compute scoring ≫1). The lower bound is load-bearing; the upper bound is our conservative belt-and-suspenders.

## What carries forward

The two-sided check stays in all subsequent versions. Documented explicitly in `docs/presentations/leg5_verification_design.pdf` slide 14.
