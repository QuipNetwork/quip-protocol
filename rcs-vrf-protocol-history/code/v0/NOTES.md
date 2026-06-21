# v0 — Initial RCS+VRF implementation

**First working implementation of the seven-leg protocol.** Uses Aaronson-Hung structure (one circuit, many shots per round).

## Files in this version

- `rcs_vrf.py` — main protocol implementation, seven legs
- `spoofing_experiment.py` — demonstrates F_XEB behaviour under classical spoofing
- `depth_sweep.py` — explores F_XEB vs circuit depth (2-design approximation)

## What was new here vs nothing

Everything. This is the foundation. Subsequent versions modify but don't replace the seven-leg structure introduced here.

## Known limitations (at this stage)

- VRF is a stub (placeholder hash)
- F_XEB check is one-sided (made two-sided in v0.1)
- Bit extraction is a stub (made real in v0.3a)
- Audit log is minimal (made real in v0.2)
- No formal test suite

## Why preserved

Historical reference for the original AH-style implementation. The spoofing experiment is pedagogically useful — it concretely shows why F_XEB > χ is non-trivial against a classical adversary.
