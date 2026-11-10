# ADR-001: Target Machines

## Status
Pending — fill in with actual team hardware

## Context
ADHIE targets consumer hardware with limited VRAM. We need to document
available machines so the planner can be tested across different profiles.

## Decision
Fill in after hardware inventory.

### Primary Machine (lowest VRAM — main test target)
- **GPU:** _______________ (VRAM: ___ GB)
- **CPU:** _______________ (Cores: ___ physical / ___ logical)
- **RAM:** ___ GB total
- **OS:** _______________
- **CUDA:** _______________
- **Storage:** SSD / HDD

### Fallback Machine (higher capability)
- **GPU:** _______________
- **CPU:** _______________
- **RAM:** _______________
- **OS:** _______________
- **CUDA:** _______________

## Notes
- Profile both machines with `adhie profile` once Module 1 is built.
- The primary machine should have ≤ 6 GB VRAM to test constrained scenarios.

