#!/usr/bin/env python3
"""Calibrated v10 rerun using measured initial mutual-distance statistics.

Measured over 25,475 mutual-nearest pairs from 24 random initializations:
- 0.1% quantile ~= 0.002385
- only 4 pairs were below 0.001
Therefore the geometric meeting radius is tightened to 0.001 with a 0.00025
soft boundary. This keeps links genuinely dormant until points nearly coincide.
No reward/punishment signal enters routing; all other v10 contracts are unchanged.
"""
import cpu_v10_onion_sparse_intersection_deepmind as v10

v10.MEET_RADIUS = 0.001
v10.MEET_TEMP = 0.00025

if __name__ == "__main__":
    v10.main()
