
#pragma once

// Measure per-core TSC offsets and write files/rdtsc_offsets.txt.
//
// `force` re-measures unconditionally; otherwise an existing file that already
// covers this machine's core count is reused. The sweep driver recalibrates on
// an interval (scripts/runner.py), so a per-run invocation should not pay for a
// measurement that has just been taken.
void find_offsets(bool force = false);
