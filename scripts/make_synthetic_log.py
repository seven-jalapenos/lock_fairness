"""Synthetic binary logs with known ground truth, for verifying the analysis
pipeline without running the benchmark.

The benchmark can only be run on the target machine (a Release build is
-march=native, thread counts differ, and find_offsets() burns 10M iterations per
core on startup), so metric changes can't be validated by sweeping on a dev box.
The binary log format is simple enough to synthesize directly -- per thread, a
size_t count followed by that many (invocation, acquisition, release) u64 triples
-- which allows fixtures whose correct metric values are known by construction
rather than eyeballed.

Run from the repo root:

    python -m scripts.make_synthetic_log
"""

import math
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

from analysis import LogParser, LogAnalyzer, DataExporter, MetricAverager, StatsExporter
from analysis import create_global_timeline
from analysis.log_analyzer import AMBIGUITY_CYCLES, COVERAGE_WARN_THRESHOLD
from analysis.log_parser import core_for_thread

# Start well away from zero so anything that accidentally treats a timestamp as
# a duration shows up as an absurd number rather than a plausible one.
BASE = 1 << 40

OUT_ROOT = Path('files/synthetic')


#############################
#
#        LOG WRITING
#
#############################

def write_log(path: Path, per_thread_events: list) -> None:
    """Write per-thread event lists in the layout dump_logs() produces."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        for events in per_thread_events:
            f.write(struct.pack('<Q', len(events)))
            for inv, acq, rel in events:
                f.write(struct.pack('<QQQ', inv, acq, rel))


def write_offsets(path: Path, num_cores: int, offsets=None) -> None:
    """Write an rdtsc_offsets.txt in the format find_offsets() emits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for core in range(num_cores):
            value = 0 if offsets is None else offsets[core]
            f.write(f"Core {core}: {value} cycles\n")


#############################
#
#         FIXTURES
#
#############################

def fair_rotation(num_threads: int, rounds: int, hold: int = 2, gap: int = 3) -> list:
    """Strict round-robin with no overlapping invocations.

    Nothing is ever pending while another thread acquires, so the overtake count
    is exactly zero, and every thread completes the same number of operations."""
    per_thread = [[] for _ in range(num_threads)]
    t = BASE
    for _ in range(rounds):
        for tid in range(num_threads):
            inv = t
            acq = t + 1
            rel = acq + hold
            per_thread[tid].append((inv, acq, rel))
            t = rel + gap
    return per_thread


def starved_pair(rounds: int, share: int = 3, hold: int = 2, gap: int = 3) -> list:
    """Two threads where thread 0 takes `share` operations for every one of
    thread 1's, and takes them consecutively.

    With share=3 the completion counts are (3R, R), so Jain's index is
    16R^2 / (2 * 10R^2) = 0.8 exactly and the throughput ratio is exactly 3."""
    per_thread = [[], []]
    t = BASE
    for _ in range(rounds):
        for _ in range(share):
            per_thread[0].append((t, t + 1, t + 1 + hold))
            t = t + 1 + hold + gap
        per_thread[1].append((t, t + 1, t + 1 + hold))
        t = t + 1 + hold + gap
    return per_thread


def overtaking_pair(rounds: int, gap: int, spacing: int = 100000) -> list:
    """Two threads where thread 1 invokes after thread 0 but acquires first.

    Thread 0 is overtaken exactly once per round and thread 1 never is, so the
    overtake percentage is exactly 50%. `gap` is the invocation separation the
    overtake decision rests on -- below AMBIGUITY_CYCLES it should be reported as
    ambiguous."""
    a, b = [], []
    t = BASE
    for _ in range(rounds):
        a_inv = t
        b_inv = t + gap
        b_acq = b_inv + 10
        b_rel = b_acq + 2
        a_acq = b_rel + 10
        a_rel = a_acq + 2
        a.append((a_inv, a_acq, a_rel))
        b.append((b_inv, b_acq, b_rel))
        t = a_rel + spacing
    return [a, b]


def truncated_rotation(num_threads: int, rounds: int) -> list:
    """A fair rotation in which thread 0's log stops halfway, as a filled buffer
    would leave it."""
    per_thread = fair_rotation(num_threads, rounds)
    per_thread[0] = per_thread[0][: len(per_thread[0]) // 2]
    return per_thread


#############################
#
#          HARNESS
#
#############################

class Checker:
    def __init__(self):
        self.failures = []
        self.checks = 0

    def check(self, label: str, ok: bool, detail: str = '') -> None:
        self.checks += 1
        if ok:
            print(f"  PASS  {label}")
        else:
            print(f"  FAIL  {label}  {detail}")
            self.failures.append(label)

    def close_to(self, label: str, got, want, tol=1e-9) -> None:
        ok = got is not None and math.isfinite(got) and abs(got - want) <= tol
        self.check(label, ok, f"(got {got!r}, want {want!r})")


def analyzer_for(per_thread_events, pin: int, num_cores: int = 8,
                 offsets=None, tag: str = 'fixture'):
    """Write a fixture to disk and push it through parse -> timeline -> analyze."""
    log_path = OUT_ROOT / 'logs' / f'{tag}.bin'
    offset_path = OUT_ROOT / 'rdtsc_offsets.txt'
    write_log(log_path, per_thread_events)
    write_offsets(offset_path, num_cores, offsets)

    parser = LogParser(str(log_path), str(offset_path), pin)
    data = parser.all_threads_data
    timeline = create_global_timeline(data)
    return LogAnalyzer(data, timeline)


def check_pin_mapping(c: Checker) -> None:
    """core_for_thread must agree with main.cpp's core_ids assignment.

    Transcribed from src/runner/main.cpp rather than imported, so the two really
    are independent statements of the same map. This is the only thing keeping
    the calibration honest -- if they disagree, every cross-thread timestamp is
    silently shifted by the wrong offset."""

    def cpp_core_ids(num_threads: int, nproc: int, pin: int) -> list:
        core_ids = [-1] * num_threads
        hot_core = nproc // 2
        if pin == 1:
            for i in range(num_threads):
                core_ids[i] = i % nproc
        elif pin == 2:
            for i in range(num_threads):
                core_ids[i] = hot_core
        elif pin == 3:
            half = num_threads // 2
            for i in range(half):
                core_ids[i] = hot_core
            for i in range(half, num_threads):
                if nproc <= 1:
                    core_ids[i] = 0
                    continue
                slot = (i - half) % (nproc - 1)
                core_ids[i] = slot if slot < hot_core else slot + 1
        return core_ids

    mismatches = []
    collisions = []
    for nproc in (1, 2, 8, 28):
        for num_threads in range(1, 57):
            for pin in (1, 2, 3):
                expected = cpp_core_ids(num_threads, nproc, pin)
                for tid in range(num_threads):
                    got = core_for_thread(tid, num_threads, nproc, pin)
                    if got != expected[tid]:
                        mismatches.append((nproc, num_threads, pin, tid, got, expected[tid]))

                # The point of policy 3 is a hot group and a cold group. If a
                # "cold" thread lands on the hot core the contrast is diluted.
                if pin == 3 and nproc > 1:
                    hot_core = nproc // 2
                    half = num_threads // 2
                    cold = expected[half:]
                    if any(core == hot_core for core in cold):
                        collisions.append((nproc, num_threads))

    c.check("pin map: Python matches main.cpp for all (nproc, threads, policy)",
            not mismatches, f"first mismatch: {mismatches[0] if mismatches else None}")
    c.check("pin map: policy 3 cold half never lands on the hot core",
            not collisions, f"collisions at: {collisions[:3]}")
    c.check("pin map: policy 0 is reported as unpinned",
            core_for_thread(0, 8, 8, 0) is None)


def check_jain_closed_form(c: Checker) -> None:
    """Jain's index against values computed by hand."""
    c.close_to("jain([1,1]) == 1.0", LogAnalyzer.jain_index(np.array([1.0, 1.0])), 1.0)
    c.close_to("jain([3,1]) == 0.8", LogAnalyzer.jain_index(np.array([3.0, 1.0])), 0.8)
    c.close_to("jain([1,0]) == 0.5", LogAnalyzer.jain_index(np.array([1.0, 0.0])), 0.5)
    c.close_to("jain([1,1,1,1]) == 1.0", LogAnalyzer.jain_index(np.array([1.0] * 4)), 1.0)
    c.close_to("jain([1,0,0,0]) == 0.25", LogAnalyzer.jain_index(np.array([1.0, 0, 0, 0])), 0.25)


def check_fair_rotation(c: Checker) -> None:
    print("\nfixture: fair rotation (4 threads, strict round robin)")
    a = analyzer_for(fair_rotation(4, 500), pin=1, tag='fair')

    c.close_to("overtake_percentage == 0", a.overtake_percentage()[0], 0.0)
    c.close_to("throughput_jain_index == 1", a.throughput_jain_index(), 1.0)
    c.close_to("throughput_ratio == 1", a.throughput_ratio(), 1.0)
    c.close_to("self_transfer_rate == 0", a.self_transfer_rate(), 0.0)
    # Strict rotation always yields the same successor, so handoff is fully
    # determined -- low entropy is correct here, not a bug.
    c.close_to("transfer_entropy == 0 (deterministic rotation)", a.transfer_entropy(), 0.0)
    c.close_to("wait_time_cov == 0", a.wait_time_cov(), 0.0)
    c.close_to("ordering_ambiguity_fraction == 0", a.ordering_ambiguity_fraction(), 0.0)
    c.check("log_coverage above warn threshold",
            a.log_coverage() >= COVERAGE_WARN_THRESHOLD, f"(got {a.log_coverage()})")
    wj = a.windowed_jain(10**5)
    c.check("windowed_jain(1e5) >= 0.9", wj >= 0.9, f"(got {wj})")
    c.close_to("total_CS_completions == 2000", float(a.total_CS_completions()), 2000.0)


def check_starvation(c: Checker) -> None:
    print("\nfixture: starved pair (thread 0 takes 3 of every 4)")
    a = analyzer_for(starved_pair(400), pin=1, tag='starved')

    c.close_to("throughput_jain_index == 0.8", a.throughput_jain_index(), 0.8, tol=1e-12)
    c.close_to("throughput_ratio == 3", a.throughput_ratio(), 3.0)
    # Two of every three of thread 0's acquisitions follow its own release.
    st = a.self_transfer_rate()
    c.check("self_transfer_rate > 0.4 (barging visible)", st > 0.4, f"(got {st})")


def check_overtakes(c: Checker) -> None:
    print("\nfixture: overtaking pair, wide invocation gap")
    wide = analyzer_for(overtaking_pair(300, gap=AMBIGUITY_CYCLES * 25),
                        pin=1, tag='overtake_wide')
    c.close_to("overtake_percentage == 50", wide.overtake_percentage()[0], 50.0)
    c.close_to("average_overtake_depth == 1", wide.average_overtake_depth()[0], 1.0)
    c.close_to("ordering_ambiguity_fraction == 0 (gap well above noise floor)",
               wide.ordering_ambiguity_fraction(), 0.0)
    d = wide.overtake_depth_percentiles()
    c.close_to("overtake_depth_max == 1", d['overtake_depth_max'], 1.0)
    c.close_to("overtake_depth_max_normalized == 1 (2 threads)",
               d['overtake_depth_max_normalized'], 1.0)

    print("\nfixture: overtaking pair, gap inside the calibration noise floor")
    tight = analyzer_for(overtaking_pair(300, gap=AMBIGUITY_CYCLES // 4),
                         pin=1, tag='overtake_tight')
    c.close_to("overtake_percentage still == 50", tight.overtake_percentage()[0], 50.0)
    c.close_to("ordering_ambiguity_fraction == 1 (every decision is noise)",
               tight.ordering_ambiguity_fraction(), 1.0)


def check_truncation(c: Checker) -> None:
    print("\nfixture: truncated log (thread 0's buffer fills halfway)")
    a = analyzer_for(truncated_rotation(4, 500), pin=1, tag='truncated')
    cov = a.log_coverage()
    c.check("log_coverage below warn threshold",
            math.isfinite(cov) and cov < COVERAGE_WARN_THRESHOLD, f"(got {cov})")
    c.check("log_coverage roughly halves", 0.4 < cov < 0.6, f"(got {cov})")


def check_calibration(c: Checker) -> None:
    """Every pinning policy must parse, and a non-zero offset must actually move
    the timestamps by that offset."""
    print("\ncalibration: all pinning policies")
    for pin in (0, 1, 2, 3):
        try:
            a = analyzer_for(fair_rotation(4, 50), pin=pin, tag=f'pin{pin}')
            c.check(f"pin policy {pin} parses", a.total_CS_completions() == 200)
        except Exception as e:  # noqa: BLE001 - the point is to report, not raise
            c.check(f"pin policy {pin} parses", False, f"raised {e!r}")

    # 8 cores, so policy 2 puts every thread on core 4. Give core 4 a known
    # offset and confirm the shift lands.
    offsets = [0] * 8
    offsets[4] = 12345
    base = analyzer_for(fair_rotation(2, 10), pin=2, num_cores=8,
                        offsets=[0] * 8, tag='cal_zero')
    shifted = analyzer_for(fair_rotation(2, 10), pin=2, num_cores=8,
                           offsets=offsets, tag='cal_offset')
    delta = int(base._data['invocation'].iloc[0]) - int(shifted._data['invocation'].iloc[0])
    c.close_to("policy 2 subtracts the hot core's offset", float(delta), 12345.0)


def check_end_to_end(c: Checker) -> None:
    """Averaging, CSV export and plotting over a two-rep run directory."""
    print("\nend to end: averager -> exporter -> plotters")
    from analysis import SingleRunPlotter

    csv_root = OUT_ROOT / 'pqt'
    fig_root = OUT_ROOT / 'figures'
    if csv_root.exists():
        shutil.rmtree(csv_root)
    if fig_root.exists():
        shutil.rmtree(fig_root)

    run_dir = csv_root / 'mcs_4_1_w1000'
    for rep in range(2):
        a = analyzer_for(fair_rotation(4, 300), pin=1, tag=f'e2e_{rep}')
        exporter = DataExporter(a._data, a._global_timeline, str(run_dir), str(rep))
        exporter.write_raw()
        exporter.close()

    averager = MetricAverager(run_dir).build_table()
    stats = averager.find_means_and_stds()
    StatsExporter(run_dir).export(stats)

    summary = run_dir / 'summary_scalar_metrics.csv'
    c.check("summary_scalar_metrics.csv written", summary.exists())
    text = summary.read_text() if summary.exists() else ''

    for gone in ('percent_time_in_CS', 'rank_inversion_penalty'):
        c.check(f"cut metric absent: {gone}", gone not in text)

    required = [
        'throughput_ops_per_Mcycle', 'throughput_jain_index', 'throughput_ratio',
        'wait_time_cov', 'mean_hold_time', 'log_coverage',
        'ordering_ambiguity_fraction', 'self_transfer_rate', 'transfer_entropy',
        'wait_p50', 'wait_p99', 'wait_max',
        'overtake_depth_p99_normalized', 'average_overtake_depth_normalized',
        'windowed_jain_1e5', 'windowed_jain_1e6', 'windowed_jain_1e7',
        'overtake_percentage', 'average_wait_time', 'total_CS_completions',
    ]
    missing = [m for m in required if m not in text]
    c.check("all new scalars exported", not missing, f"missing: {missing}")

    c.check("per_thread_throughput.csv written",
            (run_dir / 'per_thread_throughput.csv').exists())
    c.check("lock_transfer_matrix.csv written",
            (run_dir / 'lock_transfer_matrix.csv').exists())

    SingleRunPlotter(run_dir, run_dir / 'figures').plot_all()
    c.check("single-run figures rendered",
            any((run_dir / 'figures').glob('*.png')))

    from scripts.plot_all import plot_cross_runs
    plot_cross_runs(csv_root, fig_root)
    pngs = sorted(p.name for p in fig_root.glob('*.png'))
    c.check("cross-run figures rendered", bool(pngs), f"(found {len(pngs)})")
    c.check("cross-run filenames carry the pin policy",
            any('_p1' in name for name in pngs), f"(found {pngs[:3]})")
    c.check("timescale figure rendered",
            any('windowed_jain_by_window' in name for name in pngs))


def main() -> int:
    c = Checker()
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    print("pin mapping")
    check_pin_mapping(c)
    print("\njain closed forms")
    check_jain_closed_form(c)
    check_fair_rotation(c)
    check_starvation(c)
    check_overtakes(c)
    check_truncation(c)
    check_calibration(c)
    check_end_to_end(c)

    print(f"\n{c.checks - len(c.failures)}/{c.checks} checks passed")
    if c.failures:
        print("FAILED:")
        for name in c.failures:
            print(f"  - {name}")
        return 1
    print("all checks passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())
