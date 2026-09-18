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


def silent_thread(num_threads: int, rounds: int) -> list:
    """A fair rotation among all but the highest-numbered thread, which never
    acquires and so logs nothing at all.

    This is the shape a starved thread leaves behind: it writes a zero-count
    block, contributes no rows, and is invisible to anything that counts the
    thread ids present in the data. With one silent thread out of n the
    completion counts are (c,...,c,0), so Jain's index is
    ((n-1)c)^2 / (n * (n-1)c^2) = (n-1)/n exactly."""
    per_thread = fair_rotation(num_threads - 1, rounds)
    per_thread.append([])
    return per_thread


def random_contention(num_threads: int, grants: int, seed: int = 0) -> list:
    """A messy but physically valid log: threads are granted in a random order
    with varying waits, and no thread ever has two operations in flight.

    Exists to exercise the overtake scan on something with a nontrivial mix of
    depths, rather than the hand-built fixtures whose answers are round numbers.
    """
    rng = np.random.default_rng(seed)
    ready = np.full(num_threads, BASE, dtype=np.int64)
    per_thread = [[] for _ in range(num_threads)]
    clock = BASE
    for _ in range(grants):
        waiting = np.flatnonzero(ready <= clock)
        if waiting.size == 0:
            clock = int(ready.min())
            waiting = np.flatnonzero(ready <= clock)
        tid = int(waiting[rng.integers(0, waiting.size)])
        # A real operation always has a nonzero wait: two TSC reads and an lfence
        # sit between the invocation and acquisition timestamps.
        clock = max(clock, int(ready[tid]) + 1)
        hold = int(rng.integers(2, 40))
        per_thread[tid].append((int(ready[tid]), clock, clock + hold))
        ready[tid] = clock + hold + int(rng.integers(1, 30))
        clock += hold + int(rng.integers(1, 10))
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
    # The fixture knows how many threads it wrote; a silent thread contributes no
    # rows, so the count cannot be recovered from `data`.
    return LogAnalyzer(data, timeline, num_threads=len(per_thread_events))


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


def check_silent_thread(c: Checker) -> None:
    """A thread that completes nothing must still be counted.

    It is the single most consequential case in the whole pipeline: inferring the
    thread count from the data drops exactly the starved threads, which removes
    their zero from per_thread_throughput and reports the least fair run possible
    as perfectly fair."""
    print("\nfixture: 4 threads, the highest-numbered one never acquires")
    a = analyzer_for(silent_thread(4, 400), pin=1, tag='silent')

    c.check("num_threads == 4 despite thread 3 logging nothing",
            a.num_threads == 4, f"(got {a.num_threads})")
    counts = a.per_thread_throughput()
    c.check("per_thread_throughput has 4 entries", len(counts) == 4,
            f"(got {len(counts)})")
    c.check("the silent thread's entry is 0", counts[-1] == 0, f"(got {counts[-1]})")
    # (n-1)/n for one silent thread out of n.
    c.close_to("throughput_jain_index == 0.75", a.throughput_jain_index(), 0.75,
               tol=1e-12)
    ratio = a.throughput_ratio()
    c.check("throughput_ratio is nan (a thread completed nothing)",
            math.isnan(ratio), f"(got {ratio})")


def check_ragged_reps(c: Checker) -> None:
    """Reps of one combination must agree on their per-thread array length.

    When one rep starves a thread and another doesn't, an inferred thread count
    differs between them and the np.stack in find_means_and_stds raises, taking
    down the averaging pass for the whole run directory."""
    print("\nend to end: reps that disagree on which threads logged")
    run_dir = OUT_ROOT / 'pqt_ragged' / 'ttas_4_1_w1000'
    if run_dir.exists():
        shutil.rmtree(run_dir)

    for rep, events in enumerate((fair_rotation(4, 300), silent_thread(4, 300))):
        a = analyzer_for(events, pin=1, tag=f'ragged_{rep}')
        exporter = DataExporter(a._data, a._global_timeline, str(run_dir), str(rep))
        exporter.write_raw()
        exporter.close()

    try:
        averager = MetricAverager(run_dir).build_table()
        stats = averager.find_means_and_stds()
        c.check("averaging survives a rep with a silent thread", True)
        c.check("thread count taken from the directory name",
                averager.thread_count == 4, f"(got {averager.thread_count})")
        c.check("per-thread throughput stats cover all 4 threads",
                len(stats['per_thread_throughput']) == 4)
    except Exception as e:  # noqa: BLE001 - reporting is the point
        c.check("averaging survives a rep with a silent thread", False,
                f"raised {e!r}")


def _reference_overtake_timeline(timeline):
    """The per-event scan the vectorized implementation replaced.

    Kept here as an oracle: the rewrite is only reviewable if the two are checked
    against each other on every fixture."""
    from analysis.defs import INVOCATION, ACQUISITION
    import pandas as pd

    if timeline.empty:
        return pd.DataFrame(columns=['invocation_time', 'thread_id',
                                     'intervening_acquisitions',
                                     'ambiguous_acquisitions'])
    ts = timeline['timestamp'].to_numpy()
    ev = timeline['event_type'].to_numpy()
    tid = timeline['thread_id'].to_numpy()
    pending = {}
    results = []
    for timestamp, event_type, thread_id in zip(ts, ev, tid):
        if event_type == INVOCATION:
            results.append([timestamp, thread_id, 0, 0])
            pending[thread_id] = (timestamp, len(results) - 1)
        elif event_type == ACQUISITION and thread_id in pending:
            acq_inv_time, _ = pending.pop(thread_id)
            for other_inv_time, other_idx in pending.values():
                if acq_inv_time > other_inv_time:
                    results[other_idx][2] += 1
                    if acq_inv_time - other_inv_time <= AMBIGUITY_CYCLES:
                        results[other_idx][3] += 1
    return pd.DataFrame(results, columns=['invocation_time', 'thread_id',
                                          'intervening_acquisitions',
                                          'ambiguous_acquisitions'])


def check_overtake_equivalence(c: Checker) -> None:
    """The vectorized overtake scan must reproduce the per-event scan exactly."""
    print("\novertake scan: vectorized vs. the per-event reference")
    fixtures = [
        ('fair rotation', fair_rotation(4, 300)),
        ('starved pair', starved_pair(300)),
        ('overtaking pair, wide gap', overtaking_pair(200, gap=AMBIGUITY_CYCLES * 25)),
        ('overtaking pair, tight gap', overtaking_pair(200, gap=AMBIGUITY_CYCLES // 4)),
        ('random contention, 8 threads', random_contention(8, 1500, seed=1)),
        ('random contention, 24 threads', random_contention(24, 3000, seed=2)),
    ]
    for name, events in fixtures:
        a = analyzer_for(events, pin=1, tag=f'equiv_{abs(hash(name)) % 10**6}')
        want = _reference_overtake_timeline(a._global_timeline)
        got = a._overtake_timeline

        if len(want) != len(got):
            c.check(f"{name}: same row count", False,
                    f"(got {len(got)}, want {len(want)})")
            continue
        # Both are emitted in invocation order; equal invocation timestamps may
        # order differently between them, so compare as sorted multisets.
        cols = list(want.columns)
        w = want.sort_values(cols, ignore_index=True)
        g = got.sort_values(cols, ignore_index=True)
        mismatched = [col for col in cols
                      if not np.array_equal(w[col].to_numpy(), g[col].to_numpy())]
        c.check(f"{name}: identical to the reference scan", not mismatched,
                f"(columns differing: {mismatched})")
        c.check(f"{name}: dtypes preserved",
                list(w.dtypes) == list(g.dtypes),
                f"({list(g.dtypes)} vs {list(w.dtypes)})")


def check_offsets_parsing(c: Checker) -> None:
    """Offset files must survive a negative offset, however it was written."""
    print("\ncalibration: offset file parsing")
    from analysis.parse_offsets import parse_tsc_offsets

    path = OUT_ROOT / 'offsets_legacy.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    # find_offsets used to compute offsets unsigned and print them with %lu, so a
    # core whose TSC trailed core 0 landed in the file as its 2^64 complement.
    # Reading that into an int64 array raises OverflowError outright.
    path.write_text(
        "Core 0: 0 cycles\n"
        f"Core 1: {(1 << 64) - 500} cycles\n"
        "Core 2: -250 cycles\n"
        f"Core 3: {(1 << 64) - 1} cycles\n"
    )
    try:
        parsed = parse_tsc_offsets(str(path))
        c.check("legacy unsigned-wrapped offset file parses", parsed is not None)
        if parsed is not None:
            c.check("wrapped value folds to its negative", parsed[1] == -500,
                    f"(got {parsed[1]})")
            c.check("already-signed value passes through", parsed[2] == -250,
                    f"(got {parsed[2]})")
            c.check("old UINT64_MAX sentinel folds to -1", parsed[3] == -1,
                    f"(got {parsed[3]})")
    except Exception as e:  # noqa: BLE001 - reporting is the point
        c.check("legacy unsigned-wrapped offset file parses", False,
                f"raised {e!r}")


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
        'windowed_jain_k10n',
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


def check_windowed_jain_by_count(c: Checker) -> None:
    """Fixed-count windows on fixtures whose per-window split is known exactly."""
    print("\nfixture: fixed-count windowed Jain")

    # 2000 acquisitions, k = 10*4 = 40, so 50 whole windows each holding exactly
    # ten operations per thread.
    a = analyzer_for(fair_rotation(4, 500), pin=1, tag='count_fair')
    c.close_to("windowed_jain_by_count(10): strict rotation -> 1.0",
               a.windowed_jain_by_count(10), 1.0)

    # 1600 acquisitions of a repeating (0,0,0,1), k = 20. The exactness here is a
    # coincidence worth not generalizing: k is a whole multiple of the fixture's
    # period-4 interleave, so every window sees the same 15:5 split and the mean
    # of per-window Jain equals the aggregate. Jain is concave, so for a k that
    # straddled the period this would need a tolerance instead.
    b = analyzer_for(starved_pair(400), pin=1, tag='count_starved')
    c.close_to("windowed_jain_by_count(10): 3:1 split -> 0.8",
               b.windowed_jain_by_count(10), 0.8)

    # Three threads rotating with a fourth silent, k = 40. 40 = 3*13 + 1, so
    # every window is (14,13,13,0) and the value is 1600/2136 -- just under the
    # 0.75 the aggregate index reports, because a window can't split 40 three
    # ways evenly.
    d = analyzer_for(silent_thread(4, 400), pin=1, tag='count_silent')
    c.close_to("windowed_jain_by_count(10): one silent of four -> 1600/2136",
               d.windowed_jain_by_count(10), 1600 / 2136)

    # Fewer acquisitions than one window holds. A short group isn't a smaller
    # window under this definition, so there is nothing to report.
    tiny = analyzer_for(fair_rotation(4, 1), pin=1, tag='count_tiny')
    wj = tiny.windowed_jain_by_count(10)
    c.check("windowed_jain_by_count(10) is nan below one full window",
            math.isnan(wj), f"(got {wj})")


def check_acquisition_sequence_equivalence(c: Checker) -> None:
    """windowed_jain_by_count reads the flat frame while windowed_jain reads the
    melted timeline; they must see the same acquisition order."""
    print("\nacquisition order: flat frame vs melted timeline")
    from analysis.defs import ACQUISITION

    a = analyzer_for(random_contention(8, 1500, seed=3), pin=1, tag='count_equiv')
    from_data = (a._data.sort_values('acquisition', kind='stable')['thread_id']
                 .to_numpy())
    timeline = a._global_timeline
    from_timeline = (timeline[timeline['event_type'] == ACQUISITION]
                     .sort_values('timestamp', kind='stable')['thread_id']
                     .to_numpy())
    c.check("same acquisition-ordered thread sequence either way",
            np.array_equal(from_data, from_timeline),
            f"(lengths {from_data.size} vs {from_timeline.size})")


def check_scalar_merge(c: Checker) -> None:
    """A metric-only recompute must merge into the summary CSV, not rewrite it.

    export() takes whatever dict it is handed and truncates the file, so the
    partial path writing through it would silently drop every other scalar the
    full pass produced. That is the one new way this can lose data."""
    print("\nend to end: partial recompute merges without clobbering")
    import csv as csv_module
    from scripts.all_metrics import average_all_metrics, update_all_metrics

    csv_root = OUT_ROOT / 'pqt_merge'
    if csv_root.exists():
        shutil.rmtree(csv_root)

    run_dir = csv_root / 'mcs_4_1_w1000'
    for rep in range(2):
        a = analyzer_for(fair_rotation(4, 300), pin=1, tag=f'merge_{rep}')
        exporter = DataExporter(a._data, a._global_timeline, str(run_dir), str(rep))
        exporter.write_raw()
        exporter.close()

    average_all_metrics(csv_root)

    summary = run_dir / 'summary_scalar_metrics.csv'

    def read_rows():
        with open(summary, newline='') as f:
            rows = list(csv_module.reader(f))
        return {r[0]: (r[1], r[2]) for r in rows[1:]}

    before = read_rows()
    c.check("full pass writes windowed_jain_k10n too",
            'windowed_jain_k10n' in before)

    update_all_metrics(csv_root, names=['windowed_jain_k10n'])
    after = read_rows()

    changed = [k for k in before
               if k != 'windowed_jain_k10n' and before[k] != after.get(k)]
    c.check("every other metric survives the partial update unchanged",
            not changed, f"(changed: {changed})")
    c.check("no rows dropped or duplicated",
            len(before) == len(after), f"(before {len(before)}, after {len(after)})")
    c.check("windowed_jain_k10n still present after the update",
            'windowed_jain_k10n' in after)
    c.check("atomic write leaves no temp file behind",
            not list(run_dir.glob('.summary_scalar_metrics.csv.tmp*')))

    # A fair rotation scores 1.0 whichever path computed it, so the merged value
    # must match what the full pass wrote rather than merely being present.
    c.close_to("merged value matches the full pass",
               float(after['windowed_jain_k10n'][0]),
               float(before['windowed_jain_k10n'][0]))

    try:
        MetricAverager(run_dir).recompute_scalars(['no_such_metric'])
    except ValueError:
        c.check("unknown metric name rejected", True)
    else:
        c.check("unknown metric name rejected", False, "(no ValueError)")


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
    check_silent_thread(c)
    check_windowed_jain_by_count(c)
    check_acquisition_sequence_equivalence(c)
    check_overtakes(c)
    check_overtake_equivalence(c)
    check_truncation(c)
    check_calibration(c)
    check_offsets_parsing(c)
    check_end_to_end(c)
    check_scalar_merge(c)
    check_ragged_reps(c)

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
