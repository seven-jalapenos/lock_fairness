import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, Tuple, Optional

from .defs import INVOCATION, ACQUISITION

# Two invocations closer together than the residual cross-core TSC skew can't be
# reliably ordered, so an "overtake" that hinges on a gap this small is
# measurement noise rather than a fairness violation. find_offsets() takes a min
# over 10M samples, which leaves roughly an inter-core cache-line round trip.
# Reported as ordering_ambiguity_fraction so the noise floor is visible instead
# of assumed; tune per machine.
AMBIGUITY_CYCLES = 200

# Window sizes for the fairness-vs-timescale metric, in TSC cycles (~30us, ~300us
# and ~3ms at 3GHz). A lock can be fair over a 10s run and badly bursty inside
# any given millisecond; one number at one timescale can't tell those apart.
WINDOW_SIZES_CYCLES = (10**5, 10**6, 10**7)

# Suffixes used to name the per-window scalars in the exported CSV.
WINDOW_LABELS = {10**5: '1e5', 10**6: '1e6', 10**7: '1e7'}

# log_coverage is never exactly 1.0 on a healthy run: threads are started and
# joined in sequence, so the last thread's first invocation trails the first
# thread's, and the same staggering happens at the end. That costs a fraction of
# a percent on a 10s run. Warn below this instead of below 1.0, or every run
# reports itself truncated.
COVERAGE_WARN_THRESHOLD = 0.99


class LogAnalyzer:

    def __init__(self, data: pd.DataFrame, global_timeline: pd.DataFrame, overtake_timeline: Optional[pd.DataFrame]=None):
        self._data = data
        self._global_timeline = global_timeline
        self._overtake_timeline = overtake_timeline if overtake_timeline is not None else self.create_overtake_timeline()
        self.num_threads = data['thread_id'].values[-1] + 1  # Assuming thread IDs are 0-indexed and contiguous
        self.operation_count = len(data)
        self.event_count = len(global_timeline)

    def close(self) -> None:
        self._data = None
        self._global_timeline = None
        self._overtake_timeline = None

    def create_overtake_timeline(self) -> pd.DataFrame:
        """
        Creates an operation-wise timeline of overtakes.
        O(N*t) complexity using a chronological state-tracker.
        """
        assert(self._global_timeline is not None)
        timeline = self._global_timeline

        if timeline.empty:
            return pd.DataFrame(columns=['invocation_time', 'thread_id',
                                         'intervening_acquisitions',
                                         'ambiguous_acquisitions'])

        # 1. Drop out of Pandas into native NumPy arrays to eliminate loop overhead.
        # Extract each column separately so every array keeps its own dtype: a
        # single to_numpy() over mixed uint64/int8/int64 columns would coerce the
        # whole thing to float64 (losing TSC precision) or object (huge). Assumes
        # the timeline is already chronologically sorted by timestamp.
        ts = timeline['timestamp'].to_numpy()
        ev = timeline['event_type'].to_numpy()
        tid = timeline['thread_id'].to_numpy()

        # State trackers
        pending_invocations = {}  # thread_id -> (invocation_time, result_index)
        results = []              # [invocation_time, thread_id, overtakes, ambiguous]

        # 2. Step through the chronological timeline O(N)
        for timestamp, event_type, thread_id in zip(ts, ev, tid):

            if event_type == INVOCATION:
                # Save its future index in the results array so we can increment its penalty later
                idx = len(results)
                results.append([timestamp, thread_id, 0, 0])
                pending_invocations[thread_id] = (timestamp, idx)

            elif event_type == ACQUISITION:
                # The thread acquired the lock, remove it from the pending pool
                if thread_id in pending_invocations:
                    acq_inv_time, _ = pending_invocations.pop(thread_id)

                    # 3. Check who it overtook O(t)
                    # Iterate through the remaining pending threads (max size = t - 1)
                    for other_inv_time, other_idx in pending_invocations.values():

                        # If the acquiring thread invoked AFTER the pending thread,
                        # it means the pending thread was overtaken.
                        if acq_inv_time > other_inv_time:
                            results[other_idx][2] += 1
                            # ...but a gap under the calibration noise floor means
                            # we can't actually tell who invoked first.
                            if acq_inv_time - other_inv_time <= AMBIGUITY_CYCLES:
                                results[other_idx][3] += 1

        # 4. Rebuild the final dataframe in one shot
        final_ops = pd.DataFrame(results, columns=['invocation_time', 'thread_id',
                                                   'intervening_acquisitions',
                                                   'ambiguous_acquisitions'])

        return final_ops

    def print_overtake(self, file_path: Path) -> None:
        """
        Utility function to print out the overtake timeline to parquet.
        """
        assert(self._overtake_timeline is not None)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self._overtake_timeline.to_parquet(file_path)

#############################
#
#        VALIDITY GUARDS
#
#############################

    def log_coverage(self) -> float:
        """
        Fraction of the observed run spanned by the window in which *every* thread
        was still logging.

        A thread that fills its log buffer goes dark early while the others keep
        recording, which censors completion counts toward whoever survived -- and
        because the favored threads under an unfair lock fill fastest, the bias
        runs in the direction that makes unfair locks look fair. Anything below
        COVERAGE_WARN_THRESHOLD means the run truncated; such runs are invalid,
        not correctable.
        """
        assert(self._data is not None)
        if self._data.empty:
            return float('nan')

        firsts = self._data.groupby('thread_id')['invocation'].min()
        lasts = self._data.groupby('thread_id')['release'].max()

        full_span = int(lasts.max()) - int(firsts.min())
        if full_span <= 0:
            return float('nan')

        common_span = int(lasts.min()) - int(firsts.max())
        if common_span <= 0:
            return 0.0

        return min(1.0, common_span / full_span)

    def ordering_ambiguity_fraction(self) -> float:
        """
        Fraction of counted overtakes whose invocation gap fell below
        AMBIGUITY_CYCLES.

        The overtake metric orders operations by invocation timestamp -- the same
        currency a timestamp lock optimizes -- so its credibility rests on those
        timestamps being comparable across cores. A large value here means the
        metric is reading TSC skew, not fairness.
        """
        assert(self._overtake_timeline is not None)
        if self._overtake_timeline.empty:
            return float('nan')
        if 'ambiguous_acquisitions' not in self._overtake_timeline.columns:
            return float('nan')

        total = float(self._overtake_timeline['intervening_acquisitions'].sum())
        if total == 0:
            return 0.0
        return float(self._overtake_timeline['ambiguous_acquisitions'].sum()) / total

#############################
#
#        AVG METRICS
#
#############################

    def find_avg_per_thread_wait_time(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the average and variance of wait time for each thread as well as number of wait events of each thread.
        Returns: (averages_array, variances_array, counts_array)
        """
        assert(self._data is not None)
        if self._data.empty:
            return np.zeros(self.num_threads), np.zeros(self.num_threads), np.zeros(self.num_threads)

        # 1. Group and calculate both mean and variance simultaneously
        stats = self._data.groupby('thread_id')['wait_time'].agg(['mean', 'var', 'count'])

        # 2. Reindex to ensure all threads (0 to num_threads - 1) are present
        aligned = stats.reindex(range(self.num_threads), fill_value=0.0)

        # 3. Pandas variance (ddof=1) returns NaN if a thread has only 1 data point. Fill with 0.0.
        aligned['var'] = aligned['var'].fillna(0.0)

        return aligned['mean'].to_numpy(), aligned['var'].to_numpy(), aligned['count'].to_numpy()

    def wait_time_percentiles(self) -> Dict[str, float]:
        """
        Tail wait times across every operation in the run.

        Fairness arguments live in the tail: two locks can post identical mean
        wait and differ by 100x at p99, which is the difference between "slower"
        and "starves someone".
        """
        assert(self._data is not None)
        keys = ('wait_p50', 'wait_p90', 'wait_p99', 'wait_p999', 'wait_max')
        if self._data.empty:
            return {k: float('nan') for k in keys}

        waits = self._data['wait_time'].to_numpy().astype(np.float64)
        p50, p90, p99, p999 = np.percentile(waits, [50, 90, 99, 99.9])
        return {
            'wait_p50': float(p50),
            'wait_p90': float(p90),
            'wait_p99': float(p99),
            'wait_p999': float(p999),
            'wait_max': float(waits.max()),
        }

    def mean_hold_time(self) -> float:
        """
        Mean critical-section hold time.

        Not a fairness metric -- a control. At a fixed `work` this should be
        near-identical across locks; if it isn't, the critical section isn't the
        constant every cross-lock comparison assumes it is.
        """
        assert(self._data is not None)
        if self._data.empty:
            return float('nan')
        return float(self._data['hold_time'].mean())

    def overtake_percentage(self) -> Tuple[float, float]:
        """
        Computes the mean percentage and variance of operations that are overtaken.
        Returns: (mean_percentage, variance_of_percentage)
        """
        counts = self._overtake_timeline['intervening_acquisitions']

        if counts.empty:
            return 0.0, 0.0

        # Create a boolean/float series (1.0 for overtaken, 0.0 for not)
        is_overtaken = (counts > 0).astype(float)

        # Mean is the fraction of True values. Multiply by 100 for percentage.
        mean_pct = float(is_overtaken.mean() * 100)

        # Variance of the 0/1 occurrences scaled to percentage
        var_pct = float(is_overtaken.var() * (100 ** 2))

        if pd.isna(var_pct):
            var_pct = 0.0

        return mean_pct, var_pct

    def average_overtake_depth(self) -> Tuple[float, float]:
        """
        Computes the average and variance of the depth of elements with overtake depth > 0.
        Returns: (mean_depth, variance_depth)
        """
        overtake_depths = self._overtake_timeline['intervening_acquisitions']
        positive_depths = overtake_depths[overtake_depths > 0]

        if positive_depths.empty:
            return 0.0, 0.0

        mean_depth = float(positive_depths.mean())
        var_depth = float(positive_depths.var())

        if pd.isna(var_depth):
            var_depth = 0.0

        return mean_depth, var_depth

    def overtake_depth_percentiles(self) -> Dict[str, float]:
        """
        Tail overtake depth, raw and normalized by the number of threads that
        could have done the overtaking.

        Raw depth grows mechanically with thread count -- more concurrent waiters
        means more of them can slip past -- so an un-normalized depth-vs-threads
        curve conflates "less fair" with "more threads". Dividing by (t-1) gives
        the fraction of concurrent waiters bypassed, which is comparable across
        the thread sweep.
        """
        assert(self._overtake_timeline is not None)
        keys = ('overtake_depth_p99', 'overtake_depth_max',
                'overtake_depth_p99_normalized', 'overtake_depth_max_normalized')
        if self._overtake_timeline.empty:
            return {k: float('nan') for k in keys}

        depths = self._overtake_timeline['intervening_acquisitions'].to_numpy().astype(np.float64)
        p99 = float(np.percentile(depths, 99))
        dmax = float(depths.max())

        denom = max(1, int(self.num_threads) - 1)
        return {
            'overtake_depth_p99': p99,
            'overtake_depth_max': dmax,
            'overtake_depth_p99_normalized': p99 / denom,
            'overtake_depth_max_normalized': dmax / denom,
        }

##############################
#
#       PER RUN METRICS
#
##############################

    def total_CS_completions(self) -> int:
        """
        Computes the total number of critical section completions
        """
        return int(len(self._data))

    def observed_span_cycles(self) -> int:
        """TSC cycles between the first invocation and the last release of the run."""
        assert(self._data is not None)
        if self._data.empty:
            return 0
        return int(self._data['release'].max()) - int(self._data['invocation'].min())

    def throughput_ops_per_Mcycle(self) -> float:
        """
        Critical-section completions per million observed TSC cycles.

        Normalized so runs of differing length stay comparable, and expressed in
        cycles rather than seconds so nothing in the pipeline needs to know the
        machine's TSC frequency. This is the cost axis any fairness gain is
        traded against.
        """
        span = self.observed_span_cycles()
        if span <= 0:
            return float('nan')
        return self.total_CS_completions() / (span / 1e6)

    def per_thread_throughput(self) -> np.ndarray:
        """Critical-section completions per thread, indexed 0..num_threads-1."""
        assert(self._data is not None)
        if self._data.empty:
            return np.zeros(self.num_threads)
        counts = self._data.groupby('thread_id').size()
        return counts.reindex(range(self.num_threads), fill_value=0).to_numpy().astype(np.float64)

    @staticmethod
    def jain_index(values: np.ndarray) -> float:
        """
        Jain's fairness index: (sum x)^2 / (n * sum x^2).

        1.0 is a perfectly equal split, 1/n is one thread taking everything. The
        standard scalar in the fairness literature, so it needs no defending in a
        paper the way a bespoke composite would.
        """
        x = np.asarray(values, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float('nan')
        denom = x.size * np.sum(x ** 2)
        if denom == 0:
            return float('nan')
        return float((np.sum(x) ** 2) / denom)

    def throughput_jain_index(self) -> float:
        """Jain's index over per-thread completion counts: the headline fairness number."""
        return self.jain_index(self.per_thread_throughput())

    def throughput_ratio(self) -> float:
        """
        Ratio of the busiest thread's completions to the quietest thread's.

        Jain's index is smooth and bounded; this is the blunt starvation view --
        it blows up when one thread is being locked out even while the rest stay
        even. NaN when a thread completed nothing at all (an unbounded ratio).
        """
        counts = self.per_thread_throughput()
        if counts.size == 0:
            return float('nan')
        lo = float(counts.min())
        if lo <= 0:
            return float('nan')
        return float(counts.max()) / lo

    def wait_time_cov(self) -> float:
        """
        Coefficient of variation across per-thread mean wait times.

        The fairness content of the per-thread wait array is its spread, not its
        level; this reduces that spread to one scale-free number so it can be
        plotted against thread count alongside throughput fairness.
        """
        means, _, counts = self.find_avg_per_thread_wait_time()
        means = means[counts > 0]
        if means.size == 0:
            return float('nan')
        mu = float(means.mean())
        if mu == 0:
            return float('nan')
        return float(means.std(ddof=0)) / mu

    def windowed_jain(self, window_cycles: int) -> float:
        """
        Mean Jain index of per-thread acquisition counts within fixed-width time
        windows.

        Long-run fairness hides short-run starvation: a lock can hand every
        thread an equal share over ten seconds while granting it in long
        same-thread bursts. Evaluated across several window sizes, this separates
        "fair at every timescale" (ticket) from "fair only in aggregate" (TTAS).
        """
        assert(self._global_timeline is not None)
        if self._global_timeline.empty or window_cycles <= 0:
            return float('nan')

        acq = self._global_timeline[self._global_timeline['event_type'] == ACQUISITION]
        if acq.empty:
            return float('nan')

        ts = acq['timestamp'].to_numpy()
        tid = acq['thread_id'].to_numpy().astype(np.int64)

        bins = ((ts - ts.min()) // np.uint64(window_cycles)).astype(np.int64)
        n_windows = int(bins.max()) + 1

        # Drop the trailing partial window: it covers less wall time than the
        # others, so its counts are lower for a reason that has nothing to do
        # with fairness.
        if n_windows > 1:
            keep = bins < (n_windows - 1)
            bins = bins[keep]
            tid = tid[keep]
            n_windows -= 1
            if bins.size == 0:
                return float('nan')

        counts = np.zeros((n_windows, int(self.num_threads)), dtype=np.float64)
        # A thread absent from a window must count as zero, not be omitted --
        # that absence is precisely the unfairness being measured.
        np.add.at(counts, (bins, tid), 1.0)

        totals = counts.sum(axis=1)
        active = totals > 0
        if not active.any():
            return float('nan')

        counts = counts[active]
        sq = (counts ** 2).sum(axis=1)
        per_window = (counts.sum(axis=1) ** 2) / (counts.shape[1] * sq)
        return float(np.mean(per_window))

    def lock_transfer_matrix(self) -> np.ndarray:
        """
        Tracks which threads are most likely to aquire lock after some thread X releases lock
        Returns nxn np.array where n = thread count
        """

        n = self.num_threads
        transfer_table = np.zeros((n, n))

        # 1. Filter only acquisition events
        acquisitions = self._global_timeline[self._global_timeline['event_type'] == ACQUISITION]
        t_ids = acquisitions['thread_id'].to_numpy()

        # 2. 'from_threads' are elements 0 to N-1
        #    'to_threads'   are elements 1 to N
        # This aligns the previous owner with the next owner perfectly.
        from_threads = t_ids[:-1]
        to_threads = t_ids[1:]

        # 3. Use np.add.at to handle duplicate transfers correctly
        # transfer_table[from, to] += 1
        np.add.at(transfer_table, (from_threads, to_threads), 1)

        return transfer_table

    def self_transfer_rate(self, matrix: Optional[np.ndarray] = None) -> float:
        """
        Fraction of lock handoffs that went straight back to the releasing thread.

        The diagonal of the transfer matrix is barging: a thread releasing and
        immediately re-acquiring while others wait. It is the mechanism behind
        both TTAS's throughput advantage and its unfairness, and it turns the
        transfer heatmap into a number that can be plotted against thread count.
        """
        m = self.lock_transfer_matrix() if matrix is None else matrix
        total = float(m.sum())
        if total == 0:
            return float('nan')
        return float(np.trace(m)) / total

    def transfer_entropy(self, matrix: Optional[np.ndarray] = None) -> float:
        """
        Mean row entropy of the transfer matrix, normalized to [0, 1].

        1.0 means a releasing thread hands off uniformly across its peers; 0.0
        means the successor is fully determined. Queue locks sit low because
        strict rotation is deterministic, and convoying locks sit low because the
        same thread always wins -- read it alongside self_transfer_rate, which
        tells those two cases apart.
        """
        m = self.lock_transfer_matrix() if matrix is None else matrix
        n = m.shape[0]
        if n <= 1:
            return 1.0

        row_sums = m.sum(axis=1)
        active = row_sums > 0
        if not active.any():
            return float('nan')

        p = m[active] / row_sums[active][:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            terms = np.where(p > 0, -p * np.log(p), 0.0)
        return float(terms.sum(axis=1).mean() / np.log(n))
