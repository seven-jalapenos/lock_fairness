
from typing import List
import numpy as np
import pandas as pd


class LogAnalyzer:

    def __init__(self, data: pd.DataFrame, global_timeline: pd.DataFrame=None):
        self._data = data
        if not global_timeline:
            self._global_timeline = self.create_global_timeline(data)
        else:
            self._global_timeline = global_timeline
    
        self.overtake_metrics = None
    
    def build_overtake(self):
        overtake_timeline = self.create_overtake_timeline(self._global_timeline)
        self.overtake_metrics = self.OvertakeMetrics(overtake_timeline)


####################################################
#
#            BUILD TIMELINES
#
#####################################################


    @staticmethod
    def create_global_timeline(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts a flat lock event DataFrame into a chronological global timeline.
        """
        if df.empty:
            return pd.DataFrame()

        # 1. 'Melt' the three timestamp columns into a single chronological column
        timeline_df = df.melt(
            id_vars=['thread_id'],                                # Keep thread_id as is
            value_vars=['invocation', 'acquisition', 'release'],  # Columns to stack
            var_name='event_type',                                # New column holding the old column names
            value_name='timestamp'                                # New column holding the actual timestamps
        )
        
        # 2. Sort the entire dataframe chronologically by timestamp
        timeline_df = timeline_df.sort_values(by='timestamp', ignore_index=True)
        
        return timeline_df
    
    # def create_global_timeline(data: List[np.ndarray]) -> np.ndarray:
    #     timeline_dtype = np.dtype([
    #         ('timestamp', 'u8'),
    #         ('thread_id', 'u4'),
    #         ('event_type', 'U10')
    #     ])
        
    #     # 1. Pre-calculate total size
    #     total_entries = sum(len(thread['data']) for thread in data) * 3
        
    #     # 2. Pre-allocate the array
    #     timeline_array = np.empty(total_entries, dtype=timeline_dtype)
        
    #     # 3. Fill by index
    #     idx = 0
    #     for thread in data:
    #         tid = thread['thread_id']
    #         for entry in thread['data']:
    #             timeline_array[idx] = (entry['invocation'], tid, 'invocation')
    #             timeline_array[idx+1] = (entry['acquisition'], tid, 'acquisition')
    #             timeline_array[idx+2] = (entry['release'], tid, 'release')
    #             idx += 3
                
    #     # 4. Sort in place (more memory efficient)
    #     timeline_array.sort(order='timestamp')
    #     return timeline_array


    @staticmethod
    def create_overtake_timeline(self) -> np.ndarray:
        """
        Creates an operation-wise timeline of overtakes
        operations are ordered based on invocation time
        """

        timeline = self._global_timeline
        # 1. Create the cumulative sum of all acquisitions
        is_acq = (timeline['event_type'] == 'acquisition').astype(int)
        acq_cumsum = np.cumsum(is_acq)
        
        op_results = []
        
        # 2. Iterate through each thread to pair its internal events
        for tid in np.unique(timeline['thread_id']):
            # Identify the timeline indices for this thread's events
            thread_mask = (timeline['thread_id'] == tid)
            inv_indices = np.where(thread_mask & (timeline['event_type'] == 'invocation'))[0]
            acq_indices = np.where(thread_mask & (timeline['event_type'] == 'acquisition'))[0]
            
            # Get the start times (to preserve global order later)
            start_times = timeline['timestamp'][inv_indices]
            
            # Calculate intervening counts:
            # (Total acquisitions at time of capture) - (Total acquisitions at time of request) - 1
            counts_at_inv = acq_cumsum[inv_indices]
            counts_at_acq = acq_cumsum[acq_indices]
            intervening = counts_at_acq - counts_at_inv - 1
            
            # Store as (timestamp, thread_id, count)
            for ts, count in zip(start_times, intervening):
                op_results.append((ts, tid, count))

        # 3. Convert to a structured array
        dtype = [
            ('invocation_time', 'u8'),
            ('thread_id', 'u4'),
            ('intervening_acquisitions', 'i4')
        ]
        final_ops = np.array(op_results, dtype=dtype)
        
        # 4. Sort by invocation_time so the array is "per operation" in order
        # final_ops.sort(order='invocation_time')
        
        return final_ops


#############################
#
#         METRICS
#
#############################


    def calculate_all(self):
        pass

    def find_avg_per_thread_wait_time(self) -> np.ndarray:
        """
        Computes the average wait time for each thread and returns it as a NumPy array.
        """
        avg_wait_times = np.zeros(self.num_threads)
        
        for thread in self._data:
            avg_wait = np.mean(thread['wait_times'])
            avg_wait_times[thread['thread_id']] = avg_wait
        
        return avg_wait_times

    def percent_time_in_CS(self) -> np.ndarray:
        """
        Computes the percentage of time each thread spends in the critical section.
        """
        percentages = np.zeros(self.num_threads)  # Pre-allocate for efficiency
        
        for thread in self._data:
            total_wait_time = np.sum(thread['wait_times'])
            total_hold_time = np.sum(thread['hold_times'])
            
            if total_wait_time + total_hold_time > 0:
                percent_in_cs = (total_hold_time / (total_wait_time + total_hold_time)) * 100
            else:
                percent_in_cs = 0.0
            
            percentages[thread['thread_id']] = percent_in_cs
        
        return percentages

    def total_CS_completions(self) -> int:
        """
        Computes the total number of critical section completions
        """
        completions = 0
        
        for thread in self._data:
            num_completions = len(thread['hold_times'])
            completions += num_completions
        
        return completions

    def track_lock_ownership_transfer(self) -> np.ndarray:
        """
        Tracks which threads are most likely to aquire lock after some thread X releases lock
        Returns nxn np.array where n = thread count
        """

        n = self.num_threads
        transfer_table = np.zeros((n, n))
        
        # 1. Filter only acquisition events
        acquisitions = self._global_timeline[self._global_timeline['event_type'] == 'acquisition']
        t_ids = acquisitions['thread_id']
        
        # 2. 'from_threads' are elements 0 to N-1
        #    'to_threads'   are elements 1 to N
        # This aligns the previous owner with the next owner perfectly.
        from_threads = t_ids[:-1]
        to_threads = t_ids[1:]
        
        # 3. Use np.add.at to handle duplicate transfers correctly
        # transfer_table[from, to] += 1
        np.add.at(transfer_table, (from_threads, to_threads), 1)
        
        return transfer_table


###################################
#
#    OVERTAKE METRICS OBJECT
#
###################################

    class OvertakeMetrics:

        def __init__(self, overtake_timeline: np.ndarray):
            self._overtake_timeline = overtake_timeline

        def overtake_percentage(self) -> float:
            """
            Computes the percentage of operations that are overtaken.
            """
            counts = self._overtake_timeline['intervening_acquisitions']
        
            if counts.size == 0:
                return 0.0
            
            # np.mean on a boolean mask gives the fraction of True values
            return np.mean(counts > 0) * 100

        def average_overtake_depth(self) -> float:
            """
            average overtake depth is the average depth of elements with overtake depth > 0
            """
            overtake_depths = self._overtake_timeline['intervening_acquisitions']
            positive_depths = overtake_depths[overtake_depths > 0]
            
            if len(positive_depths) == 0:
                return 0.0
            
            return np.mean(positive_depths)
        
        def rank_inversion_penalty(self, denom=100000) -> float:
            """
            Computes the total rank inversion penalty across all operations.
            returns as per-denom, default 100k operations
            k = 2, or quadratic penalty
            one operation overtaken 3 times is worse than 3 operations overtaken once each
            """

            total_rip = np.sum(self._overtake_timeline['intervening_acquisitions'] ** 2)
            scaled_rip = total_rip / denom
            return scaled_rip
    