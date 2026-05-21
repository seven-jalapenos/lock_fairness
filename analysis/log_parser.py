
import numpy as np
import pandas as pd
import os
from .parse_offsets import parse_tsc_offsets

class LogParser:
    def __init__(self, file_path: str, offset_file_path: str, pinning_policy: int = 1):
        self.file_path = file_path
        self.offset_file_path = offset_file_path
        self.pinning_policy = pinning_policy

        self.offset_table = parse_tsc_offsets(offset_file_path)

        self.all_threads_data = self.parse_logs()
    
    def parse_logs(self) -> pd.DataFrame:
        # Define the LogEntry structure (3 x uint64_t)
        log_entry_dtype = np.dtype([
            ('invocation', 'u8'),
            ('acquisition', 'u8'),
            ('release', 'u8')
        ])
        
        size_t_dtype = np.dtype('u8')

        if not os.path.exists(self.file_path):
            print(f"Error: {self.file_path} not found.")
            return pd.DataFrame() # Return empty DataFrame instead of None
        
        thread_dfs = [] # We will store a flat DataFrame for each thread here

        with open(self.file_path, 'rb') as f:
            thread_id = 0
            while True:
                # 1. Read the number of entries (size_t)
                size_data = np.fromfile(f, dtype=size_t_dtype, count=1)
                if size_data.size == 0:
                    break
                
                num_entries = size_data[0]
                
                # 2. Read all LogEntries for this thread in one block
                data = np.fromfile(f, dtype=log_entry_dtype, count=num_entries)
                
                # Pack into a temp dict so your existing calibration function still works
                tmp = {
                    'thread_id': thread_id,
                    'data': data
                }
                tmp = self.calibrate_log(tmp, self.offset_table, self.pinning_policy)

                # --- THE FLATTENING MAGIC HAPPENS HERE ---
                # Passing the structured array directly to Pandas splits 
                # 'invocation', 'acquisition', and 'release' into individual flat columns!
                df_thread = pd.DataFrame(tmp['data'])
                
                # Add the thread_id as a standard column (Pandas broadcasts the scalar automatically)
                df_thread.insert(0, 'thread_id', thread_id)

                # Calculate metrics as flat columns using clean vectorized math
                df_thread['wait_time'] = df_thread['acquisition'] - df_thread['invocation']
                df_thread['hold_time'] = df_thread['release'] - df_thread['acquisition']

                thread_dfs.append(df_thread)
                thread_id += 1
        
        if not thread_dfs:
            return pd.DataFrame()

        # Vertically stack all individual thread dataframes into one massive table
        final_df = pd.concat(thread_dfs, ignore_index=True)
        return final_df

    # def parse_logs(self) -> list:
    #     # Define the LogEntry structure (3 x uint64_t)
    #     # 'u8' stands for unsigned 8-byte integer
    #     log_entry_dtype = np.dtype([
    #         ('invocation', 'u8'),
    #         ('acquisition', 'u8'),
    #         ('release', 'u8')
    #     ])
        
    #     # size_t is usually 8 bytes on 64-bit systems
    #     size_t_dtype = np.dtype('u8')

    #     all_threads = []

    #     if not os.path.exists(self.file_path):
    #         print(f"Error: {self.file_path} not found.")
    #         return None
        
    #     with open(self.file_path, 'rb') as f:
    #         thread_id = 0
    #         while True:
    #             # 1. Read the number of entries (size_t)
    #             size_data = np.fromfile(f, dtype=size_t_dtype, count=1)
    #             if size_data.size == 0:
    #                 break
                
    #             num_entries = size_data[0]
                
    #             # 2. Read all LogEntries for this thread in one block
    #             data = np.fromfile(f, dtype=log_entry_dtype, count=num_entries)
                
    #             tmp = {
    #                 'thread_id': thread_id,
    #                 'data': data
    #             }

    #             tmp = self.calibrate_log(tmp, self.offset_table, self.pinning_policy)

    #             wait_times = tmp['data']['acquisition'] - tmp['data']['invocation']
    #             hold_times = tmp['data']['release'] - tmp['data']['acquisition']

    #             tmp['wait_times'] = wait_times
    #             tmp['hold_times'] = hold_times

    #             all_threads.append(tmp)

    #             thread_id += 1
        
    #     return all_threads

    # TODO: implement pinning policy 3
    def calibrate_log(self, thread_data: dict, offset_table: np.ndarray, pinning_policy: int) -> dict:
        """
        Applies offsets to make TSC values comparable across threads.
        """

        tid = thread_data['thread_id']
        core = None
        if pinning_policy == 0:
            # no pinning
            return thread_data
        elif pinning_policy == 1:
            # Round-robin pinning: Core = Thread ID % Num Cores
            num_cores = len(offset_table)
            core = tid % num_cores
        elif pinning_policy == 2:
            # all on one core, so assume constant offset
            return thread_data
        elif pinning_policy == 3:
            # TODO: implement pinning policy 3
            # half on one core, round-robin for the rest
            num_cores = len(offset_table)
            half = num_cores // 2
            core_map = lambda tid: half if tid < num_cores // 2 else (tid - half) % (num_cores - half)
        
        offset = offset_table[core]
        
        # Vectorized subtraction on the entire NumPy structured array
        thread_data['data']['invocation']  -= int(offset)
        thread_data['data']['acquisition'] -= int(offset)
        thread_data['data']['release']     -= int(offset)
        
        return thread_data
    