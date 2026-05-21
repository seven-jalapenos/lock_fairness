
import re
import numpy as np

def parse_tsc_offsets(file_path: str = "../logs/rdtsc_offsets.txt") -> np.ndarray:
    """
    Parses core offsets and returns a NumPy array where the index
    corresponds to the Core ID.
    """
    offsets = {}
    
    # Regex to capture the Core ID and the cycle value
    # Matches "Core 14: 4 cycles"
    pattern = re.compile(r"Core\s+(\d+):\s+(-?\d+)\s+cycles")

    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    core_id = int(match.group(1))
                    offset_value = int(match.group(2))
                    offsets[core_id] = offset_value
    except FileNotFoundError:
        print(f"Error: Offset file {file_path} not found.")
        return None

    if not offsets:
        return None

    # Convert to a dense NumPy array for fast indexing
    # We find the max core ID to ensure the array is large enough
    max_core = max(offsets.keys())
    offset_array = np.zeros(max_core + 1, dtype=np.int64)
    
    for core_id, val in offsets.items():
        offset_array[core_id] = val
        
    return offset_array
