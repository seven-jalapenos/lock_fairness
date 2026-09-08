
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
        # Files written before find_offsets computed offsets signed printed them
        # with %lu, so a core whose TSC trailed core 0 appears as its unsigned
        # two's-complement image (~1.8e19), as does the old "no sample" sentinel
        # UINT64_MAX. Assigning that straight into an int64 array raises
        # OverflowError and takes the whole run down, so fold it back to the
        # negative value it stands for.
        if val >= 2 ** 63:
            wrapped = val - 2 ** 64
            print(f"Warning: {file_path} lists core {core_id} as {val}, which is "
                  f"an unsigned wrap written by an older build; reading it as "
                  f"{wrapped} cycles. Re-run `lock_exe --calibrate-only` to "
                  "regenerate the file.")
            val = wrapped
        offset_array[core_id] = val

    missing = [c for c in range(max_core + 1) if c not in offsets]
    if missing:
        print(f"Warning: {file_path} has no entry for core(s) {missing}; they are "
              "treated as offset 0, so timestamps from threads pinned there are "
              "uncalibrated.")

    return offset_array
