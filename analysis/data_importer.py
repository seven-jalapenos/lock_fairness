
from pathlib import Path
import pandas as pd

def import_parquet(file_path: Path) -> pd.DataFrame:
    """
    Imports a Parquet file into a pandas DataFrame.

    Restores wait_time/hold_time when the file doesn't carry them. They are pure
    subtractions of the stored timestamps, so DataExporter leaves them off disk;
    recomputing here is what makes a stored frame indistinguishable from a freshly
    parsed one, and keeps run directories written before that change readable.
    """
    try:
        df = pd.read_parquet(file_path)
        if 'acquisition' in df.columns:
            if 'wait_time' not in df.columns:
                df['wait_time'] = df['acquisition'] - df['invocation']
            if 'hold_time' not in df.columns:
                df['hold_time'] = df['release'] - df['acquisition']
        return df
    except Exception as e:
        print(f"Error importing Parquet: {e}")
        return pd.DataFrame()
