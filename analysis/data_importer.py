
from pathlib import Path
import pandas as pd

def import_parquet(file_path: Path) -> pd.DataFrame:
    """
    Imports a Parquet file into a pandas DataFrame.
    """
    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        print(f"Error importing Parquet: {e}")
        return pd.DataFrame()