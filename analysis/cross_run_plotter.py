import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
from typing import Callable, Dict, Any, Optional

from .log_analyzer import WINDOW_SIZES_CYCLES, WINDOW_LABELS

class CrossRunPlotter:
    """
    Aggregates scalar metrics across multiple run directories and generates 
    comparative plots based on run parameters (e.g., lock type, thread count).
    """
    def __init__(self, runs_csv_dir: Path | str, figures_dir: Path | str):
        self.runs_csv_dir = Path(runs_csv_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Internal dataframe to hold all aggregated data
        self.aggregated_data: pd.DataFrame = pd.DataFrame()

    def load_data(self, param_parser: Optional[Callable[[str], Dict[str, Any]]] = None) -> 'CrossRunPlotter':
        """
        Walks through the run directories, parses parameters from the folder name, 
        and aggregates the summary_scalar_metrics.csv files into a single DataFrame.
        """
        if param_parser is None:
            param_parser = self._default_param_parser

        all_records = []

        # Find all run directories containing the summary scalar metrics
        for run_dir in self.runs_csv_dir.iterdir():
            if not run_dir.is_dir():
                continue
                
            summary_file = run_dir / "summary_scalar_metrics.csv"
            if not summary_file.exists():
                continue

            # Extract parameters (e.g., lock_type, threads) from the folder name
            params = param_parser(run_dir.name)
            if not params:
                print(f"Warning: Could not parse parameters from '{run_dir.name}'. Skipping.")
                continue

            # Read the metrics for this run
            df = pd.read_csv(summary_file)
            
            # Append parameters and run_id to every row
            for key, value in params.items():
                df[key] = value
            df['run_id'] = run_dir.name
            
            all_records.append(df)

        if all_records:
            self.aggregated_data = pd.concat(all_records, ignore_index=True)
            print(f"Successfully loaded data across {len(all_records)} runs.")
        else:
            print("No data loaded. Check directory structure and naming conventions.")
            
        return self

    # Run directories are named <lock>_<threads>_<pin>_w<work> (see
    # scripts/runner.py:run_dir_id). Anchored, because an unanchored match would
    # accept the `<lock>_<threads>_<pin>` prefix of a work-suffixed directory and
    # silently collapse every work size onto one line. The work group is optional
    # so directories from sweeps predating the work dimension still parse.
    _DIR_PATTERN = re.compile(
        r'^(?P<lock>[A-Za-z]+)_(?P<threads>\d+)_(?P<pin>\d+)(?:_w(?P<work>\d+))?$'
    )

    def _default_param_parser(self, folder_name: str) -> Dict[str, Any]:
        """
        Parses run folder names like `mcs_8_1_w10000` (or legacy `mcs_8_1`) into
        'lock_type', 'threads', 'pin' and 'work'. Modify this if your naming differs.
        """
        match = self._DIR_PATTERN.match(folder_name)
        if match:
            work = match.group('work')
            return {
                'lock_type': match.group('lock'),
                'threads': int(match.group('threads')),
                'pin': int(match.group('pin')),
                'work': int(work) if work is not None else None
            }
        return {}

    def plot_metric(self, metric_name: str, x_axis: str, line_axis: Optional[str] = None,
                    save_csv: bool = False, where: Optional[Dict[str, Any]] = None,
                    name_suffix: str = '') -> None:
        """
        Generates a plot tracking a specific metric across an x-axis (e.g., threads).
        Optionally separates data into different lines based on a line_axis (e.g., lock_type).

        `where` restricts the rows to an exact-match on parameter columns (e.g.
        {'work': 10000}) and `name_suffix` distinguishes the resulting files --
        together they let one aggregated tree emit a separate figure per work size
        rather than drawing different CS lengths as points on the same line.
        """
        if self.aggregated_data.empty:
            print("No data to plot. Call load_data() first.")
            return

        # Filter down to the specific metric we want to plot
        df_metric = self.aggregated_data[self.aggregated_data['Metric'] == metric_name]

        if where:
            for key, value in where.items():
                if key not in df_metric.columns:
                    continue
                # isna() rather than == for the legacy no-work group, since NaN != NaN
                mask = df_metric[key].isna() if value is None else df_metric[key] == value
                df_metric = df_metric[mask]

        if df_metric.empty:
            print(f"Metric '{metric_name}' not found in aggregated data.")
            return

        if save_csv:
            # Determine exactly which columns are relevant to this specific visualization
            columns_to_save = [x_axis]
            if line_axis and line_axis in df_metric.columns:
                columns_to_save.append(line_axis)
            columns_to_save.extend(['Average', 'Standard_Deviation'])
            
            # Filter to unique columns to avoid errors if any variables overlap
            columns_to_save = list(dict.fromkeys([col for col in columns_to_save if col in df_metric.columns]))
            
            # Sort columns predictably matching how they are drawn on the chart
            sort_order = [line_axis, x_axis] if (line_axis and line_axis in df_metric.columns) else [x_axis]
            df_csv = df_metric[columns_to_save].sort_values(by=sort_order, ignore_index=True)
            
            # Save the CSV companion dataset
            csv_filename = f"cross_run_{metric_name}_by_{x_axis}{name_suffix}.csv"
            csv_path = self.figures_dir / csv_filename
            df_csv.to_csv(csv_path, index=False)
            print(f"Saved cross-run metric data to: {csv_path}")

        plt.figure(figsize=(10, 6))

        # If we have a line_axis (e.g., lock_type), we draw a separate line for each type
        if line_axis and line_axis in df_metric.columns:
            groups = df_metric.groupby(line_axis)
            for name, group in groups:
                # Sort by x_axis so the line draws correctly left-to-right
                group = group.sort_values(by=x_axis)
                plt.errorbar(
                    group[x_axis], 
                    group['Average'], 
                    yerr=group['Standard_Deviation'], 
                    marker='o', 
                    capsize=5, 
                    label=str(name).upper(),
                    linestyle='-',
                    linewidth=2
                )
        else:
            # Just one line if no grouping axis is provided
            df_metric = df_metric.sort_values(by=x_axis)
            plt.errorbar(
                df_metric[x_axis], 
                df_metric['Average'], 
                yerr=df_metric['Standard_Deviation'], 
                marker='o', 
                capsize=5,
                color='b',
                linestyle='-',
                linewidth=2
            )

        # Formatting
        title = f"{metric_name.replace('_', ' ').title()} vs {x_axis.title()}"
        if where:
            title += " (" + ", ".join(f"{k}={v}" for k, v in where.items()) + ")"
        plt.title(title, fontsize=14)
        plt.xlabel(x_axis.replace('_', ' ').title(), fontsize=12)
        plt.ylabel(f"{metric_name.replace('_', ' ').title()} (Average)", fontsize=12)
        
        # Ensure x-axis only shows integer ticks if it's thread count
        if pd.api.types.is_numeric_dtype(df_metric[x_axis]):
            plt.xticks(df_metric[x_axis].unique())

        if line_axis:
            plt.legend(title=line_axis.replace('_', ' ').title())

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # Save figure
        out_filename = f"cross_run_{metric_name}_by_{x_axis}{name_suffix}.png"
        out_path = self.figures_dir / out_filename
        plt.savefig(out_path, dpi=300)
        plt.close()
        
        print(f"Saved cross-run plot to: {out_path}")

    def plot_timescale(self, where: Optional[Dict[str, Any]] = None,
                       name_suffix: str = '', threads: Optional[int] = None) -> None:
        """
        Fairness against the timescale it is measured over: windowed Jain index
        vs window size, one line per lock.

        A single fairness number is always a number at some timescale. A lock
        that hands out equal shares over ten seconds but grants the lock in long
        same-thread bursts scores well at a coarse window and badly at a fine
        one; a strictly rotating lock is flat across all of them. The shape of
        this curve is the distinction, and no scalar carries it.
        """
        if self.aggregated_data.empty:
            print("No data to plot. Call load_data() first.")
            return

        df = self.aggregated_data
        if where:
            for key, value in where.items():
                if key not in df.columns:
                    continue
                mask = df[key].isna() if value is None else df[key] == value
                df = df[mask]

        if df.empty:
            return

        # Contention is what makes the timescale question interesting, so default
        # to the busiest configuration measured.
        if threads is None:
            threads = int(df['threads'].max())
        df = df[df['threads'] == threads]
        if df.empty:
            return

        windows = sorted(WINDOW_SIZES_CYCLES)

        plt.figure(figsize=(10, 6))
        drew_any = False

        for lock, group in df.groupby('lock_type'):
            xs, ys, errs = [], [], []
            for cycles in windows:
                row = group[group['Metric'] == f'windowed_jain_{WINDOW_LABELS[cycles]}']
                if row.empty:
                    continue
                xs.append(cycles)
                ys.append(float(row['Average'].iloc[0]))
                errs.append(float(row['Standard_Deviation'].iloc[0]))

            if not xs:
                continue

            drew_any = True
            plt.errorbar(xs, ys, yerr=errs, marker='o', capsize=5,
                         label=str(lock).upper(), linestyle='-', linewidth=2)

        if not drew_any:
            plt.close()
            return

        plt.xscale('log')
        plt.xticks(windows, [WINDOW_LABELS[w] for w in windows])
        title = f"Windowed Jain Index vs Window Size ({threads} threads)"
        if where:
            title += " (" + ", ".join(f"{k}={v}" for k, v in where.items()) + ")"
        plt.title(title, fontsize=14)
        plt.xlabel("Window Size (TSC cycles)", fontsize=12)
        plt.ylabel("Jain Fairness Index (1.0 = equal shares)", fontsize=12)
        plt.legend(title="Lock Type")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        out_path = self.figures_dir / f"cross_run_windowed_jain_by_window{name_suffix}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"Saved timescale plot to: {out_path}")
