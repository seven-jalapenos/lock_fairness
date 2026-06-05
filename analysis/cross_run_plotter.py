import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
from typing import Callable, Dict, Any, Optional

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

    def _default_param_parser(self, folder_name: str) -> Dict[str, Any]:
        """
        Default parser assuming folder names like: `run_mcs_t4` or `run_ticket_t8`.
        Extracts 'lock_type' and 'threads'. Modify this if your naming differs.
        """
        # Looks for words before '_t' and digits after '_t'
        match = re.search(r'(?P<lock>[A-Za-z]+)_(?P<threads>\d+)_1', folder_name)
        if match:
            # if match.group('lock') == 'ttasb' or match.group('lock') == 'ttas':
            #     return {}
            return {
                'lock_type': match.group('lock'),
                'threads': int(match.group('threads'))
            }
        return {}

    def plot_metric(self, metric_name: str, x_axis: str, line_axis: Optional[str] = None, save_csv: bool = False) -> None:
        """
        Generates a plot tracking a specific metric across an x-axis (e.g., threads).
        Optionally separates data into different lines based on a line_axis (e.g., lock_type).
        """
        if self.aggregated_data.empty:
            print("No data to plot. Call load_data() first.")
            return

        # Filter down to the specific metric we want to plot
        df_metric = self.aggregated_data[self.aggregated_data['Metric'] == metric_name]
        
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
            csv_filename = f"cross_run_{metric_name}_by_{x_axis}.csv"
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
        plt.title(f"{metric_name.replace('_', ' ').title()} vs {x_axis.title()}", fontsize=14)
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
        out_filename = f"cross_run_{metric_name}_by_{x_axis}.png"
        out_path = self.figures_dir / out_filename
        plt.savefig(out_path, dpi=300)
        plt.close()
        
        print(f"Saved cross-run plot to: {out_path}")