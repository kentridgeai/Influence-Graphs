import os
import re
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker



# --- Aggregation helpers ---
def extract_quantile_from_filename(filename, analysis_type='epoch', negative_weight_mode=None):
    """Extract integer quantile value from filenames like epoch_10_w_neg_metrics_graph_IGv5.csv."""
    if negative_weight_mode is None:
        mode_pattern = r'(?:w|wo|abs)'
    else:
        mode_pattern = re.escape(str(negative_weight_mode))
    pattern = rf'^{re.escape(str(analysis_type))}_(\d+(?:p\d+)?)_{mode_pattern}_neg_metrics_graph_IGv5\.csv$'
    m = re.search(pattern, filename)
    if not m:
        return None
    quantile_str = m.group(1).replace('p', '.')
    quantile_val = float(quantile_str)
    return int(quantile_val) if float(quantile_val).is_integer() else quantile_val


def aggregate_graph_metric_csvs(output_dir, analysis_type='epoch', negative_weight_mode='w'):
    """
    Read all per-threshold graph-metric CSV files in output_dir matching the selected
    negative-weight mode, such as `epoch_xx_w_neg_metrics_graph_IGv5.csv`, keep only
    xx in {10, 20, 30, 40, 50, 60, 70, 80, 90}, average the graph metrics over files
    for each dataset/epoch, and save the aggregated CSV.
    """
    allowed_quantiles = {10, 20, 30, 40, 50, 60, 70, 80, 90}
    pattern = re.compile(rf'^{analysis_type}_(\d+(?:p\d+)?)_{re.escape(str(negative_weight_mode))}_neg_metrics_graph_IGv5\.csv$')

    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    candidate_files = []
    for fname in os.listdir(output_dir):
        if not pattern.match(fname):
            continue
        quantile_val = extract_quantile_from_filename(
            fname,
            analysis_type=analysis_type,
            negative_weight_mode=negative_weight_mode,
        )
        if quantile_val in allowed_quantiles:
            candidate_files.append(os.path.join(output_dir, fname))

    if len(candidate_files) == 0:
        raise FileNotFoundError(
            f"No matching CSV files found in {output_dir} for mode {negative_weight_mode} and quantiles {sorted(allowed_quantiles)}"
        )

    print(f"[Info] Found {len(candidate_files)} graph-metric CSV files for aggregation under mode {negative_weight_mode}.")

    dataframes = []
    required_metric_cols = ['Average Path Length', 'MMID', 'Number of Clusters', 'Average Intra-cluster Weight']
    key_cols = ['dataset', 'epoch']

    for csv_file in sorted(candidate_files):
        df = pd.read_csv(csv_file)
        missing_cols = [c for c in key_cols + required_metric_cols if c not in df.columns]
        if missing_cols:
            print(f"[Skip] Missing columns {missing_cols} in {csv_file}")
            continue
        df = df[key_cols + required_metric_cols].copy()
        dataframes.append(df)

    if len(dataframes) == 0:
        raise ValueError("No valid CSV files remained after column validation.")

    combined_df = pd.concat(dataframes, ignore_index=True)
    aggregated_df = (
        combined_df
        .groupby(key_cols, as_index=False)[required_metric_cols]
        .mean()
        .sort_values(by=key_cols)
    )

    aggregated_df['apl_quantiles'] = f'avg_over_q{{10,20,30,40,50,60,70,80,90}}_{negative_weight_mode}'
    aggregated_df['nc_quantiles'] = f'avg_over_q{{10,20,30,40,50,60,70,80,90}}_{negative_weight_mode}'

    ordered_cols = [
        'dataset', 'epoch', 'apl_quantiles', 'nc_quantiles',
        'Average Path Length', 'MMID', 'Number of Clusters', 'Average Intra-cluster Weight'
    ]
    aggregated_df = aggregated_df[ordered_cols]

    save_path = os.path.join(output_dir, f'{analysis_type}_IGv5_average_{negative_weight_mode}.csv')
    aggregated_df.to_csv(save_path, index=False)
    print(f"[Info] Aggregated graph-metric CSV saved to: {save_path}")
    return save_path

def plot_graph_metrics(
        csv_path,
        analysis_type,
        metrics_to_plot,
        metric_label_map,
        save_dir='visualizations/graph_plots',
        datasets_to_plot=None,
        rows_as_datasets=True,
        file_tag_suffix=""
):
    """
    Plot only the required graph metrics in a grid layout.
    """
    try:
        mpl.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'mathtext.fontset': 'stix',
        })
    except Exception as e:
        print(f"Warning: Could not set font to Times New Roman. Using default. Error: {e}")

    mpl.rcParams['savefig.bbox'] = 'standard'
    mpl.rcParams['savefig.pad_inches'] = 0.02

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}. Cannot generate plot.")
        return

    LABEL_FONTSIZE = 18
    TICK_FONTSIZE = 18

    iteration_col = 'noise_level' if analysis_type == 'noise_level' else 'epoch'
    if analysis_type == 'noise_level':
        x_label = 'Noise Level'
    else:
        x_label = 'Epoch'

    if datasets_to_plot is None:
        datasets_to_plot = sorted(df['dataset'].unique())
    else:
        existing = set(df['dataset'].unique())
        datasets_to_plot = [d for d in datasets_to_plot if d in existing]

    DISPLAY_NAME_ALIAS = {
        'Flowers102_withpretraining': 'Flowers102 \n w. pre-training',
        'Flowers102': 'Flowers102',
        'FGVCAircraft_dataaug': 'FGVC Aircraft\nw. data-augmentation',
        'CIFAR10 (5k train)': 'CIFAR10',
        'MNIST (5k train)': 'MNIST',
        'CIFAR10 (full train)': 'CIFAR10',
        'MNIST (full train)': 'MNIST',
        'IGv5_Flowers102':'Flowers102',
        'IGv5_FGVCAircraft':'FGVC Aircraft',
        'IGv5_CIFAR10':'CIFAR10',
        'IGv5_MNIST':'MNIST',
    }

    if metrics_to_plot is None or len(metrics_to_plot) == 0:
        print("[Warn] metrics_to_plot is empty; nothing to plot.")
        return
    if datasets_to_plot is None or len(datasets_to_plot) == 0:
        present = sorted(df['dataset'].unique())
        print(f"[Warn] No datasets selected/found for plotting. Present in CSV: {present}")
        return

    num_metrics = len(metrics_to_plot)
    num_datasets = len(datasets_to_plot)

    if rows_as_datasets:
        fig, axes = plt.subplots(
            num_datasets, num_metrics,
            figsize=(4 * num_metrics, 3 * num_datasets),
            squeeze=False,
            constrained_layout=False,
        )
        for i, dataset_name in enumerate(datasets_to_plot):
            for j, metric_name in enumerate(metrics_to_plot):
                ax = axes[i, j]
                dataset_df = df[(df['dataset'] == dataset_name) & (df[metric_name].notna())].sort_values(by=iteration_col)
                if dataset_df.empty:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    continue
                ax.plot(
                    dataset_df[iteration_col],
                    dataset_df[metric_name],
                    marker='o', linestyle='-', markersize=10, linewidth=4, color='k'
                )
                ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
                col_label = metric_label_map.get(metric_name, metric_name)
                ax.set_title(col_label, fontsize=LABEL_FONTSIZE + 2, fontweight='bold')
                if j == 0:
                    display_name = DISPLAY_NAME_ALIAS.get(dataset_name, dataset_name)
                    ax.set_ylabel(display_name, fontsize=LABEL_FONTSIZE, fontweight='bold')
                    ax.yaxis.label.set_multialignment('center')
                formatter = ticker.FuncFormatter(lambda x, pos: f'{x:.2f}')
                ax.yaxis.set_major_formatter(formatter)
                y_axis_min, y_axis_max = ax.get_ylim()
                min_metric_value = float(dataset_df[metric_name].min())
                current_ticks = ax.get_yticks()
                if np.isfinite(min_metric_value):
                    tick_values = current_ticks[current_ticks >= min_metric_value]
                    tick_values = tick_values[tick_values <= y_axis_max]
                    tick_values = np.sort(np.unique(tick_values))
                    tick_values = np.sort(np.unique(np.append(tick_values, min_metric_value)))
                    tick_values = tick_values[tick_values <= y_axis_max]
                    if tick_values.size >= 2 and np.isclose(tick_values[0], min_metric_value):
                        nearest_gap = tick_values[1] - tick_values[0]
                        if tick_values.size >= 3:
                            ref_spacing = np.min(np.diff(tick_values[1:]))
                        else:
                            ref_spacing = max(abs(tick_values[1]) * 0.05, 1e-8)
                        if nearest_gap < 0.35 * ref_spacing:
                            tick_values = np.delete(tick_values, 1)
                    ax.set_yticks(tick_values)
                    ax.set_ylim(y_axis_min, y_axis_max)
                ax.tick_params(axis='y', labelleft=True)
                ax.set_xlabel(x_label, fontsize=LABEL_FONTSIZE, fontweight='bold')
                if analysis_type == 'noise_level':
                    ax.set_xlim(0.0, 0.5)
                    ax.set_xticks(np.arange(0.0, 0.51, 0.1))
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    else:
        fig, axes = plt.subplots(
            num_metrics, num_datasets,
            figsize=(4 * num_datasets, 2.5 * num_metrics),
            squeeze=False,
            constrained_layout=False,
        )
        for i, metric_name in enumerate(metrics_to_plot):
            for j, dataset_name in enumerate(datasets_to_plot):
                ax = axes[i, j]
                dataset_df = df[(df['dataset'] == dataset_name) & (df[metric_name].notna())].sort_values(by=iteration_col)
                if dataset_df.empty:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    continue
                ax.plot(
                    dataset_df[iteration_col],
                    dataset_df[metric_name],
                    marker='o', linestyle='-', markersize=10, linewidth=4, color='k'
                )
                ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
                display_name = DISPLAY_NAME_ALIAS.get(dataset_name, dataset_name)
                row_label = metric_label_map.get(metric_name, metric_name)
                ax.set_title(f"{display_name}", fontsize=LABEL_FONTSIZE + 2, fontweight='bold')
                if j == 0:
                    ax.set_ylabel(row_label, fontsize=LABEL_FONTSIZE, fontweight='bold')
                    ax.yaxis.label.set_multialignment('center')
                formatter = ticker.FuncFormatter(lambda x, pos: f'{x:.2f}')
                ax.yaxis.set_major_formatter(formatter)
                y_axis_min, y_axis_max = ax.get_ylim()
                min_metric_value = float(dataset_df[metric_name].min())
                current_ticks = ax.get_yticks()
                if np.isfinite(min_metric_value):
                    tick_values = current_ticks[current_ticks >= min_metric_value]
                    tick_values = tick_values[tick_values <= y_axis_max]
                    tick_values = np.sort(np.unique(tick_values))
                    tick_values = np.sort(np.unique(np.append(tick_values, min_metric_value)))
                    tick_values = tick_values[tick_values <= y_axis_max]
                    if tick_values.size >= 2 and np.isclose(tick_values[0], min_metric_value):
                        nearest_gap = tick_values[1] - tick_values[0]
                        if tick_values.size >= 3:
                            ref_spacing = np.min(np.diff(tick_values[1:]))
                        else:
                            ref_spacing = max(abs(tick_values[1]) * 0.05, 1e-8)
                        if nearest_gap < 0.35 * ref_spacing:
                            tick_values = np.delete(tick_values, 1)
                    ax.set_yticks(tick_values)
                    ax.set_ylim(y_axis_min, y_axis_max)
                ax.tick_params(axis='y', labelleft=True)
                ax.set_xlabel(x_label, fontsize=LABEL_FONTSIZE, fontweight='bold')
                if analysis_type == 'noise_level':
                    ax.set_xlim(0.0, 0.5)
                    ax.set_xticks(np.arange(0.0, 0.51, 0.1))
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    fig.subplots_adjust(
        left=0.08,
        bottom=0.1,
        right=0.99,
        top=0.95,
        wspace=0.35,
        hspace=0.8
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    base_name = f'{analysis_type}'

    if file_tag_suffix:
        filename = f'{base_name}_{file_tag_suffix}.pdf'
    else:
        filename = f'{base_name}.pdf'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format='pdf', dpi=300)
    print(f"\n--- Whole graph metrics grid plot saved to: {save_path} ---")
    # plt.show()
    plt.close()


if __name__ == '__main__':
    ANALYSIS_TYPE = 'epoch'
    OUTPUT_DIR = 'visualizations/graph_metric_AISTATS_IGv5'
    NEGATIVE_WEIGHT_MODE = 'wo'  # choose from 'w', 'wo', 'abs'

    file_tag_suffix = f'average_{NEGATIVE_WEIGHT_MODE}'

    metrics_to_plot = [
        # 'Average Path Length',
        'MMID',
        'Number of Clusters',
        'Average Intra-cluster Weight'
    ]

    metric_label_map = {
        'Average Path Length': 'Avg. Path Len.', #\n w/o Neg. Wts
        'MMID': 'MMDI',
        'Number of Clusters': 'Num. Clusters',
        'Average Intra-cluster Weight': 'Avg. Intra-cluster \n Weight',
    }

    PLOT_DATASETS = ['IGv5_CIFAR10', 'IGv5_FGVCAircraft',
                     'IGv5_Flowers102', 'IGv5_MNIST']
    LAYOUT_ROWS_AS_DATASETS = False

    csv_path = aggregate_graph_metric_csvs(
        output_dir=OUTPUT_DIR,
        analysis_type=ANALYSIS_TYPE,
        negative_weight_mode=NEGATIVE_WEIGHT_MODE,
    )

    if os.path.exists(csv_path):
        print(f"\n--- Generating Whole Graph Plot for {ANALYSIS_TYPE.replace('_', ' ').title()} ---")
        plot_graph_metrics(
            csv_path,
            ANALYSIS_TYPE,
            metrics_to_plot,
            metric_label_map,
            OUTPUT_DIR,
            datasets_to_plot=PLOT_DATASETS,
            rows_as_datasets=LAYOUT_ROWS_AS_DATASETS,
            file_tag_suffix=file_tag_suffix
        )
    else:
        print(f"CSV not found for plotting: {csv_path}")