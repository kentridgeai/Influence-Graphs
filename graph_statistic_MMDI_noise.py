import os
import re
import datetime
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker


# ---- Utility: sparse matrix memory / shape reporter ----
def print_sparse_matrix_stats(mat, label=""):
    """
    Print sparse matrix stats without copying the matrix:
    shape, nnz, density, dtype, current-backing arrays size,
    and an approximate CSR footprint assuming int32 indices.
    """
    try:
        n, m = mat.shape
        nnz = mat.nnz
        density = nnz / (n * m) if n * m > 0 else 0.0

        total_bytes = 0
        data_dtype = None
        if hasattr(mat, 'data') and mat.data is not None:
            total_bytes += mat.data.nbytes
            data_dtype = mat.data.dtype
        if hasattr(mat, 'indices') and mat.indices is not None:
            total_bytes += mat.indices.nbytes
        if hasattr(mat, 'indptr') and mat.indptr is not None:
            total_bytes += mat.indptr.nbytes
        if hasattr(mat, 'row') and hasattr(mat, 'col'):
            total_bytes += mat.row.nbytes + mat.col.nbytes
            if data_dtype is None and hasattr(mat, 'data') and mat.data is not None:
                data_dtype = mat.data.dtype

        itemsize = np.dtype(data_dtype).itemsize if data_dtype is not None else 8
        approx_csr_bytes = nnz * (itemsize + 4 + 4) + (n + 1) * 4

        def _fmt(b):
            return f"{b / 1024**2:.2f} MB ({b / 1024**3:.2f} GB)"

        print(f"  [Sparse Stats] {label} shape={mat.shape}, nnz={nnz:,}, density={density:.6f}, dtype={data_dtype}")
        print(f"  [Sparse Stats] Backing arrays (current format) ~ {_fmt(total_bytes)}; approx CSR footprint ~ {_fmt(approx_csr_bytes)}")
    except Exception as e:
        print(f"  [Sparse Stats] Unable to read stats for {label}: {e}")




def compute_mmid(sub_graph_mat):
    """Mean of mean incoming edge weights over all nodes, counting empty nodes as zero."""
    num_nodes = sub_graph_mat.shape[0]
    if num_nodes == 0:
        return np.nan
    mat_csc = sub_graph_mat.tocsc()
    in_sum = np.asarray(mat_csc.sum(axis=0)).ravel()
    in_cnt = mat_csc.getnnz(axis=0)
    mean_in = np.zeros(num_nodes, dtype=float)
    nz_in = in_cnt > 0
    mean_in[nz_in] = in_sum[nz_in] / in_cnt[nz_in]
    return float(mean_in.mean())



def calculate_graph_metrics(sub_graph_mat):
    """Only compute MMDI for the whole graph."""
    metrics = {}
    num_nodes = sub_graph_mat.shape[0]
    num_edges = sub_graph_mat.nnz
    if num_nodes == 0:
        return None

    metrics['num_nodes'] = num_nodes
    metrics['num_edges'] = num_edges
    metrics['MMID'] = compute_mmid(sub_graph_mat)
    return metrics


def analyze_and_save_graph_metrics(
        project_root_path,
        output_dir,
        data_folder_relative,
        analysis_type,
        datasets_to_analyze=None,
        noise_prefix=None,
        file_tag_suffix=""
):
    """
    Analyze only MMDI for the whole graph and write rows incrementally.
    """
    base_data_path = os.path.join(project_root_path, data_folder_relative)

    if analysis_type == 'noise_level':
        tag = f"noise_level_{noise_prefix}" if noise_prefix else analysis_type
    else:
        tag = analysis_type

    if file_tag_suffix:
        tag = f"{tag}_{file_tag_suffix}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csv_save_path = os.path.join(output_dir, f'{tag}_metrics_graph_IGv5.csv')

    if os.path.exists(csv_save_path):
        os.remove(csv_save_path)
        print(f"[Info] Removed existing CSV: {csv_save_path}")

    datasets_to_scan = datasets_to_analyze if datasets_to_analyze is not None else [
        d for d in os.listdir(base_data_path) if os.path.isdir(os.path.join(base_data_path, d))
    ]

    for dataset_name in datasets_to_scan:
        print(f"\n{'=' * 20} Processing WHOLE GRAPH for Dataset: {dataset_name} {'=' * 20}")
        dataset_path = os.path.join(base_data_path, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        if analysis_type == 'noise_level':
            col_name = 'noise_level'
            if noise_prefix is None:
                raise ValueError("When analysis_type == 'noise_level', please provide noise_prefix ('symmetric' or 'asymmetric').")
            prefix = noise_prefix
            files = [f for f in os.listdir(dataset_path) if f.startswith(prefix) and f.endswith('.npz')]
            iter_vals = []
            for f in files:
                m = re.search(r'(\d+(?:\.\d+)?)', f)
                if m:
                    iter_vals.append(float(m.group(1)))
                else:
                    print(f"    [Skip] Cannot parse noise level from filename: {f}")
            iterations = sorted(set(iter_vals))
        else:
            prefix, col_name = 'epoch_', 'epoch'
            files = [f for f in os.listdir(dataset_path) if f.startswith(prefix) and f.endswith('.npz')]
            iter_vals = []
            for f in files:
                m = re.search(r'(\d+)', f)
                if m:
                    iter_vals.append(int(m.group(1)))
                else:
                    print(f"    [Skip] Cannot parse epoch number from filename: {f}")
            iterations = sorted(set(iter_vals))

        for iteration in iterations:
            print(f"  Processing iteration: {iteration}...")
            if analysis_type == 'noise_level':
                if abs(float(iteration)) < 1e-12:
                    iteration_str = '0'
                else:
                    iteration_str = f"{float(iteration):.1f}"
                fname = f"{prefix}{iteration_str}.npz"
            else:
                iteration_str = str(int(iteration))
                fname = f"{prefix}{iteration_str}.npz"

            file_path_npz = os.path.join(dataset_path, fname)
            if not os.path.exists(file_path_npz):
                print(f"    Warning: File not found {file_path_npz}")
                continue

            graph_mat = sp.load_npz(file_path_npz)
            print_sparse_matrix_stats(graph_mat, label=f"{dataset_name}:{fname}")
            gm = calculate_graph_metrics(graph_mat)

            record = {
                'dataset': dataset_name,
                col_name: iteration,
                'MMID': gm.get('MMID', np.nan),
            }

            df_row = pd.DataFrame([record])
            if not os.path.exists(csv_save_path):
                df_row.to_csv(csv_save_path, mode='w', header=True, index=False)
            else:
                df_row.to_csv(csv_save_path, mode='a', header=False, index=False)

            del gm
            del df_row
            del record
            del graph_mat
            gc.collect()
    return csv_save_path


def plot_graph_metrics(
        csv_path,
        analysis_type,
        metrics_to_plot,
        save_dir='visualizations/graph_plots',
        noise_prefix=None,
        datasets_to_plot=None,
        rows_as_datasets=True,
        file_tag_suffix=""
):
    """
    Plot the selected metric(s) using the paper-style formatting in a fixed
    single-row, multi-dataset layout. In this script we use one metric (MMDI)
    across four datasets, so the output is a 1 x 4 figure.
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

    metric_label_map = {
        'MMID': 'MMDI',
    }

    iteration_col = 'noise_level'
    prefix = (noise_prefix or '').strip().lower()
    if prefix in ('symmetric', 'asymmetric'):
        x_label = f"{prefix.capitalize()} Noise Level"
    else:
        x_label = 'Noise Level'

    if datasets_to_plot is None:
        datasets_to_plot = sorted(df['dataset'].unique())
    else:
        existing = set(df['dataset'].unique())
        datasets_to_plot = [d for d in datasets_to_plot if d in existing]

    DISPLAY_NAME_ALIAS = {
        'IGv5_CIFAR10 (full train)_sym_noise': 'CIFAR10',
        'IGv5_MNIST (full train)_sym_noise': 'MNIST',
        'IGv5_Flowers102_sym_noise': 'Flowers102',
        'IGv5_FGVCAircraft_sym_noise': 'FGVC Aircraft',
        'IGv5_FGVCAircraft': 'FGVC Aircraft',
        'IGv4_FGVCAircraft': 'FGVC Aircraft',
        'CIFAR10 (full train)': 'CIFAR10',
        'MNIST (full train)': 'MNIST',
        'FGVCAircraft': 'FGVC Aircraft',
        'Flowers102': 'Flowers102_withpretraining',
        'Flowers102_fromscratch': 'Flowers102',
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

    fig, axes = plt.subplots(
        1, num_datasets,
        figsize=(4 * num_datasets, 2.5),
        squeeze=False,
        constrained_layout=False,
    )

    metric_name = metrics_to_plot[0]
    for j, dataset_name in enumerate(datasets_to_plot):
        ax = axes[0, j]
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
        ax.set_title(display_name, fontsize=LABEL_FONTSIZE + 2, fontweight='bold')

        if j == 0:
            row_label = metric_label_map.get(metric_name, metric_name)
            ax.set_ylabel(row_label, fontsize=LABEL_FONTSIZE, fontweight='bold')
            ax.yaxis.label.set_multialignment('center')

        formatter = ticker.FuncFormatter(lambda x, pos: f'{x:.2f}')
        ax.yaxis.set_major_formatter(formatter)
        ax.tick_params(axis='y', labelleft=True)
        ax.set_xlabel(x_label, fontsize=LABEL_FONTSIZE, fontweight='bold')
        ax.set_xlim(0.0, 0.5)
        ax.set_xticks(np.arange(0.0, 0.51, 0.1))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    fig.subplots_adjust(
        left=0.08,
        bottom=0.38,
        right=0.98,
        top=0.78,
        wspace=0.35,
        hspace=0.4
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    prefix_part = (noise_prefix or '').lower()
    base_name = f'noise_level_{prefix_part}' if prefix_part else 'noise_level'

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
    PROJECT_ROOT = '../PycharmProjects/InfluenceGraphs/'
    ANALYSIS_TYPE = 'noise_level'
    NOISE_PREFIX = 'asymmetric'
    DATA_FOLDER_RELATIVE = 'label_noise_sym/'
    OUTPUT_DIR = 'visualizations/graph_metric_noise'
    RECOMPUTE_CSV = False

    file_tag_suffix = "MMDI"

    metrics_to_plot = ['MMID']

    # PLOT_DATASETS =  ['IGv5_CIFAR10 (full train)_sym_noise', 'IGv4_FGVCAircraft','IGv5_Flowers102_sym_noise', 'IGv5_MNIST (full train)_sym_noise'] #
    PLOT_DATASETS = None # means all
    LAYOUT_ROWS_AS_DATASETS = False  # kept for compatibility; plotting is fixed to a single-row layout

    if RECOMPUTE_CSV:
        csv_path = analyze_and_save_graph_metrics(
            project_root_path=PROJECT_ROOT,
            output_dir=OUTPUT_DIR,
            data_folder_relative=DATA_FOLDER_RELATIVE,
            analysis_type=ANALYSIS_TYPE,
            noise_prefix=NOISE_PREFIX,
            file_tag_suffix=file_tag_suffix
        )
    else:
        tag = f"noise_level_{NOISE_PREFIX}" if NOISE_PREFIX else 'noise_level'
        if file_tag_suffix:
            tag = f"{tag}_{file_tag_suffix}"
        csv_path = os.path.join(OUTPUT_DIR, f'{tag}_metrics_graph_IGv5.csv')

    if os.path.exists(csv_path):
        print(f"\n--- Generating Whole Graph Plot for {ANALYSIS_TYPE.replace('_', ' ').title()} ---")
        plot_graph_metrics(
            csv_path,
            ANALYSIS_TYPE,
            metrics_to_plot,
            OUTPUT_DIR,
            noise_prefix=NOISE_PREFIX,
            datasets_to_plot=PLOT_DATASETS,
            rows_as_datasets=LAYOUT_ROWS_AS_DATASETS,
            file_tag_suffix=file_tag_suffix
        )
    else:
        print(f"CSV not found for plotting: {csv_path}")