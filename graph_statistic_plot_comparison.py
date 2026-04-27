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
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse import triu

EXTERNAL_FINAL_CSV_PATH = '../PycharmProjects/InfluenceGraphs/results/epoch_metrics_graph_final.csv'

def make_undirected_sparse(sub_graph_mat):
    """Symmetrize a sparse adjacency matrix using edge presence only."""
    if sub_graph_mat.nnz == 0:
        return sp.csr_matrix(sub_graph_mat.shape, dtype=np.int8)
    A = sub_graph_mat.tocsr()
    A_bin = A.copy()
    A_bin.data = np.ones_like(A_bin.data, dtype=np.int8)
    A_und = ((A_bin + A_bin.T) > 0).astype(np.int8).tocsr()
    A_und.eliminate_zeros()
    return A_und

def compute_mmid(sub_graph_mat):
    """Mean of mean incoming edge weights over all nodes, counting empty nodes as zero.
    Temporary debug behavior: take absolute values of edge weights before computing MMDI.
    """
    num_nodes = sub_graph_mat.shape[0]
    if num_nodes == 0:
        return np.nan

    mat_csc = sub_graph_mat.tocsc(copy=True)
    ##### Begin: take absolute values of edge weights before calculating MMDI #####
    # Temporary debug behavior: convert negative edge weights to absolute values
    # so that all edge weights participate as non-negative magnitudes in MMDI.
    if mat_csc.nnz > 0:
        mat_csc.data = np.abs(mat_csc.data)
    ##### End: take absolute values of edge weights before calculating MMDI #####

    in_sum = np.asarray(mat_csc.sum(axis=0)).ravel()
    in_cnt = mat_csc.getnnz(axis=0)
    mean_in = np.zeros(num_nodes, dtype=float)
    nz_in = in_cnt > 0
    mean_in[nz_in] = in_sum[nz_in] / in_cnt[nz_in]
    return float(mean_in.mean())


def exact_average_path_length_sparse(sub_graph_mat):
    """
    Exact unweighted average shortest path length on the undirected graph.
    Average only over connected node pairs, which matches the common
    disconnected-graph convention and is appropriate for small graphs.
    """
    if sub_graph_mat.shape[0] == 0:
        return np.nan

    A_und = make_undirected_sparse(sub_graph_mat)
    n = A_und.shape[0]
    if n <= 1 or A_und.nnz == 0:
        return np.nan

    n_components, labels = connected_components(A_und, directed=False, return_labels=True)
    if n_components == 0:
        return np.nan

    total_dist_sum = 0.0
    total_pair_count = 0

    for comp_id in range(n_components):
        nodes = np.where(labels == comp_id)[0]
        comp_size = nodes.size
        if comp_size <= 1:
            continue

        A_comp = A_und[nodes][:, nodes].tocsr()
        dists = shortest_path(A_comp, directed=False, unweighted=True)
        dists = np.asarray(dists, dtype=np.float64)

        upper = triu(sp.csr_matrix(dists), k=1).tocoo()
        if upper.nnz == 0:
            continue

        finite_mask = np.isfinite(upper.data)
        if not np.any(finite_mask):
            continue

        total_dist_sum += float(np.sum(upper.data[finite_mask]))
        total_pair_count += int(np.sum(finite_mask))

    if total_pair_count == 0:
        return np.nan
    return float(total_dist_sum / total_pair_count)


def compute_number_of_clusters(sub_graph_mat):
    """Exact number of connected components on the undirected graph."""
    if sub_graph_mat.shape[0] == 0:
        return np.nan
    undirected = make_undirected_sparse(sub_graph_mat)
    n_comp, _ = connected_components(undirected, directed=False, return_labels=True)
    return float(n_comp)


def compute_cluster_edge_weight_stats(sub_graph_mat):
    """
    Compute cluster-level intra-cluster edge-weight statistics on the undirected graph.
    For each connected component (cluster), collect all directed edges whose two endpoints
    both belong to that cluster, then compute that cluster's mean and std of edge weights.
    Finally, return the average of cluster means and the average of cluster stds.
    """
    if sub_graph_mat.shape[0] == 0:
        return np.nan, np.nan
    if sub_graph_mat.nnz == 0:
        return np.nan, np.nan

    undirected = make_undirected_sparse(sub_graph_mat)
    n_comp, labels = connected_components(undirected, directed=False, return_labels=True)
    if n_comp == 0:
        return np.nan, np.nan

    coo = sub_graph_mat.tocoo()
    if coo.nnz == 0:
        return np.nan, np.nan

    ##### Begin: take absolute values of edge weights before intra-cluster statistics #####
    # Temporary debug behavior: convert negative intra-cluster edge weights to
    # absolute values so that mean/std are computed on edge-weight magnitudes.
    if coo.nnz > 0:
        coo.data = np.abs(coo.data)
    ##### End: take absolute values of edge weights before intra-cluster statistics #####

    cluster_means = []
    cluster_stds = []

    for comp_id in range(n_comp):
        nodes = np.where(labels == comp_id)[0]
        if nodes.size == 0:
            continue
        node_mask = np.zeros(sub_graph_mat.shape[0], dtype=bool)
        node_mask[nodes] = True
        edge_mask = node_mask[coo.row] & node_mask[coo.col]
        if not np.any(edge_mask):
            continue
        cluster_weights = coo.data[edge_mask]
        cluster_means.append(float(np.mean(cluster_weights)))
        cluster_stds.append(float(np.std(cluster_weights)))

    if len(cluster_means) == 0:
        return np.nan, np.nan
    return float(np.mean(cluster_means)), float(np.mean(cluster_stds))

def threshold_sparse_matrix_abs(sub_graph_mat, threshold):
    """Keep only edges with |weight| > threshold, preserving sparse format."""
    if sub_graph_mat.nnz == 0:
        return sub_graph_mat.copy()
    coo = sub_graph_mat.tocoo()
    ##### Begin: take absolute values of edge weights before threshold filtering #####
    # Temporary debug behavior: convert negative edge weights to absolute values
    # so that thresholding and retained weights both use absolute magnitudes.
    if coo.nnz > 0:
        coo.data = np.abs(coo.data)
    ##### End: take absolute values of edge weights before threshold filtering #####
    keep = coo.data > threshold
    if not np.any(keep):
        return sp.csr_matrix(sub_graph_mat.shape, dtype=sub_graph_mat.dtype)
    return sp.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=sub_graph_mat.shape)


def filter_matrix_by_quantile(sub_graph_mat, quantile=None):
    """
    Return a filtered sparse matrix by absolute-weight quantile.
    If quantile is None, keep the original graph with absolute-valued weights.
    Example: quantile=0.60 keeps edges with absolute-valued weight > q60 threshold.
    """
    if quantile is None:
        filtered_mat = sub_graph_mat.tocsr(copy=True)
        ##### Begin: take absolute values of edge weights before returning the filtered matrix #####
        # Temporary debug behavior: convert negative edge weights to absolute values
        # so that the returned graph keeps absolute-magnitude weights.
        if filtered_mat.nnz > 0:
            filtered_mat.data = np.abs(filtered_mat.data)
        ##### End: take absolute values of edge weights before returning the filtered matrix #####
        return filtered_mat
    if sub_graph_mat.nnz == 0:
        return sp.csr_matrix(sub_graph_mat.shape, dtype=sub_graph_mat.dtype)
    q = float(quantile)
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"Quantile must be between 0 and 1, got {quantile}")

    filtered_mat = sub_graph_mat.tocsr(copy=True)
    ##### Begin: take absolute values of edge weights before computing the quantile threshold #####
    # Temporary debug behavior: convert negative edge weights to absolute values
    # so that quantile thresholds are computed on absolute-magnitude weights.
    if filtered_mat.nnz > 0:
        filtered_mat.data = np.abs(filtered_mat.data)
    ##### End: take absolute values of edge weights before computing the quantile threshold #####
    if filtered_mat.nnz == 0:
        return sp.csr_matrix(sub_graph_mat.shape, dtype=sub_graph_mat.dtype)

    thr = float(np.quantile(filtered_mat.data, q))
    return threshold_sparse_matrix_abs(filtered_mat, thr)


def compute_number_of_clusters_multi_threshold(sub_graph_mat, edge_quantiles=None):
    """
    Compute Number of Clusters as the average connected-component count over
    multiple user-specified quantile thresholds. If edge_quantiles is None or empty,
    use the original graph only.
    """
    if edge_quantiles is None or len(edge_quantiles) == 0:
        return compute_number_of_clusters(sub_graph_mat)

    comp_counts = []
    for q in edge_quantiles:
        filtered_mat = filter_matrix_by_quantile(sub_graph_mat, quantile=q)
        comp_counts.append(compute_number_of_clusters(filtered_mat))
        del filtered_mat
        gc.collect()

    if len(comp_counts) == 0:
        return np.nan
    return float(np.mean(comp_counts))


def compute_cluster_edge_weight_stats_multi_threshold(sub_graph_mat, edge_quantiles=None):
    """
    Compute cluster-level intra-cluster edge-weight statistics as the average over
    multiple user-specified quantile thresholds. If edge_quantiles is None or empty,
    use the original graph only.
    """
    if edge_quantiles is None or len(edge_quantiles) == 0:
        return compute_cluster_edge_weight_stats(sub_graph_mat)

    mean_values = []
    std_values = []
    for q in edge_quantiles:
        filtered_mat = filter_matrix_by_quantile(sub_graph_mat, quantile=q)
        cluster_mean, cluster_std = compute_cluster_edge_weight_stats(filtered_mat)
        mean_values.append(cluster_mean)
        std_values.append(cluster_std)
        del filtered_mat
        gc.collect()

    if len(mean_values) == 0:
        return np.nan, np.nan
    return float(np.nanmean(mean_values)), float(np.nanmean(std_values))


def compute_average_path_length_multi_threshold(sub_graph_mat, edge_quantiles=None):
    """
    Compute Average Path Length as the average over multiple user-specified
    quantile thresholds. If edge_quantiles is None or empty, use the original graph only.
    """
    if edge_quantiles is None or len(edge_quantiles) == 0:
        return exact_average_path_length_sparse(sub_graph_mat)

    apl_values = []
    for q in edge_quantiles:
        filtered_mat = filter_matrix_by_quantile(sub_graph_mat, quantile=q)
        apl_values.append(exact_average_path_length_sparse(filtered_mat))
        del filtered_mat
        gc.collect()

    if len(apl_values) == 0:
        return np.nan
    return float(np.nanmean(apl_values))


def calculate_graph_metrics(sub_graph_mat, apl_quantiles=None, nc_quantiles=None):
    """
    Compute MMDI on the original graph.
    Compute APL as the average over multiple quantile-filtered graphs controlled by apl_quantiles.
    Compute Number of Clusters as the average over multiple quantile-filtered graphs controlled by nc_quantiles.
    """
    metrics = {}
    num_nodes = sub_graph_mat.shape[0]
    num_edges = sub_graph_mat.nnz
    if num_nodes == 0:
        return None

    metrics['num_nodes'] = num_nodes
    metrics['num_edges'] = num_edges

    metrics['MMID'] = compute_mmid(sub_graph_mat)

    metrics['Average Path Length'] = compute_average_path_length_multi_threshold(
        sub_graph_mat,
        edge_quantiles=apl_quantiles,
    )

    metrics['Number of Clusters'] = compute_number_of_clusters_multi_threshold(
        sub_graph_mat,
        edge_quantiles=nc_quantiles,
    )

    cluster_edge_weight_mean, cluster_edge_weight_std = compute_cluster_edge_weight_stats_multi_threshold(
        sub_graph_mat,
        edge_quantiles=nc_quantiles,
    )
    metrics['Average Intra-cluster Weight'] = cluster_edge_weight_mean
    metrics['Average Intra-cluster Weight Std'] = cluster_edge_weight_std

    return metrics


# --- External CSV helpers ---

def load_external_final_metrics(csv_path):
    """Load the external final CSV used for selected datasets."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"External final CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def get_external_metrics_row(external_df, dataset_name, epoch_value):
    """
    Fetch one row from the external final CSV for a given dataset and epoch.
    Expected columns: dataset, epoch, MMID, Average Path Length, Number of Clusters.
    """
    matched = external_df[
        (external_df['dataset'] == dataset_name) &
        (external_df['epoch'] == int(epoch_value))
    ]
    if matched.empty:
        return None
    return matched.iloc[0]


def analyze_and_save_graph_metrics(
        project_root_path,
        output_dir,
        data_folder_relative,
        datasets_to_analyze,
        dataset_apl_quantile_lists=None,
        dataset_nc_quantiles=None,
        file_tag_suffix="compare_exact"
):
    """
    Analyze graph metrics for multiple dataset folders and save one combined CSV.
    MMDI is computed on the original graph.
    APL is computed as the average over multiple per-dataset quantile thresholds.
    Number of Clusters is computed as the average over multiple per-dataset quantile thresholds.
    """
    base_data_path = os.path.join(project_root_path, data_folder_relative)
    tag = f"epoch_{file_tag_suffix}" if file_tag_suffix else "epoch"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csv_save_path = os.path.join(output_dir, f'{tag}_metrics_graph_IGv5.csv')

    if os.path.exists(csv_save_path):
        os.remove(csv_save_path)
        print(f"[Info] Removed existing CSV: {csv_save_path}")

    external_csv_datasets = {
        'Flowers102_withpretraining',
        'Flowers102',
        'FGVCAircraft',
        'FGVCAircraft_dataaug',
    }
    external_final_df = None
    if any(d in external_csv_datasets for d in datasets_to_analyze):
        external_final_df = load_external_final_metrics(EXTERNAL_FINAL_CSV_PATH)

    for dataset_name in datasets_to_analyze:
        print(f"\n{'=' * 20} Processing WHOLE GRAPH for Dataset: {dataset_name} {'=' * 20}")
        dataset_path = os.path.join(base_data_path, dataset_name)
        if not os.path.isdir(dataset_path):
            print(f"  [Skip] Not a directory: {dataset_path}")
            continue

        dataset_apl_qs = None
        if dataset_apl_quantile_lists is not None:
            dataset_apl_qs = dataset_apl_quantile_lists.get(dataset_name, None)

        dataset_nc_qs = None
        if dataset_nc_quantiles is not None:
            dataset_nc_qs = dataset_nc_quantiles.get(dataset_name, None)

        use_external_csv = dataset_name in external_csv_datasets
        if use_external_csv:
            print(f"  [Config] {dataset_name} will be read from external CSV: {EXTERNAL_FINAL_CSV_PATH}")
        else:
            print(f"  [Config] APL edge quantiles for {dataset_name}: {dataset_apl_qs}")
            print(f"  [Config] NC edge quantiles for {dataset_name}: {dataset_nc_qs}")

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
            iteration_str = str(int(iteration))
            fname = f"{prefix}{iteration_str}.npz"
            file_path_npz = os.path.join(dataset_path, fname)
            if not os.path.exists(file_path_npz):
                print(f"    Warning: File not found {file_path_npz}")
                continue

            if use_external_csv:
                ext_row = get_external_metrics_row(external_final_df, dataset_name, iteration)
                if ext_row is None:
                    print(f"    [Skip] No external CSV row for {dataset_name} epoch {iteration}.")
                    continue

                record = {
                    'dataset': dataset_name,
                    col_name: iteration,
                    'apl_edge_quantiles': str(dataset_apl_qs),
                    'nc_edge_quantiles': str(dataset_nc_qs),
                    'MMID': ext_row.get('MMID', np.nan),
                    'Average Path Length': ext_row.get('Average Path Length', np.nan),
                    'Number of Clusters': ext_row.get('Number of Clusters', np.nan),
                    'Average Intra-cluster Weight': ext_row.get('Average Intra-cluster Weight', np.nan),
                    'Average Intra-cluster Weight Std': ext_row.get('Average Intra-cluster Weight Std', np.nan),
                }
            else:
                graph_mat = sp.load_npz(file_path_npz)
                gm = calculate_graph_metrics(
                    graph_mat,
                    apl_quantiles=dataset_apl_qs,
                    nc_quantiles=dataset_nc_qs,
                )

                record = {
                    'dataset': dataset_name,
                    col_name: iteration,
                    'apl_edge_quantiles': str(dataset_apl_qs),
                    'nc_edge_quantiles': str(dataset_nc_qs),
                    'MMID': gm.get('MMID', np.nan),
                    'Average Path Length': gm.get('Average Path Length', np.nan),
                    'Number of Clusters': gm.get('Number of Clusters', np.nan),
                    'Average Intra-cluster Weight': gm.get('Average Intra-cluster Weight', np.nan),
                    'Average Intra-cluster Weight Std': gm.get('Average Intra-cluster Weight Std', np.nan),
                }

            df_row = pd.DataFrame([record])
            if not os.path.exists(csv_save_path):
                df_row.to_csv(csv_save_path, mode='w', header=True, index=False)
            else:
                df_row.to_csv(csv_save_path, mode='a', header=False, index=False)

            if not use_external_csv:
                del gm
                del graph_mat
            del df_row
            del record
            gc.collect()

    return csv_save_path


def plot_compare_metrics(
        csv_path,
        metrics_to_plot,
        dataset_pairs,
        save_dir,
        file_tag_suffix="compare_exact"
):
    """
    Plot a 2 x N comparison figure using a unified paper-style format.
    Each row corresponds to one dataset pair. Each column corresponds to one metric.
    Within each small subplot, the two datasets in the pair are drawn with twin y-axes:
    left axis for the first dataset, right axis for the second dataset.
    The left and right axes use different colors, and their y-axis limits are matched
    using the larger combined range so that the two curves are visually comparable.
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
        'Average Path Length': 'Avg. Path Len.',
        'MMID': 'MMDI',
        'Number of Clusters': 'Num. Clusters',
        'Average Intra-cluster Weight': 'Avg. Intra-cluster Weight',
    }

    DISPLAY_NAME_ALIAS = {
        'Flowers102': 'Flowers102  w/o pre-training',
        'Flowers102_withpretraining': 'Flowers102  w. pre-training',
        'IGv5_Flowers102': 'Flowers102  w/o pre-training',
        'IGv5_Flowers102_withpretraining': 'Flowers102  w. pre-training',

        'FGVCAircraft': 'FGVC Aircraft  w/o data-augmentation', # augmentation
        'FGVCAircraft_dataaug': 'FGVC Aircraft  w. data-augmentation',
        'IGv5_FGVCAircraft': 'FGVC Aircraft  w/o data-augmentation',
        'IGv5_FGVCAircraft_dataaug': 'FGVC Aircraft  w. data-augmentation',  # v5
    }

    LEFT_YLABEL_ALIAS = {
        'Flowers102': 'Flowers102  w/o pre-training', # v4
        'Flowers102_withpretraining': 'Flowers102  w. pre-training', # v4
        'IGv5_Flowers102': 'Flowers102  w/o pre-training', # v5
        'IGv5_Flowers102_withpretraining': 'Flowers102  w. pre-training', # v5

        'FGVCAircraft': 'FGVC Aircraft  w/o data-augmentation', # v4
        'FGVCAircraft_dataaug': 'FGVC Aircraft  w. data-augmentation', # v4
        'IGv5_FGVCAircraft': 'FGVC Aircraft  w/o data-augmentation', # v5
        'IGv5_FGVCAircraft_dataaug': 'FGVC Aircraft  w. data-augmentation',  # v5
    }

    if len(dataset_pairs) != 2:
        raise ValueError(f"This figure is fixed to 2 rows. Please provide exactly 2 dataset pairs, got {len(dataset_pairs)}.")

    num_metrics = len(metrics_to_plot)
    fig_width = 4 * num_metrics
    fig, axes = plt.subplots(
        2, num_metrics,
        figsize=(fig_width, 6.5),
        squeeze=False,
        constrained_layout=False,
    )

    left_color = 'tab:red'
    right_color = 'k'
    left_line_kwargs = dict(marker='o', linestyle='-', markersize=5, linewidth=2, color=left_color)
    right_line_kwargs = dict(marker='s', linestyle='--', markersize=5, linewidth=2, color=right_color)
    row_legend_handles = {}
    row_legend_labels = {}

    for i, pair in enumerate(dataset_pairs):
        if len(pair) != 2:
            raise ValueError(f"Each dataset pair must contain exactly 2 dataset names, got: {pair}")
        left_dataset, right_dataset = pair

        for j, metric_name in enumerate(metrics_to_plot):
            ax = axes[i, j]
            ax_right = ax.twinx()

            left_df = df[(df['dataset'] == left_dataset) & (df[metric_name].notna())].sort_values(by='epoch')
            right_df = df[(df['dataset'] == right_dataset) & (df[metric_name].notna())].sort_values(by='epoch')

            left_std_col = None
            right_std_col = None

            # Use the natural x-axis range of the available data after CSV generation.

            if left_df.empty and right_df.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                continue

            left_handle = None
            right_handle = None
            if not left_df.empty:
                left_handle, = ax.plot(left_df['epoch'], left_df[metric_name], **left_line_kwargs)
            if not right_df.empty:
                right_handle, = ax_right.plot(right_df['epoch'], right_df[metric_name], **right_line_kwargs)

            if i not in row_legend_handles:
                row_legend_handles[i] = []
                row_legend_labels[i] = []
            if left_handle is not None and left_dataset not in row_legend_labels[i]:
                row_legend_handles[i].append(left_handle)
                row_legend_labels[i].append(left_dataset)
            if right_handle is not None and right_dataset not in row_legend_labels[i]:
                row_legend_handles[i].append(right_handle)
                row_legend_labels[i].append(right_dataset)

            # Match the left/right y-axis limits using the combined finite range.
            combined_vals = []
            if not left_df.empty:
                combined_vals.extend(left_df[metric_name].dropna().tolist())
            if not right_df.empty:
                combined_vals.extend(right_df[metric_name].dropna().tolist())
            combined_vals = np.asarray(combined_vals, dtype=float)
            finite_vals = combined_vals[np.isfinite(combined_vals)]
            y_axis_min = np.nan
            y_axis_max = np.nan
            min_metric_value = np.nan
            if finite_vals.size > 0:
                min_metric_value = float(np.min(finite_vals))
                y_min = float(np.min(finite_vals))
                y_max = float(np.max(finite_vals))
                if y_min == y_max:
                    delta = 1.0 if y_min == 0 else abs(y_min) * 0.05
                    y_min -= delta
                    y_max += delta
                else:
                    pad = 0.05 * (y_max - y_min)
                    y_min -= pad
                    y_max += pad
                y_axis_min = y_min
                y_axis_max = y_max
                ax.set_ylim(y_axis_min, y_axis_max)
                ax_right.set_ylim(y_axis_min, y_axis_max)

            ax.tick_params(axis='both', labelsize=TICK_FONTSIZE, colors='k')
            ax.spines['left'].set_color(left_color)
            ax_right.spines['right'].set_color(right_color)
            ax.yaxis.label.set_color('k')
            ax_right.tick_params(axis='y', labelright=False, colors='k')
            if np.isfinite(min_metric_value):
                current_ticks = ax.get_yticks()
                tick_values = current_ticks[current_ticks >= min_metric_value]
                if np.isfinite(y_axis_max):
                    tick_values = tick_values[tick_values <= y_axis_max]
                tick_values = np.sort(np.unique(tick_values))
                tick_values = np.sort(np.unique(np.append(tick_values, min_metric_value)))
                if np.isfinite(y_axis_max):
                    tick_values = tick_values[tick_values <= y_axis_max]
                if tick_values.size >= 2 and np.isclose(tick_values[0], min_metric_value):
                    nearest_gap = tick_values[1] - tick_values[0]
                    if tick_values.size >= 3:
                        ref_spacing = np.min(np.diff(tick_values[1:]))
                    else:
                        ref_spacing = max(abs(tick_values[1]) * 0.05, 1e-8)
                    if nearest_gap < 0.25 * ref_spacing:
                        tick_values = np.delete(tick_values, 1)
                ax.set_yticks(tick_values)
                ax_right.set_yticks(tick_values)
                if np.isfinite(y_axis_min) and np.isfinite(y_axis_max):
                    ax.set_ylim(y_axis_min, y_axis_max)
                    ax_right.set_ylim(y_axis_min, y_axis_max)

            col_label = metric_label_map.get(metric_name, metric_name)
            ax.set_title(col_label, fontsize=LABEL_FONTSIZE + 2, fontweight='bold')

            ax.set_ylabel('')
            ax_right.set_ylabel('')

            ax.set_xlabel('Epoch', fontsize=LABEL_FONTSIZE, fontweight='bold')

            formatter = ticker.FuncFormatter(lambda x, pos: f'{x:.2f}')
            ax.yaxis.set_major_formatter(formatter)
            ax_right.yaxis.set_major_formatter(formatter)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.setp(ax.get_xticklabels(), rotation=0, ha='center', color='k')
            plt.setp(ax.get_yticklabels(), color='k')
            plt.setp(ax_right.get_yticklabels(), color='k')

    legend_ncol = max(
        (len(row_legend_labels.get(row_idx, [])) for row_idx in range(len(dataset_pairs))),
        default=1,
    )

    legend_left = 0.08
    legend_width = 0.87
    legend_height = 0.04

    for row_idx in range(len(dataset_pairs)):
        handles = row_legend_handles.get(row_idx, [])
        labels = row_legend_labels.get(row_idx, [])
        if not handles:
            continue
        display_labels = [DISPLAY_NAME_ALIAS.get(lbl, lbl) for lbl in labels]
        legend_bottom = 0.54 if row_idx == 0 else 0.015
        fig.legend(
            handles,
            display_labels,
            loc='lower left',
            bbox_to_anchor=(legend_left, legend_bottom, legend_width, legend_height),
            mode='expand',
            ncol=legend_ncol,
            frameon=True,
            fancybox=False,
            edgecolor='black',
            prop={'size': 18, 'weight': 'bold'},
            borderaxespad=0.0,
        )

    fig.subplots_adjust(
        left=0.08,
        bottom=0.20,
        right=0.95,
        top=0.92,
        wspace=0.25,
        hspace=1.5,
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = f'epoch_{file_tag_suffix}.pdf' if file_tag_suffix else 'epoch_compare_v5_wo_abs.pdf'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, format='pdf', dpi=300)
    print(f"\n--- Compare figure saved to: {save_path} ---")
    plt.close()


if __name__ == '__main__':
    PROJECT_ROOT = '/Users/jiayin/PycharmProjects/InfluenceGraphs/'
    DATA_FOLDER_RELATIVE = 'snapshots/'
    OUTPUT_DIR = 'visualizations/graph_metric_AISTATS_IGv5_compare'
    RECOMPUTE_CSV = False

    # You may specify 4 / 6 / 8 dataset folders here for computation.
    DATASETS_TO_ANALYZE = [
        'IGv5_Flowers102', # flower V5 wo pretraining
        'IGv5_Flowers102_withpretraining', # flower V5 with pretraining
        'IGv5_FGVCAircraft', # FGVC V5 wo data aug
        'IGv5_FGVCAircraft_dataaug',  # FGVC V5 with data aug
    ]

    # Set a list of quantiles for APL per dataset. APL is averaged over these thresholds.
    DATASET_APL_QUANTILE_LISTS = {
        'IGv5_Flowers102': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        'IGv5_Flowers102_withpretraining': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        'IGv5_FGVCAircraft':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], #[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],
        'IGv5_FGVCAircraft_dataaug': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    }

    # Number of Clusters is averaged over these manually specified thresholds.
    DATASET_NC_QUANTILES = {
        'IGv5_Flowers102': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        'IGv5_Flowers102_withpretraining': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        'IGv5_FGVCAircraft':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], #[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],
        'IGv5_FGVCAircraft_dataaug': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    }

    # Only four datasets are plotted:
    # two rows, each row is one dataset pair.
    PLOT_DATASET_PAIRS = [
        ('IGv5_Flowers102_withpretraining','IGv5_Flowers102'),
        ('IGv5_FGVCAircraft_dataaug','IGv5_FGVCAircraft'),
    ]

    METRICS_TO_PLOT = [
        # 'Average Path Length',
        'Number of Clusters',
        'Average Intra-cluster Weight',
        'MMID',
    ]

    FILE_TAG_SUFFIX = 'compare_v5_wo_neg'

    csv_filename = f"epoch_{FILE_TAG_SUFFIX}_metrics_graph_IGv5.csv" if FILE_TAG_SUFFIX else "epoch_metrics_graph_IGv5.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)

    if RECOMPUTE_CSV:
        csv_path = analyze_and_save_graph_metrics(
            project_root_path=PROJECT_ROOT,
            output_dir=OUTPUT_DIR,
            data_folder_relative=DATA_FOLDER_RELATIVE,
            datasets_to_analyze=DATASETS_TO_ANALYZE,
            dataset_apl_quantile_lists=DATASET_APL_QUANTILE_LISTS,
            dataset_nc_quantiles=DATASET_NC_QUANTILES,
            file_tag_suffix=FILE_TAG_SUFFIX,
        )
    else:
        print(f"[Info] Reusing existing CSV: {csv_path}")

    if os.path.exists(csv_path):
        plot_compare_metrics(
            csv_path=csv_path,
            metrics_to_plot=METRICS_TO_PLOT,
            dataset_pairs=PLOT_DATASET_PAIRS,
            save_dir=OUTPUT_DIR,
            file_tag_suffix=FILE_TAG_SUFFIX,
        )
    else:
        print(f"CSV not found for plotting: {csv_path}")