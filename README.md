# Setup (CPU)
Code using Python version 3.11.5
- Do git clone stuff as usual
- Highly suggested to create a virtual environment for project, do so by:
> python3.11 -m venv influence_env
- Now start your virtual environment (*You have to do this step everytime you want to run project*)
> source influence_env/bin/activate
- Install dependencies via requirements.txt (For CPU)
> pip install -r requirements.txt

# Setup (GPU, using conda)
Code using Python version 3.11.5
- Do git clone stuff as usual
- Highly suggested to create a virtual environment for project, do so by:
> conda create -n influence_env python=3.11 -y
- Now start your virtual environment (*You have to do this step everytime you want to run project*)
> conda activate influence_env
- Install PyTorch with CUDA
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
- [Optional] Verify that CUDA is working
> python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
- Install dependencies via requirements+cuda.txt (For GPU)
> pip install -r requirements+cuda.txt

# Run
- Go to project directory on your console (Terminal for mac)
> cd {Drag and Drop project folder onto Terminal}
- Run project virtual environment if you chose to set one up
> source influence_env/bin/activate
- Start notebook as usual
> python -m notebook
- Running label noise experiment example (MNIST)
> python main_vision_label_noise.py --dataset MNIST --model_name ShallowMNIST --noise_type symmetric --save_mode store --img_size 28 --log_verbosity 1
- Running train snapshots generation (MNIST)
> python main_vision_snapshot_train.py --dataset MNIST --model_name ShallowMNIST --img_size 28 --log_verbosity 1
- Running batchwise influence estimation for snapshots (MNIST)
> python main_vision_snapshot_BW.py --dataset MNIST


# Influence Graph Statistics Analysis

Use 'graph_statistics_...' scripts to compute graph-level statistics from saved influence graph matrices and generate PDF plots. Input graphs are expected to be sparse adjacency matrices saved with `scipy.sparse.save_npz`.

## Data Layout

The scripts read graph files from folders such as:

```text
InfluenceGraphs/
  snapshots/
    DATASET_NAME/
      epoch_0.npz
      epoch_1.npz
      ...
  label_noise_sym/
    DATASET_NAME/
      symmetric0.npz
      symmetric0.1.npz
      ...
      asymmetric0.npz
      asymmetric0.1.npz
      ...
```

If your paths or dataset names are different, edit the configuration block under `if __name__ == '__main__':` in each script.

## Scripts

### `graph_statistics_NC_AIW_MMDI_epoch.py`

Computes graph statistics over training epochs:

- `MMID` / MMDI
- `Average Path Length`
- `Number of Clusters`
- `Average Intra-cluster Weight`
- `Average Intra-cluster Weight Std`

Run:

```bash
python graph_statistics_NC_AIW_MMDI_epoch.py
```

Example outputs:

```text
visualizations/graph_metric_AISTATS_IGv5_compare/
  epoch_compare_v5_wo_neg_metrics_graph_IGv5.csv
  epoch_compare_v5_wo_neg.pdf
```

### `graph_statistic_plot_comparison.py`

Aggregates graph-metric CSV files computed under different edge-weight thresholds and plots epoch-wise comparisons across datasets.

It expects input CSV files named like:

```text
epoch_10_wo_neg_metrics_graph_IGv5.csv
epoch_20_wo_neg_metrics_graph_IGv5.csv
...
epoch_90_wo_neg_metrics_graph_IGv5.csv
```

Run:

```bash
python graph_statistic_plot_comparison.py
```

Example outputs:

```text
visualizations/graph_metric_AISTATS_IGv5/
  epoch_IGv5_average_wo.csv
  epoch_average_wo.pdf
```

### `graph_statistic_MMDI_noise.py`

Computes and plots MMDI under label-noise experiments.

Run:

```bash
python graph_statistic_MMDI_noise.py
```

Example outputs:

```text
visualizations/graph_metric_noise/
  noise_level_asymmetric_MMDI_metrics_graph_IGv5.csv
  noise_level_asymmetric_MMDI.pdf
```

## Typical Workflow

1. Place `.npz` influence graph files under the corresponding dataset folders.
2. Edit the configuration block at the bottom of the target script.
3. Set `RECOMPUTE_CSV = True` if metrics need to be recomputed.
4. Run the script to generate CSV statistics and PDF plots.


