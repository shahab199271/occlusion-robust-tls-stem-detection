# Occlusion-Robust Stem Detection in TLS Point Clouds

Code accompanying the manuscript:

**Occlusion-Robust Stem Detection in Individual-Tree Terrestrial Laser Scanning Point Clouds Using Graph-Based Deep Learning**

Shahab Alaedin Baloochi, Pasi Raumonen, Kim Calders, and Esa Rahtu

Code author and maintainer: **Shahab Alaedin Baloochi**

## Overview

This repository contains the core implementation used for stem detection in individual-tree terrestrial laser scanning (TLS) point clouds. The method combines a fixed Euclidean k-nearest-neighbour graph, engineered geometric features, graph-based feature extraction, local attention, subgraph-level attention pooling, FiLM conditioning, and geometric post-processing.

The implementation is organised around whole-tree preprocessing and connected subgraph processing. The same fixed geometric graph is used for feature computation and network neighbourhoods, and full-tree inference covers every point exactly once using non-overlapping connected subgraphs.

## Repository contents

| Path | Purpose |
| --- | --- |
| `preprocessing/` | Point-cloud loading, 10D feature construction, fixed k-NN graph construction, TreeQSM label handling, and connected subgraph sampling |
| `models/` | Fixed-graph EdgeConv, local multi-head attention, PMA pooling, FiLM conditioning, and the stem detector assembly |
| `occlusion/` | Synthetic occlusion generation for controlled robustness experiments |
| `inference/` | Exhaustive non-overlapping full-tree inference |
| `postprocessing/` | High-confidence core selection and two-pass axis-envelope expansion |
| `evaluation/` | Point-level metrics, threshold selection, height-stratified recall, stem-height metrics, and stem-volume error metrics |
| `requirements.txt` | Pinned Python environment for a general installation |
| `requirements-lumi.txt` | Additional packages for the LUMI environment |
| `LUMI_ENVIRONMENT.md` | LUMI / ROCm software environment notes |

## Data and labels

The experiments use individual-tree TLS point clouds from Wytham Woods. The study contains 876 trees, split at tree level into 613 training, 131 validation, and 132 test trees.

TreeQSM branch indices are used to define the binary labels:

- branch index `1`: stem
- branch index `>1`: non-stem
- branch index `0`: unsegmented and excluded from training/evaluation

Some converted local CSV files may use a different sentinel, such as `-1`, for excluded points. In that case the sentinel must be passed explicitly through `unsegmented_id`.

The dataset itself is not redistributed in this repository.

## Input representation

Each point is represented by metric local XYZ coordinates together with seven geometric/structural descriptors:

1. relative height
2. radial distance from the tree centre
3. distance to a coarse stem axis
4. angle-weighted depth from the basal region
5. vertical continuity
6. local curvature
7. local verticality

The fixed graph is a symmetric 3D Euclidean k-NN graph with `k = 16`. The same precomputed neighbourhoods are used by the PCA-based descriptors and by the graph network.

## Installation

Python 3.12 was used for the experiments.

For a standard environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
```

For LUMI, use the ROCm-enabled PyTorch container described in [LUMI_ENVIRONMENT.md](LUMI_ENVIRONMENT.md) and install the additional packages from:

```bash
pip install -r requirements-lumi.txt
```

Do not replace the PyTorch build supplied by the LUMI container with a generic pip build.

## Preprocessing

A single tree can be preprocessed directly from the command line:

```bash
python -m preprocessing.build_dataset path/to/tree.csv \
    --output path/to/tree_preprocessed.npz \
    --require-labels \
    --validate
```

For files where the excluded branch index has been remapped to `-1`:

```bash
python -m preprocessing.build_dataset path/to/tree.csv \
    --output path/to/tree_preprocessed.npz \
    --unsegmented-id -1 \
    --require-labels \
    --validate
```

The preprocessing stage builds the fixed graph, computes the 10D representation, attaches binary labels when available, and validates compatibility with the connected 8,192-point subgraph workflow.

The same functionality is available from Python:

```python
from preprocessing import build_tree_dataset

tree = build_tree_dataset(
    "path/to/tree.csv",
    k=16,
    subgraph_size=8192,
    require_labels=True,
)
```

## Model

The stem detector contains three fixed-graph EdgeConv layers followed by local multi-head attention. The three EdgeConv outputs are concatenated into a multi-scale representation, refined with attention on the unchanged k-NN graph, and pooled with two learned PMA seed queries. The resulting subgraph context is used for FiLM conditioning before point-wise classification.

The classification head is:

```text
448 -> 256 -> 128 -> 64 -> 1
```

with ReLU activations and dropout.

Model components are exposed through the `models` package so that they can be assembled and tested independently.

## Full-tree inference

Inference processes a tree through connected, non-overlapping subgraphs until every point has been covered exactly once. Predictions are returned in the original point order.

With a configured model and a preprocessed tree:

```python
from inference import infer_tree_dataset

result = infer_tree_dataset(
    model,
    tree,
    threshold=0.50,
)
```

The primary decision threshold reported in the manuscript is `0.50`. A sensitivity analysis is also available at `0.63`.

## Synthetic occlusion

The `occlusion` package implements the controlled occlusion protocol used to test robustness to missing data. Evaluation scenarios contain two horizontal bands together with a height-dependent number of additional spatially coherent regions. Training variants use the same framework with stochastic band activation.

Occlusion is applied before rebuilding the point representation. Features and the Euclidean k-NN graph must therefore be recomputed from the surviving points.

## Geometric post-processing

The repository implements the first two geometric post-processing stages:

1. selection of a base-anchored high-confidence connected stem core
2. two-pass centreline/envelope estimation and connected graph expansion

The envelope uses adaptive vertical bins, median horizontal centres, a 95th-percentile radius constrained to 0.05--0.55 m, nearest-bin propagation, and a 7-bin moving average.

The final TreeQSM-based patch/cylinder filtering and cylinder reconstruction used in the study are part of the TreeQSM workflow and are not duplicated in this repository.

## Evaluation

The `evaluation` package provides:

- accuracy, precision, recall, F1, and F2
- pooled and mean-per-tree classification summaries
- validation-set threshold search on `0.00, 0.01, ..., 1.00`
- stem recall in lower, middle, and upper thirds of tree height
- tree-level stem-height error summaries
- tree-level stem-volume relative error summaries

For final volume analysis, predicted and reference volumes should be obtained with the same TreeQSM reconstruction procedure before applying the volume-error utilities.

## Reproducibility notes

The repository keeps tree point order fixed throughout preprocessing, inference, and evaluation. Unsegmented points remain in geometric arrays so graph indexing is stable, but they are masked from label-based metrics. Synthetic-occlusion scenarios can be serialized and reused so different methods are evaluated under identical visibility conditions.

The LUMI software environment used as the reference HPC setup is documented separately in [LUMI_ENVIRONMENT.md](LUMI_ENVIRONMENT.md).

## Citation

If you use this code, please cite the accompanying manuscript:

**Shahab Alaedin Baloochi, Pasi Raumonen, Kim Calders, and Esa Rahtu.  
“Occlusion-Robust Stem Detection in Individual-Tree Terrestrial Laser Scanning Point Clouds Using Graph-Based Deep Learning.”**

Publication metadata will be added here when available.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.
