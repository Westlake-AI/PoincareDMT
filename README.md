## Complex hierarchical structures analysis in single-cell data with Poincaré deep manifold transformation

## Overview

We propose the Poincaré deep manifold transformation PoincaréDMT method. This approach leverages hyperbolic neural networks to map high-dimensional data to a Poincaré disk, effectively representing continuous hierarchical structures. PoincaréDMT inherits the strength of global structure preservation from a graph Laplacian of the pairwise distance matrix and achieves local structure correction through a dedicated structure module combined with data augmentation, making it adept at representing complex hierarchical data. To integrate datasets from diverse sources into a unified embedding, we alleviate batch effects via integrating a batch graph that accounts for batch IDs to low-dimensional embedding during network training. To explain the important marker genes in the cell differentiation process, we introduce Shapley additive explanations method based on low-dimensional embeddings. The proposed dimensionality reduction and visualization method can be applied to multiple downstream taks, including developmental trajectories inference, pseudotime inference, batch correction, and marker gene selection.

![image](https://github.com/Westlake-AI/PoincareDMT/tree/main/Figures/Framework.png)

## Installation

1. **Install dependencies**:
conda env create -f env.yaml

2. **Activate environment**:
conda activate poincaredmt

## Usage

1. **Dimensionality reduction**:
python main.py

2. **Batch correction**:
python main_batch_correction.py

3. **Gene selection**:
python main_gene_selection.py

4. **Baseline**:
python main_baseline.py
python main_baseline_batch_correction.py

5. **Visualization and Pseudotime inference**:
python main_visualization_pseudotime.py

## Contact

If you have any problem about this package, please create an Issue or send us an Email at:
- **Author**: [Yongjie Xu][xuyongjie@westlake.edu.cn]
- **Author**: [Zelin Zang][zangzelin@westlake.edu.cn]
