#!/usr/bin/env python3
"""
Preprocess h5ad files for BMFM training
Handles: NaN cleaning, log transform, binning, train/val split
"""

import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix
import sys

def preprocess_h5ad(input_file, output_file):
    print(f"Loading {input_file}...")
    adata = ad.read_h5ad(input_file)
    print(f"  Shape: {adata.shape}")
    
    # 1. Make unique names
    adata.obs_names_make_unique()
    
    # 2. Convert to dense
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    # 3. Replace NaN
    if np.isnan(X).any():
        print(f"  Replacing NaN...")
        X = np.nan_to_num(X, nan=0.0)
    
    print(f"  Range: {X.min():.2e} to {X.max():.2e}")
    
    # 4. Log2 transform
    print("  Log2 transform...")
    X_log = np.log2(X + 1)
    
    # 5. Bin to 0-49
    print("  Binning to 50 bins...")
    percentiles = np.linspace(0, 100, 51)
    bin_edges = np.percentile(X_log[X_log > 0], percentiles)
    bin_edges[0] = X_log.min() - 0.001
    
    X_binned = np.digitize(X_log, bin_edges) - 1
    X_binned = np.clip(X_binned, 0, 49)
    
    # 6. Convert to sparse
    adata.X = csr_matrix(X_binned)
    sparsity = 100 * (1 - adata.X.nnz / (adata.shape[0] * adata.shape[1]))
    print(f"  Sparsity: {sparsity:.1f}%")
    
    # 7. Add train/val split
    print("  Creating 80/20 split...")
    n = adata.shape[0]
    n_train = int(0.8 * n)
    np.random.seed(1234)
    indices = np.random.permutation(n)
    split = np.array(['val'] * n, dtype=object)
    split[indices[:n_train]] = 'train'
    adata.obs['split_random'] = split
    
    print(f"Saving to {output_file}...")
    adata.write_h5ad(output_file)
    print(f"✓ Done!")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python preprocess_h5ad.py input.h5ad output.h5ad")
        sys.exit(1)
    preprocess_h5ad(sys.argv[1], sys.argv[2])
