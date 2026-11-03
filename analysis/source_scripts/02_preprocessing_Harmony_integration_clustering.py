# =============================================================================
# Global preprocessing, Harmony integration, clustering, and UMAP visualization
# for the merged multi-dataset AnnData object
# =============================================================================

import os
import scanpy as sc
import scanpy.external as sce
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

os.chdir("/home/lishaogang/NEU2026_July")
# =============================================================================
# 0. Set paths and parameters
# =============================================================================

base_dir = "/home/lishaogang/NEU2026_July/GEO_data"
os.chdir(base_dir)

input_h5ad = os.path.join(base_dir, "MEGA_PAN_DISEASE_CLEAN.h5ad")

output_full_h5ad = os.path.join(
    base_dir,
    "adata_mega_Harmony_dataset_clustered_fullgenes.h5ad"
)

output_hvg_h5ad = os.path.join(
    base_dir,
    "adata_mega_HVG_Harmony_dataset_clustered.h5ad"
)

figure_dir = os.path.join(base_dir, "figures")
os.makedirs(figure_dir, exist_ok=True)

batch_key = "dataset"

n_top_genes = 3000
n_pcs = 50
n_neighbors = 30
n_pcs_neighbors = 30
leiden_resolution = 0.6
random_state = 0


# =============================================================================
# 1. Load or copy the merged AnnData object
# =============================================================================
# If adata_mega already exists in memory, use it directly.
# Otherwise, load it from the input h5ad file.

try:
    adata_mega
    print("Using existing adata_mega object from memory.")
except NameError:
    print(f"adata_mega not found in memory. Loading from: {input_h5ad}")
    adata_mega = sc.read_h5ad(input_h5ad)

adata_mega_full = adata_mega.copy()
adata_mega_full.var_names_make_unique()

print(adata_mega_full)


# =============================================================================
# 2. Check batch annotation
# =============================================================================
# The dataset column is used as the batch variable for Harmony integration.

if batch_key not in adata_mega_full.obs.columns:
    raise ValueError(
        f"obs does not contain '{batch_key}', which is required for Harmony integration."
    )

adata_mega_full.obs[batch_key] = (
    adata_mega_full.obs[batch_key]
    .astype(str)
    .astype("category")
)

print("\nDataset distribution:")
print(adata_mega_full.obs[batch_key].value_counts())


# =============================================================================
# 3. Store raw counts in layers['counts']
# =============================================================================
# The input adata_mega.X is expected to contain raw counts.
# The raw count matrix is stored as layers['counts'] before normalization.

if "counts" not in adata_mega_full.layers:
    adata_mega_full.layers["counts"] = adata_mega_full.X.copy()
    print("\nRaw count matrix has been stored in layers['counts'].")
else:
    print("\nlayers['counts'] already exists. It will be used as the raw count matrix.")

# Use the raw count matrix as X for QC and normalization
adata_mega_full.X = adata_mega_full.layers["counts"].copy()


# =============================================================================
# 4. Clean invalid values
# =============================================================================
# Replace NaN or infinite values with zero to avoid errors during QC, PCA, or Harmony.

if sp.issparse(adata_mega_full.X):
    adata_mega_full.X.data = np.nan_to_num(
        adata_mega_full.X.data,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )
else:
    adata_mega_full.X = np.nan_to_num(
        adata_mega_full.X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


# =============================================================================
# 5. Calculate quality-control metrics
# =============================================================================
# Mitochondrial genes are defined by gene symbols starting with "MT-".
# This step generates obs['n_genes_by_counts'] and obs['pct_counts_mt'].

adata_mega_full.var["mt"] = adata_mega_full.var_names.astype(str).str.startswith("MT-")

sc.pp.calculate_qc_metrics(
    adata_mega_full,
    qc_vars=["mt"],
    inplace=True
)

print("\nQC metrics calculated.")
print("Required QC columns:")
print(adata_mega_full.obs[["n_genes_by_counts", "pct_counts_mt"]].head())


# =============================================================================
# 6. Normalize counts and store log-normalized expression
# =============================================================================
# Raw counts are normalized to counts per 10,000 and then log-transformed.
# The resulting log-normalized matrix is stored in layers['logcounts'].

sc.pp.normalize_total(
    adata_mega_full,
    target_sum=1e4
)

sc.pp.log1p(adata_mega_full)

adata_mega_full.layers["logcounts"] = adata_mega_full.X.copy()

print("\nNormalization and log1p transformation finished.")
print("Log-normalized matrix has been stored in layers['logcounts'].")


# =============================================================================
# 7. Select highly variable genes
# =============================================================================
# Highly variable genes are selected from the log-normalized expression matrix.
# batch_key is used to improve robustness across datasets.

sc.pp.highly_variable_genes(
    adata_mega_full,
    n_top_genes=n_top_genes,
    flavor="seurat",
    batch_key=batch_key
)

hvg_number = int(adata_mega_full.var["highly_variable"].sum())
print(f"\nNumber of highly variable genes: {hvg_number}")

if hvg_number == 0:
    raise ValueError("No highly variable genes were detected. Please check the expression matrix.")

adata_mega_hvg = adata_mega_full[:, adata_mega_full.var["highly_variable"]].copy()

# Use log-normalized expression as the working matrix for dimensionality reduction
adata_mega_hvg.X = adata_mega_hvg.layers["logcounts"].copy()

print("\nHVG-restricted AnnData object:")
print(adata_mega_hvg)


# =============================================================================
# 8. Scale HVG matrix
# =============================================================================
# zero_center=False is used to avoid converting a large sparse matrix into a dense matrix.
# This is more memory-efficient for large-scale single-cell datasets.

sc.pp.scale(
    adata_mega_hvg,
    max_value=10,
    zero_center=False
)

print("\nScaling finished.")


# =============================================================================
# 9. PCA
# =============================================================================
# PCA is performed on the scaled HVG matrix.
# This step generates obsm['X_pca'] and varm['PCs'].

sc.tl.pca(
    adata_mega_hvg,
    n_comps=n_pcs,
    svd_solver="arpack",
    random_state=random_state
)

print("\nPCA finished.")
print("obsm keys after PCA:")
print(adata_mega_hvg.obsm.keys())
print("varm keys after PCA:")
print(adata_mega_hvg.varm.keys())


# Plot PCA variance ratio
sc.pl.pca_variance_ratio(
    adata_mega_hvg,
    n_pcs=n_pcs,
    log=True,
    show=False
)

plt.savefig(
    os.path.join(figure_dir, "adata_mega_pca_variance_ratio.pdf"),
    bbox_inches="tight"
)

plt.show()


# =============================================================================
# 10. Harmony integration
# =============================================================================
# Harmony is applied to the PCA space to correct dataset-level batch effects.
# The corrected embedding is stored as obsm['X_pca_harmony'].

sce.pp.harmony_integrate(
    adata_mega_hvg,
    key=batch_key,
    basis="X_pca",
    adjusted_basis="X_pca_harmony",
    max_iter_harmony=30
)

print("\nHarmony integration finished.")
print("obsm keys after Harmony:")
print(adata_mega_hvg.obsm.keys())


# =============================================================================
# 11. Construct neighborhood graph using Harmony-corrected PCA space
# =============================================================================
# This step generates obsp['connectivities'] and obsp['distances'].

sc.pp.neighbors(
    adata_mega_hvg,
    n_neighbors=n_neighbors,
    n_pcs=n_pcs_neighbors,
    use_rep="X_pca_harmony",
    metric="cosine"
)

print("\nNeighborhood graph construction finished.")
print("obsp keys:")
print(adata_mega_hvg.obsp.keys())


# =============================================================================
# 12. Leiden clustering
# =============================================================================

sc.tl.leiden(
    adata_mega_hvg,
    resolution=leiden_resolution,
    key_added="leiden",
    random_state=random_state
)

print("\nCluster distribution:")
print(
    adata_mega_hvg.obs["leiden"]
    .value_counts()
    .sort_index()
)


# =============================================================================
# 13. UMAP embedding
# =============================================================================
# UMAP is computed from the neighbor graph built on Harmony-corrected PCA space.
# This step generates obsm['X_umap'].

sc.tl.umap(
    adata_mega_hvg,
    min_dist=0.35,
    spread=1.0,
    random_state=random_state
)

print("\nUMAP computation finished.")
print("obsm keys after UMAP:")
print(adata_mega_hvg.obsm.keys())


# =============================================================================
# 14. Visualize UMAP
# =============================================================================

plot_columns = [
    "leiden",
    "dataset",
    "orig.ident",
    "Disease",
    "cell_type",
    "cell_subtype",
    "cell_subtype_short"
]

plot_columns = [
    col for col in plot_columns
    if col in adata_mega_hvg.obs.columns
]

print("\nUMAP plotting columns:")
print(plot_columns)

if len(plot_columns) > 0:
    sc.pl.umap(
        adata_mega_hvg,
        color=plot_columns,
        ncols=2,
        frameon=False,
        size=2,
        legend_loc="right margin"
    )

sc.pl.umap(
    adata_mega_hvg,
    color="leiden",
    frameon=False,
    size=2,
    legend_loc="on data",
    legend_fontsize=8,
    legend_fontoutline=2,
    title="Global clustering based on log-normalized counts + Harmony(dataset)"
)


# =============================================================================
# 15. Transfer integration results back to the full-gene object
# =============================================================================
# The full-gene object keeps all genes and contains:
# obsm['X_pca'], obsm['X_pca_harmony'], obsm['X_umap']
# varm['PCs']
# obsp['connectivities'], obsp['distances']
# layers['counts'], layers['logcounts']
# obs['n_genes_by_counts'], obs['pct_counts_mt']

adata_mega_full.obs["leiden"] = adata_mega_hvg.obs["leiden"].copy()

adata_mega_full.obsm["X_pca"] = adata_mega_hvg.obsm["X_pca"].copy()
adata_mega_full.obsm["X_pca_harmony"] = adata_mega_hvg.obsm["X_pca_harmony"].copy()
adata_mega_full.obsm["X_umap"] = adata_mega_hvg.obsm["X_umap"].copy()

adata_mega_full.obsp["connectivities"] = adata_mega_hvg.obsp["connectivities"].copy()
adata_mega_full.obsp["distances"] = adata_mega_hvg.obsp["distances"].copy()

adata_mega_full.uns["neighbors"] = adata_mega_hvg.uns["neighbors"].copy()
adata_mega_full.uns["umap"] = adata_mega_hvg.uns["umap"].copy()

if "leiden" in adata_mega_hvg.uns:
    adata_mega_full.uns["leiden"] = adata_mega_hvg.uns["leiden"].copy()


# =============================================================================
# 16. Store PCA loadings in the full-gene object
# =============================================================================
# PCA loadings are calculated only for HVGs.
# To keep a full-gene AnnData object with varm['PCs'], loadings for non-HVG genes
# are filled with zeros, while HVG loadings are copied from the HVG object.

pcs_hvg = adata_mega_hvg.varm["PCs"]
pcs_full = np.zeros(
    (adata_mega_full.n_vars, pcs_hvg.shape[1]),
    dtype=pcs_hvg.dtype
)

hvg_positions = adata_mega_full.var_names.get_indexer(adata_mega_hvg.var_names)

valid_mask = hvg_positions >= 0
pcs_full[hvg_positions[valid_mask], :] = pcs_hvg[valid_mask, :]

adata_mega_full.varm["PCs"] = pcs_full

print("\nIntegration results have been transferred back to the full-gene object.")


# =============================================================================
# 17. Check final AnnData structure
# =============================================================================

print("\nFinal full-gene AnnData object:")
print(adata_mega_full)

print("\nRequired obsm keys:")
print([key for key in ["X_pca", "X_pca_harmony", "X_umap"] if key in adata_mega_full.obsm])

print("\nRequired varm keys:")
print([key for key in ["PCs"] if key in adata_mega_full.varm])

print("\nRequired layers:")
print([key for key in ["counts", "logcounts"] if key in adata_mega_full.layers])

print("\nRequired obsp keys:")
print([key for key in ["connectivities", "distances"] if key in adata_mega_full.obsp])

print("\nRequired obs columns:")
print(
    [
        key for key in ["n_genes_by_counts", "pct_counts_mt"]
        if key in adata_mega_full.obs.columns
    ]
)


# =============================================================================
# 18. Save integrated objects
# =============================================================================
# Two objects are saved:
# 1. HVG object used for PCA, Harmony, neighbors, clustering, and UMAP.
# 2. Full-gene object containing all genes and all required embeddings/layers/graphs.

adata_mega_hvg.write_h5ad(
    output_hvg_h5ad,
    compression="gzip"
)

adata_mega_full.write_h5ad(
    output_full_h5ad,
    compression="gzip"
)

print("\nSaved integrated AnnData objects:")
print(f"HVG object:       {output_hvg_h5ad}")
print(f"Full-gene object: {output_full_h5ad}")