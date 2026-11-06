# =============================================================================
# Multi-center single-cell data loading, gene filtering, dataset merging,
# and UpSet plot generation
# =============================================================================

import os
import scanpy as sc
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import from_contents, plot
os.chdir("/home/lishaogang/NEU2026_July")

# =============================================================================
# 1. Set paths and input files
# =============================================================================

# Directory containing the input h5ad files
base_dir = "/home/lishaogang/NEU2026_July/GEO_data/"
data_dir = base_dir

# Set the working directory
os.chdir(base_dir)

# Whitelist of h5ad files included in this analysis
target_files = [
    "GSE133028_raw_count.h5ad",
    "GSE134577_raw_count.h5ad",
    "GSE138266_raw_count.h5ad",
    "GSE141578_raw_count.h5ad",
    "GSE163005_raw_count.h5ad",
    "GSE186802_raw_count_subset.h5ad",
    "GSE194078_raw_count.h5ad",
    "GSE200164_raw_count.h5ad",
    "GSE243905_raw_count.h5ad",
    "GSE244499_raw_count.h5ad",
    "GSE247134_raw_count.h5ad"
]

# Output path for the merged AnnData object
output_mega_path = os.path.join(
    base_dir,
    "MEGA_PAN_DISEASE_CLEAN.h5ad"
)

# Output path for the UpSet plot
output_fig_path = os.path.join(
    base_dir,
    "datasets_expressed_genes_upset_plot.png"
)


# =============================================================================
# 2. Load selected h5ad datasets
# =============================================================================

adata_dict = {}
stats_list = []

print(f"Starting whitelist-based loading for {len(target_files)} selected files...\n")

for filename in target_files:
    
    file_path = os.path.join(data_dir, filename)
    dataset_name = filename.replace(".h5ad", "")
    
    # Check whether the file exists to avoid interruption caused by filename errors
    if not os.path.exists(file_path):
        print(f"File not found and skipped: {filename}")
        continue
    
    # Load the h5ad file
    try:
        adata = sc.read_h5ad(file_path)
        adata_dict[dataset_name] = adata
        
        # Record basic information for each dataset
        stats_list.append({
            "Dataset": dataset_name,
            "Cells": adata.n_obs,
            "Genes": adata.n_vars
        })
        
        print(
            f"Successfully loaded: {dataset_name:<35} | "
            f"Number of cells: {adata.n_obs}"
        )
        
    except Exception as e:
        print(f"Failed to load {filename}: {e}")


# Convert dataset loading statistics into a DataFrame for inspection or export
stats_df = pd.DataFrame(stats_list)


# =============================================================================
# 3. Harmonize sample ID annotation as obs['orig.ident']
# =============================================================================
# Note:
# Different GEO datasets use different column names for sample identifiers.
# To facilitate downstream integration, these columns are copied into obs['orig.ident'].

orig_ident_mapping = {
    "GSE134577_raw_count": "GSM",
    "GSE186802_raw_count_subset": "GSM_ID_1",
    "GSE194078_raw_count": "GSM_ID",
    "GSE243905_raw_count": "GSM_ID",
    "GSE247134_raw_count": "GSM_ID"
}

for dataset_name, source_column in orig_ident_mapping.items():
    
    if dataset_name not in adata_dict:
        print(f"Dataset not loaded; unable to update orig.ident: {dataset_name}")
        continue
    
    if source_column not in adata_dict[dataset_name].obs.columns:
        print(
            f"Column {source_column} not found; "
            f"unable to update orig.ident for {dataset_name}"
        )
        continue
    
    adata_dict[dataset_name].obs["orig.ident"] = (
        adata_dict[dataset_name].obs[source_column].copy()
    )
    
    print(
        f"Updated orig.ident for {dataset_name} "
        f"using obs['{source_column}']"
    )


# =============================================================================
# 4. Filter unexpressed genes and construct expressed gene sets
# =============================================================================
# Note:
# Genes with zero expression across all cells in a given dataset are removed.
# min_cells=3 means that a gene must be expressed in at least three cells to be retained.
# sc.pp.filter_genes modifies the AnnData object in place, consistent with the original code.

print("\nStarting gene filtering and UpSet plot workflow...\n")
print("Checking each dataset and removing genes with zero expression across all cells:")

gene_contents = {}

for dataset_key, adata in adata_dict.items():
    
    # Use the GSE accession as the dataset label in the UpSet plot
    gse_id = dataset_key.split("_")[0]
    
    # Record the number of genes before filtering
    n_genes_before = adata.n_vars
    
    # Remove genes that are not expressed in any cell
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Record the number of genes after filtering
    n_genes_after = adata.n_vars
    dropped_genes = n_genes_before - n_genes_after
    
    # Store expressed gene names as a set to speed up intersection calculation
    gene_contents[gse_id] = set(adata.var_names)
    
    print(
        f"  {gse_id:<12} | "
        f"Removed {dropped_genes:>5} unexpressed genes | "
        f"Remaining expressed genes: {n_genes_after}"
    )


# =============================================================================
# 5. Concatenate datasets using shared expressed genes
# =============================================================================
# Note:
# Before concatenation, this step checks whether each dataset contains obs['orig.ident'].
# If obs['orig.ident'] is missing, it is repaired using obs['GSM_ID'] when available.
# If obs['GSM_ID'] is unavailable, the dataset name is used as a fallback sample label.
#
# ad.concat(..., join='inner') keeps only genes shared by all datasets.
# label='dataset' records the source dataset for each cell.
# index_unique='-' appends dataset-specific suffixes to duplicated cell barcodes.

print("\nChecking datasets for missing obs['orig.ident'] before concatenation...")

for dataset_name, adata in adata_dict.items():
    
    if "orig.ident" in adata.obs.columns:
        continue
    
    print(f"  Missing obs['orig.ident'] detected in: {dataset_name}")
    
    if "GSM_ID" in adata.obs.columns:
        adata.obs["orig.ident"] = adata.obs["GSM_ID"]
        print("    Repaired using obs['GSM_ID']")
    else:
        adata.obs["orig.ident"] = dataset_name
        print("    Repaired using the dataset name")


print("\nConcatenating all datasets using shared expressed genes...")

adata_mega = ad.concat(
    adata_dict,
    join="inner",
    label="dataset",
    index_unique="-"
)

# Ensure cell names are unique after concatenation
adata_mega.obs_names_make_unique()

# Retain only essential metadata columns to reduce object size
if "orig.ident" in adata_mega.obs.columns:
    adata_mega.obs = adata_mega.obs[["orig.ident", "dataset"]].copy()
else:
    print("Warning: obs['orig.ident'] is still missing after concatenation.")

print("\nDataset concatenation completed.")
print(
    f"Merged AnnData shape: {adata_mega.n_obs} cells, "
    f"{adata_mega.n_vars} shared genes"
)

print("\nPreview of merged metadata:")
print(adata_mega.obs.head())


# Save the cleaned merged AnnData object
adata_mega.write(output_mega_path)

print(f"\nMerged AnnData object saved to: {output_mega_path}")


# =============================================================================
# 6. Calculate expressed gene intersections and generate the UpSet plot
# =============================================================================

print("\nCalculating the expressed gene intersection matrix and generating the UpSet plot...")

# Convert the dictionary of expressed gene sets into the format required by upsetplot
upset_data = from_contents(gene_contents)

# Create the figure
fig = plt.figure(figsize=(18, 10))

# Draw the UpSet plot
plot(
    upset_data,
    fig=fig,
    orientation="horizontal",   # Display intersection bars horizontally
    sort_by="cardinality",      # Sort intersections by the number of genes in descending order
    min_subset_size=100,        # Only show intersections containing at least 100 genes
    show_counts=True            # Display the number of genes for each intersection
)

# Add the plot title
plt.suptitle(
    "Expressed Gene Intersections Across Multi-Center Datasets",
    fontsize=16,
    y=0.95
)

# Save the high-resolution figure
plt.savefig(
    output_fig_path,
    bbox_inches="tight",
    dpi=300
)

plt.show()

print(
    f"\nDone. The high-resolution UpSet plot has been saved to: "
    f"{output_fig_path}"
)
