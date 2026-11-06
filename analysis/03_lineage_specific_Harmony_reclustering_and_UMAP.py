import os
import scanpy as sc
os.chdir("/home/lishaogang/NEU2026_July")
adata = sc.read_h5ad('/home/lishaogang/NEU2026_July/adata_final.h5ad')

# =============================================================================
# Lineage-specific reclustering and UMAP plotting based on logcounts + Harmony
# =============================================================================

import scanpy as sc
import scanpy.external as sce
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
from adjustText import adjust_text


# =============================================================================
# 1. Global parameters
# =============================================================================

CELLTYPE_COL = "cell_subtype_short"
BATCH_KEY = "dataset"
LOG_LAYER = "logcounts"

HARMONY_BASIS = "X_pca_harmony_logcounts"
UMAP_BASIS = "X_umap_logcounts_harmony"
PCA_BASIS = "X_pca_logcounts"
LEIDEN_KEY = "leiden_logcounts_harmony"

N_TOP_GENES = 3000
N_PCS = 50
N_NEIGHBORS = 30
N_PCS_NEIGHBORS = 30
LEIDEN_RESOLUTION = 0.6

ARIAL_FAMILY = "Arial"

mpl.rcParams["font.family"] = ARIAL_FAMILY
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"


# =============================================================================
# 2. Color map
# =============================================================================

SOFT_COLOR_MAP = {
    # B lineage
    "Naive B": "#C96F6F",
    "Transitional B": "#D89A84",
    "Memory B": "#B85C68",
    "Plasma cell": "#B77AB5",

    # CD4 T cells
    "CD4 Naive": "#7DA7CF",
    "CD4 Memory": "#AEC9E4",
    "CD4 CTL": "#3E7C9D",

    # CD8 T / MAIT
    "CD8 Naive": "#4F9A78",
    "MAIT": "#75B38A",
    "CD8 Memory": "#98C970",
    "CD8 CTL": "#C3D894",

    # Unconventional T cells
    "γδT": "#A6B85A",
    "NKT": "#BFB552",

    # Treg
    "Treg": "#3F6BA7",

    # NK / ILC
    "Transit. NK": "#A995C1",
    "NK dim": "#C99235",
    "NK bright": "#D7BE66",
    "ILC2": "#DDD36A",

    # Myeloid cells
    "cMonocytes": "#c84242",
    "ncMonocytes": "#d6cd9c",
    "mdMac": "#b97cb9",
    "trMac": "#d16565",
    "infMac": "#7b5bb8",
    "cDC2": "#967262",
    "pDC": "#7096c7",
    "ASDC": "#3675b9",

    # Additional cell types
    "Basophil": "#91BD9E",
    "Neutrophils": "#B7AC5A"
}


# =============================================================================
# 3. Create lineage-specific subsets
# =============================================================================

def create_lineage_subset(adata_obj, lineage_name):
    """Create a lineage-specific AnnData subset from the full object."""
    if "cell_type" not in adata_obj.obs.columns:
        raise ValueError("obs does not contain 'cell_type'.")

    if lineage_name == "Myeloid":
        return adata_obj[
            adata_obj.obs["cell_type"].isin(["Myeloid cell", "pDC"])
        ].copy()

    if lineage_name == "B_cell":
        return adata_obj[
            adata_obj.obs["cell_type"] == "B cell"
        ].copy()

    if lineage_name == "ILCs":
        return adata_obj[
            adata_obj.obs["cell_type"] == "Innate lymphoid cell"
        ].copy()

    if lineage_name == "T_cell":
        return adata_obj[
            adata_obj.obs["cell_type"] == "T cell"
        ].copy()

    raise ValueError(f"Unknown lineage name: {lineage_name}")


try:
    adata_Myeloid
except NameError:
    adata_Myeloid = create_lineage_subset(adata, "Myeloid")

try:
    adata_B
except NameError:
    adata_B = create_lineage_subset(adata, "B_cell")

try:
    adata_ILCs
except NameError:
    adata_ILCs = create_lineage_subset(adata, "ILCs")

try:
    adata_T
except NameError:
    adata_T = create_lineage_subset(adata, "T_cell")


# =============================================================================
# 4. Align subset objects to the parent AnnData object
# =============================================================================
# This step ensures that each subset:
#   1. Uses the same genes as the parent AnnData object.
#   2. Receives layers['counts'] and layers['logcounts'] from the parent object.
#   3. Keeps subset-specific metadata when available.
#
# This is important when an existing subset object has a different gene space.

def align_subset_to_parent(parent_adata, subset_adata, subset_name):
    """Align a subset object to the parent AnnData object by cell names."""
    print("\n" + "=" * 80)
    print(f"Aligning subset to parent object: {subset_name}")
    print("=" * 80)

    if not parent_adata.obs_names.is_unique:
        raise ValueError("parent_adata.obs_names are not unique.")

    if not parent_adata.var_names.is_unique:
        raise ValueError("parent_adata.var_names are not unique.")

    if not subset_adata.obs_names.is_unique:
        raise ValueError(f"{subset_name}.obs_names are not unique.")

    missing_cells = subset_adata.obs_names.difference(parent_adata.obs_names)

    if len(missing_cells) > 0:
        raise ValueError(
            f"{subset_name}: {len(missing_cells)} cells are not found in parent adata. "
            f"Examples: {list(missing_cells[:5])}"
        )

    subset_fixed = parent_adata[subset_adata.obs_names, :].copy()

    for col in subset_adata.obs.columns:
        subset_fixed.obs[col] = subset_adata.obs.loc[subset_fixed.obs_names, col].copy()

    if LOG_LAYER not in subset_fixed.layers:
        if "cp10k_log1p" in subset_fixed.layers:
            subset_fixed.layers[LOG_LAYER] = subset_fixed.layers["cp10k_log1p"].copy()
        else:
            raise ValueError(
                f"{subset_name}: neither layers['{LOG_LAYER}'] nor "
                f"layers['cp10k_log1p'] was found."
            )

    if "counts" not in subset_fixed.layers:
        if "cp10k_counts" in subset_fixed.layers:
            subset_fixed.layers["counts"] = subset_fixed.layers["cp10k_counts"].copy()
        else:
            print(f"Warning: {subset_name} does not contain layers['counts'].")

    print(subset_fixed)
    print("Available layers:", list(subset_fixed.layers.keys()))

    return subset_fixed


adata_Myeloid = align_subset_to_parent(adata, adata_Myeloid, "adata_Myeloid")
adata_B = align_subset_to_parent(adata, adata_B, "adata_B")
adata_ILCs = align_subset_to_parent(adata, adata_ILCs, "adata_ILCs")
adata_T = align_subset_to_parent(adata, adata_T, "adata_T")


# =============================================================================
# 5. Recluster one lineage based on logcounts + Harmony
# =============================================================================

def clean_invalid_values(adata_obj):
    """Replace NaN and infinite values with zero."""
    if sp.issparse(adata_obj.X):
        adata_obj.X.data = np.nan_to_num(
            adata_obj.X.data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )
    else:
        adata_obj.X = np.nan_to_num(
            adata_obj.X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )


def recluster_lineage_logcounts_harmony(
    adata_obj,
    lineage_name,
    batch_key=BATCH_KEY,
    log_layer=LOG_LAYER,
    n_top_genes=N_TOP_GENES,
    n_pcs=N_PCS,
    n_neighbors=N_NEIGHBORS,
    n_pcs_neighbors=N_PCS_NEIGHBORS,
    leiden_resolution=LEIDEN_RESOLUTION
):
    """Run HVG selection, PCA, Harmony, neighbors, Leiden clustering, and UMAP."""
    print("\n" + "=" * 80)
    print(f"Reclustering lineage: {lineage_name}")
    print("=" * 80)

    adata_log = adata_obj.copy()

    if log_layer not in adata_log.layers:
        raise ValueError(f"{lineage_name}: layers['{log_layer}'] was not found.")

    adata_log.X = adata_log.layers[log_layer].copy()

    if batch_key not in adata_log.obs.columns:
        raise ValueError(
            f"{lineage_name}: obs does not contain '{batch_key}', "
            "which is required for Harmony integration."
        )

    adata_log.obs[batch_key] = (
        adata_log.obs[batch_key]
        .astype(str)
        .astype("category")
    )

    print("Dataset distribution:")
    print(adata_log.obs[batch_key].value_counts())

    clean_invalid_values(adata_log)

    sc.pp.highly_variable_genes(
        adata_log,
        n_top_genes=n_top_genes,
        flavor="seurat"
    )

    hvg_number = int(adata_log.var["highly_variable"].sum())
    print(f"HVG number: {hvg_number}")

    if hvg_number == 0:
        raise ValueError(f"{lineage_name}: no highly variable genes were detected.")

    adata_hvg = adata_log[:, adata_log.var["highly_variable"]].copy()

    sc.pp.scale(
        adata_hvg,
        max_value=10
    )

    sc.tl.pca(
        adata_hvg,
        n_comps=n_pcs,
        svd_solver="arpack"
    )

    sce.pp.harmony_integrate(
        adata_hvg,
        key=batch_key,
        basis="X_pca",
        adjusted_basis=HARMONY_BASIS,
        max_iter_harmony=30
    )

    print("Harmony finished.")
    print("obsm keys:", adata_hvg.obsm.keys())

    sc.pp.neighbors(
        adata_hvg,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs_neighbors,
        use_rep=HARMONY_BASIS,
        metric="cosine"
    )

    sc.tl.leiden(
        adata_hvg,
        resolution=leiden_resolution,
        key_added=LEIDEN_KEY
    )

    print("Cluster distribution:")
    print(
        adata_hvg.obs[LEIDEN_KEY]
        .value_counts()
        .sort_index()
    )

    sc.tl.umap(
        adata_hvg,
        min_dist=0.35,
        spread=1.0
    )

    # Write reclustering results back to the full-gene lineage object.
    adata_obj.obs[LEIDEN_KEY] = adata_hvg.obs[LEIDEN_KEY].copy()
    adata_obj.obsm[UMAP_BASIS] = adata_hvg.obsm["X_umap"].copy()
    adata_obj.obsm[PCA_BASIS] = adata_hvg.obsm["X_pca"].copy()
    adata_obj.obsm[HARMONY_BASIS] = adata_hvg.obsm[HARMONY_BASIS].copy()

    adata_obj.uns[LEIDEN_KEY] = adata_hvg.uns[LEIDEN_KEY].copy()
    adata_obj.uns["umap_logcounts_harmony"] = adata_hvg.uns["umap"].copy()

    print(f"Reclustering results written back to {lineage_name}.")
    print(f"obs:  {LEIDEN_KEY}")
    print(f"obsm: {UMAP_BASIS}")
    print(f"obsm: {PCA_BASIS}")
    print(f"obsm: {HARMONY_BASIS}")

    return adata_obj, adata_hvg


# =============================================================================
# 6. Run lineage-specific reclustering
# =============================================================================

LINEAGE_RECLUSTER_CONFIG = {
    "Myeloid": {
        "adata": adata_Myeloid,
        "output_object": "adata_Myeloid_logcounts_Harmony_dataset_reclustered.h5ad"
    },
    "B_cells": {
        "adata": adata_B,
        "output_object": "adata_B_logcounts_Harmony_dataset_reclustered.h5ad"
    },
    "ILCs": {
        "adata": adata_ILCs,
        "output_object": "adata_ILCs_logcounts_Harmony_dataset_reclustered.h5ad"
    },
    "T_cells": {
        "adata": adata_T,
        "output_object": "adata_T_logcounts_Harmony_dataset_reclustered.h5ad"
    }
}

lineage_hvg_objects = {}

for lineage_name, config in LINEAGE_RECLUSTER_CONFIG.items():
    lineage_obj, lineage_hvg = recluster_lineage_logcounts_harmony(
        adata_obj=config["adata"],
        lineage_name=lineage_name
    )

    config["adata"] = lineage_obj
    lineage_hvg_objects[lineage_name] = lineage_hvg

    #lineage_obj.write_h5ad(config["output_object"])
    print(f"Saved reclustered lineage object: {config['output_object']}")


adata_Myeloid = LINEAGE_RECLUSTER_CONFIG["Myeloid"]["adata"]
adata_B = LINEAGE_RECLUSTER_CONFIG["B_cells"]["adata"]
adata_ILCs = LINEAGE_RECLUSTER_CONFIG["ILCs"]["adata"]
adata_T = LINEAGE_RECLUSTER_CONFIG["T_cells"]["adata"]


# =============================================================================
# 7. Helper functions for UMAP plotting
# =============================================================================

def get_group_center(adata_obj, groupby, basis):
    """Calculate median UMAP coordinates for each group."""
    coord_df = pd.DataFrame(
        adata_obj.obsm[basis],
        columns=["UMAP1", "UMAP2"],
        index=adata_obj.obs_names
    )

    coord_df[groupby] = adata_obj.obs[groupby].astype(str).values

    center_dict = {}

    for group, sub_df in coord_df.groupby(groupby, observed=True):
        center_dict[group] = sub_df[["UMAP1", "UMAP2"]].median(axis=0).values

    return center_dict


def rasterize_scatter_points(ax):
    """Rasterize scatter points while keeping text editable."""
    for collection in ax.collections:
        collection.set_rasterized(True)


def add_umap_axis_arrows(
    ax,
    arrow_x=0.05,
    arrow_y=0.05,
    arrow_len=0.16,
    lw=2,
    label_fontsize=12,
    fontfamily=ARIAL_FAMILY
):
    """Add UMAP1 and UMAP2 arrows to the lower-left corner."""
    ax.annotate(
        "",
        xy=(arrow_x + arrow_len, arrow_y),
        xytext=(arrow_x, arrow_y),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            lw=lw,
            color="black"
        )
    )

    ax.annotate(
        "",
        xy=(arrow_x, arrow_y + arrow_len),
        xytext=(arrow_x, arrow_y),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            lw=lw,
            color="black"
        )
    )

    ax.text(
        arrow_x + arrow_len / 2,
        arrow_y - 0.04,
        "UMAP1",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=label_fontsize,
        fontfamily=fontfamily
    )

    ax.text(
        arrow_x - 0.04,
        arrow_y + arrow_len / 2,
        "UMAP2",
        transform=ax.transAxes,
        ha="right",
        va="center",
        rotation=90,
        fontsize=label_fontsize,
        fontfamily=fontfamily
    )


def remove_axis_traces(ax):
    """Remove axis ticks, labels, titles, and spines."""
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("")

    for spine in ax.spines.values():
        spine.set_visible(False)


# =============================================================================
# 8. Plot one lineage-specific UMAP using the recalculated Harmony UMAP only
# =============================================================================

def plot_reclustered_lineage_umap(
    adata_obj,
    lineage_name,
    celltype_order,
    output_file,
    celltype_col=CELLTYPE_COL,
    basis=UMAP_BASIS,
    exclude_celltypes=None,
    color_map=None,
    point_size=8,
    figsize=(8, 8),
    label_fontsize=11
):
    """Plot one lineage-specific UMAP using only the recalculated Harmony UMAP."""
    if exclude_celltypes is None:
        exclude_celltypes = []

    if color_map is None:
        color_map = {}

    adata_plot = adata_obj.copy()

    if basis not in adata_plot.obsm:
        raise ValueError(
            f"{lineage_name}: '{basis}' was not found in .obsm. "
            "Please run lineage-specific logcounts + Harmony reclustering first."
        )

    if celltype_col not in adata_plot.obs.columns:
        raise ValueError(
            f"{lineage_name}: obs does not contain '{celltype_col}'."
        )

    celltype_series = adata_plot.obs[celltype_col]

    if celltype_series.dtype.name == "category":
        new_categories = {
            category: "mdMac" if category == "maMac" else category
            for category in celltype_series.cat.categories
        }
        adata_plot.obs[celltype_col] = celltype_series.cat.rename_categories(new_categories)
    else:
        adata_plot.obs[celltype_col] = celltype_series.replace("maMac", "mdMac")

    if exclude_celltypes:
        keep_mask = ~adata_plot.obs[celltype_col].astype(str).isin(exclude_celltypes)
        adata_plot = adata_plot[keep_mask, :].copy()

    observed_celltypes = adata_plot.obs[celltype_col].astype(str).unique().tolist()

    final_order = [
        cell_type for cell_type in celltype_order
        if cell_type in observed_celltypes
    ]

    if len(final_order) == 0:
        raise ValueError(f"{lineage_name}: none of the requested cell types were found.")

    adata_plot = adata_plot[
        adata_plot.obs[celltype_col].astype(str).isin(final_order),
        :
    ].copy()

    adata_plot.obs[celltype_col] = pd.Categorical(
        adata_plot.obs[celltype_col].astype(str),
        categories=final_order,
        ordered=True
    )

    missing_colors = [
        cell_type for cell_type in final_order
        if cell_type not in color_map
    ]

    if missing_colors:
        print(f"Warning: missing colors for {lineage_name}: {missing_colors}")

    adata_plot.uns[f"{celltype_col}_colors"] = [
        color_map.get(cell_type, "#BDBDBD")
        for cell_type in final_order
    ]

    label_center = get_group_center(
        adata_plot,
        groupby=celltype_col,
        basis=basis
    )

    fig, ax = plt.subplots(figsize=figsize)

    sc.pl.embedding(
        adata_plot,
        basis=basis,
        color=celltype_col,
        size=point_size,
        alpha=1,
        ax=ax,
        show=False,
        frameon=False,
        legend_loc=None,
        title=None
    )

    rasterize_scatter_points(ax)

    text_list = []

    for cell_type in final_order:
        if cell_type not in label_center:
            continue

        x_pos, y_pos = label_center[cell_type]

        text = ax.text(
            x_pos,
            y_pos,
            cell_type,
            fontsize=label_fontsize,
            fontfamily=ARIAL_FAMILY,
            weight="bold",
            ha="center",
            va="center",
            color="black",
            path_effects=[
                pe.withStroke(
                    linewidth=3,
                    foreground="white"
                )
            ]
        )

        text_list.append(text)

    adjust_text(
        text_list,
        ax=ax,
        expand_text=(1.1, 1.2),
        force_text=(0.3, 0.4),
        arrowprops=dict(
            arrowstyle="-",
            lw=0.7,
            color="#333333",
            alpha=0.5
        )
    )

    add_umap_axis_arrows(
        ax=ax,
        arrow_x=0.05,
        arrow_y=0.05,
        arrow_len=0.16,
        lw=2,
        label_fontsize=12,
        fontfamily=ARIAL_FAMILY
    )

    remove_axis_traces(ax)

    fig.savefig(
        output_file,
        bbox_inches="tight",
        transparent=True,
        dpi=300
    )

    plt.show()

    print(f"Saved {lineage_name} UMAP to: {output_file}")

    return adata_plot


# =============================================================================
# 9. Plot all lineage-specific UMAPs
# =============================================================================

LINEAGE_PLOT_CONFIG = {
    "Myeloid": {
        "adata": adata_Myeloid,
        "celltype_order": [
            "cMonocytes",
            "ncMonocytes",
            "trMac",
            "infMac",
            "mdMac",
            "cDC2",
            "pDC",
            "ASDC"
        ],
        "exclude_celltypes": ["Basophil", "Neutrophils"],
        "output": "Myeloid_UMAP_logcounts_Harmony_FigureC.pdf"
    },

    "B_cells": {
        "adata": adata_B,
        "celltype_order": [
            "Naive B",
            "Transitional B",
            "Memory B",
            "Plasma cell"
        ],
        "exclude_celltypes": [],
        "output": "B_cells_UMAP_logcounts_Harmony.pdf"
    },

    "ILCs": {
        "adata": adata_ILCs,
        "celltype_order": [
            "Transit. NK",
            "NK dim",
            "NK bright",
            "ILC2"
        ],
        "exclude_celltypes": [],
        "output": "ILCs_UMAP_logcounts_Harmony.pdf"
    },

    "T_cells": {
        "adata": adata_T,
        "celltype_order": [
            "CD4 Naive",
            "CD4 Memory",
            "CD4 CTL",
            "CD8 Naive",
            "CD8 Memory",
            "CD8 CTL",
            "MAIT",
            "γδT",
            "NKT",
            "Treg"
        ],
        "exclude_celltypes": [],
        "output": "T_cells_UMAP_logcounts_Harmony.pdf"
    }
}

lineage_umap_results = {}

for lineage_name, config in LINEAGE_PLOT_CONFIG.items():
    print("\n" + "=" * 80)
    print(f"Plotting reclustered lineage UMAP: {lineage_name}")
    print("=" * 80)

    lineage_umap_results[lineage_name] = plot_reclustered_lineage_umap(
        adata_obj=config["adata"],
        lineage_name=lineage_name,
        celltype_order=config["celltype_order"],
        output_file=config["output"],
        celltype_col=CELLTYPE_COL,
        basis=UMAP_BASIS,
        exclude_celltypes=config["exclude_celltypes"],
        color_map=SOFT_COLOR_MAP,
        point_size=8,
        figsize=(8, 8),
        label_fontsize=11
    )

adata_ILCs.write_h5ad("/home/lishaogang/NEU2026_July/adata_final_ILCS.h5ad")
adata_B.write_h5ad("/home/lishaogang/NEU2026_July/adata_final_B.h5ad")
adata_T.write_h5ad("/home/lishaogang/NEU2026_July/adata_final_T.h5ad")
adata_Myeloid.write_h5ad("/home/lishaogang/NEU2026_July/adata_final_Myeloid.h5ad")
