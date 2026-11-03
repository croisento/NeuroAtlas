# %%
# =============================================================================
# Supplementary Figure 1-2: CSF QC violin plots and dataset UMAP
# =============================================================================
# This notebook-style script generates:
#   1. Supplementary Figure 1A: QC violin plots grouped by dataset.
#   2. Supplementary Figure 1B: UMAP colored by dataset.
#   3. Important summary tables for dataset-level QC and cell/sample counts.
# =============================================================================

import os

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager


# =============================================================================
# 1. Global configuration
# =============================================================================
PROJECT_DIR = os.environ.get("NEU_PROJECT_DIR", "/home/lishaogang/NEU2026_July")
os.chdir(PROJECT_DIR)
INPUT_H5AD = os.environ.get(
    "CSF_INPUT_H5AD", os.path.join(PROJECT_DIR, "adata_final_csf.h5ad")
)
RESULT_ROOT = os.environ.get("FIGURE_RESULT_ROOT", "Result")
OUTDIR = os.path.join(RESULT_ROOT, "Supplement_Figure1", "Supplement_Figure1A_B")
TABLE_OUTDIR = os.path.join(OUTDIR, "tables")

DATASET_COL = "dataset"
SAMPLE_COL = "orig.ident"
DISEASE_COL = "Disease"
CELLTYPE_COL = "cell_subtype_short"
QC_KEYS = [
    "n_genes_by_counts",
    "total_counts",
    "pct_counts_mt",
    "pct_counts_ribo",
]
UMAP_BASIS = "X_umap"

SUPPFIG1A_PDF = os.path.join(OUTDIR, "SupplementaryFigure1A_CSF_QC_violin_by_dataset.pdf")
SUPPFIG1A_PNG = os.path.join(OUTDIR, "SupplementaryFigure1A_CSF_QC_violin_by_dataset.png")
SUPPFIG1B_PDF = os.path.join(OUTDIR, "SupplementaryFigure1B_CSF_UMAP_by_dataset.pdf")
SUPPFIG1B_PNG = os.path.join(OUTDIR, "SupplementaryFigure1B_CSF_UMAP_by_dataset.png")

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(TABLE_OUTDIR, exist_ok=True)

ARIAL_FONT_PATH = "/home/lishaogang/.fonts/arial.ttf"
if os.path.exists(ARIAL_FONT_PATH):
    font_manager.fontManager.addfont(ARIAL_FONT_PATH)
    FONT_FAMILY = font_manager.FontProperties(fname=ARIAL_FONT_PATH).get_name()
else:
    FONT_FAMILY = "Arial"

mpl.rcParams.update({
    "font.family": FONT_FAMILY,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

sc.set_figure_params(figsize=(8, 4), dpi=100, dpi_save=300)


# =============================================================================
# 2. Helper functions
# =============================================================================

def validate_obs_columns(adata, columns):
    """Check whether required columns are present in adata.obs."""
    missing = [col for col in columns if col not in adata.obs.columns]
    if missing:
        raise ValueError(f"Missing columns in adata.obs: {missing}")


def rasterize_axis_collections(ax):
    """Rasterize only collection artists while keeping text editable."""
    for collection in ax.collections:
        collection.set_rasterized(True)


def add_raw_count_metrics(adata):
    """Add total counts and ribosomal percentage from the raw-count layer."""
    if "counts" not in adata.layers:
        raise ValueError("layers['counts'] is required to calculate pct_counts_ribo.")

    gene_names = adata.var_names.astype(str).str.upper()
    ribosomal = gene_names.str.startswith("RPS") | gene_names.str.startswith("RPL")
    counts = adata.layers["counts"]
    total = np.asarray(counts.sum(axis=1)).ravel()
    ribosomal_counts = np.asarray(counts[:, ribosomal].sum(axis=1)).ravel()
    adata.obs["total_counts"] = total
    adata.obs["pct_counts_ribo"] = np.divide(
        ribosomal_counts * 100.0,
        total,
        out=np.zeros_like(ribosomal_counts, dtype=float),
        where=total > 0,
    )


def save_current_figure(output_pdf, output_png=None, dpi=300, transparent=True):
    """Save the current matplotlib figure as PDF and optionally PNG."""
    fig = plt.gcf()
    fig.savefig(output_pdf, dpi=dpi, bbox_inches="tight", transparent=transparent)
    if output_png is not None:
        fig.savefig(output_png, dpi=dpi, bbox_inches="tight", transparent=transparent)
    print(f"Saved PDF: {output_pdf}")
    if output_png is not None:
        print(f"Saved PNG: {output_png}")


def save_dataset_summary_tables(
    adata,
    table_outdir=TABLE_OUTDIR,
    dataset_col=DATASET_COL,
    sample_col=SAMPLE_COL,
    disease_col=DISEASE_COL,
    celltype_col=CELLTYPE_COL,
    qc_keys=QC_KEYS,
):
    """Save important dataset-level summary tables used for QC inspection."""
    required_cols = [dataset_col, sample_col, disease_col, celltype_col] + qc_keys
    validate_obs_columns(adata, required_cols)

    obs_df = adata.obs[required_cols].copy()

    for col in [dataset_col, sample_col, disease_col, celltype_col]:
        obs_df[col] = obs_df[col].astype(str).str.strip()

    for key in qc_keys:
        obs_df[key] = pd.to_numeric(obs_df[key], errors="coerce")

    dataset_summary = (
        obs_df
        .groupby(dataset_col, observed=True)
        .agg(
            n_cells=(dataset_col, "size"),
            n_samples=(sample_col, "nunique"),
            n_diseases=(disease_col, "nunique"),
            diseases=(disease_col, lambda x: ";".join(sorted(pd.unique(x.astype(str))))),
            median_n_genes_by_counts=("n_genes_by_counts", "median"),
            mean_n_genes_by_counts=("n_genes_by_counts", "mean"),
            median_pct_counts_mt=("pct_counts_mt", "median"),
            mean_pct_counts_mt=("pct_counts_mt", "mean"),
            median_total_counts=("total_counts", "median"),
            mean_total_counts=("total_counts", "mean"),
            median_pct_counts_ribo=("pct_counts_ribo", "median"),
            mean_pct_counts_ribo=("pct_counts_ribo", "mean"),
        )
        .reset_index()
        .sort_values(dataset_col)
    )

    sample_summary = (
        obs_df
        .groupby([dataset_col, sample_col, disease_col], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
        .sort_values([dataset_col, disease_col, sample_col])
    )

    celltype_counts_by_dataset = (
        obs_df
        .groupby([dataset_col, celltype_col], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
        .sort_values([dataset_col, celltype_col])
    )

    output_paths = {
        "dataset_qc_summary": os.path.join(table_outdir, "CSF_dataset_QC_summary.csv"),
        "sample_cell_counts": os.path.join(table_outdir, "CSF_sample_cell_counts_by_dataset.csv"),
        "celltype_counts_by_dataset": os.path.join(table_outdir, "CSF_celltype_counts_by_dataset.csv"),
    }

    dataset_summary.to_csv(output_paths["dataset_qc_summary"], index=False)
    sample_summary.to_csv(output_paths["sample_cell_counts"], index=False)
    celltype_counts_by_dataset.to_csv(output_paths["celltype_counts_by_dataset"], index=False)

    for label, path in output_paths.items():
        print(f"Saved {label}: {path}")

    return dataset_summary, sample_summary, celltype_counts_by_dataset


# =============================================================================
# 3. Supplementary Figure 1A: QC violin plots by dataset
# =============================================================================

def plot_supplementary_figure_1a_qc_violin(
    adata,
    keys=QC_KEYS,
    groupby=DATASET_COL,
    output_pdf=SUPPFIG1A_PDF,
    output_png=SUPPFIG1A_PNG,
):
    """Plot QC violin plots grouped by dataset."""
    validate_obs_columns(adata, [groupby] + keys)

    sc.pl.violin(
        adata,
        keys=keys,
        groupby=groupby,
        jitter=False,
        stripplot=False,
        multi_panel=True,
        rotation=45,
        show=False,
    )

    for ax in plt.gcf().axes:
        rasterize_axis_collections(ax)

    save_current_figure(output_pdf, output_png, transparent=True)
    plt.show()
    plt.close()


# =============================================================================
# 4. Supplementary Figure 1B: UMAP colored by dataset
# =============================================================================

def plot_supplementary_figure_1b_dataset_umap(
    adata,
    color=DATASET_COL,
    basis=UMAP_BASIS,
    output_pdf=SUPPFIG1B_PDF,
    output_png=SUPPFIG1B_PNG,
):
    """Plot UMAP colored by dataset, rasterizing only the points."""
    validate_obs_columns(adata, [color])

    if basis not in adata.obsm:
        raise ValueError(f"{basis} is not available in adata.obsm.")

    fig, ax = plt.subplots(figsize=(6, 6))

    sc.pl.embedding(
        adata,
        basis=basis,
        color=color,
        size=3,
        alpha=0.75,
        ax=ax,
        show=False,
        frameon=False,
        legend_loc="center left",
        title=None,
        legend_fontsize=7,
    )

    rasterize_axis_collections(ax)

    legend = ax.get_legend()
    if legend is not None:
        legend.set_bbox_to_anchor((1.03, 1))
        legend.set_loc("upper left")
        legend.set_frame_on(False)

    plt.tight_layout()
    save_current_figure(output_pdf, output_png, transparent=True)
    plt.show()
    plt.close()


# %%
# =============================================================================
# 5. Load data
# =============================================================================

adata = sc.read_h5ad(INPUT_H5AD)
add_raw_count_metrics(adata)
print(adata)


# %%
# =============================================================================
# 6. Save important summary tables
# =============================================================================

dataset_qc_summary, sample_cell_counts, celltype_counts_by_dataset = save_dataset_summary_tables(
    adata
)


# %%
# =============================================================================
# 7. Generate Supplementary Figure 1A
# =============================================================================

plot_supplementary_figure_1a_qc_violin(adata)


# %%
# =============================================================================
# 8. Generate Supplementary Figure 1B
# =============================================================================

plot_supplementary_figure_1b_dataset_umap(adata)
