"""Plotting functions extracted from 08_SupplementaryFigure1AB.py; top-level data reads omitted."""
from __future__ import annotations
import os,re
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter, MaxNLocator
from matplotlib.gridspec import GridSpec
from scipy.stats import ranksums,zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text
from typing import Sequence,Iterable,Dict,Tuple
DATASET_COL = 'dataset'
SAMPLE_COL = 'orig.ident'
DISEASE_COL = 'Disease'
CELLTYPE_COL = 'cell_subtype_short'
QC_KEYS = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_ribo']
UMAP_BASIS = 'X_umap'
ARIAL_FONT_PATH = '/home/lishaogang/.fonts/arial.ttf'
ARIAL_FAMILY = 'DejaVu Sans'
FONT_FAMILY = 'DejaVu Sans'
SUPPFIG1B_PDF = None
SUPPFIG1B_PNG = None
SUPPFIG1A_PDF = None
SUPPFIG1A_PNG = None
def validate_obs_columns(adata, columns):
    """Check whether required columns are present in adata.obs."""
    missing = [col for col in columns if col not in adata.obs.columns]
    if missing:
        raise ValueError(f"Missing columns in adata.obs: {missing}")

def rasterize_axis_collections(ax):
    """Rasterize only collection artists while keeping text editable."""
    for collection in ax.collections:
        collection.set_rasterized(True)

def save_current_figure(output_pdf, output_png=None, dpi=300, transparent=True):
    """Save the current matplotlib figure as PDF and optionally PNG."""
    fig = plt.gcf()
    fig.savefig(output_pdf, dpi=dpi, bbox_inches="tight", transparent=transparent)
    if output_png is not None:
        fig.savefig(output_png, dpi=dpi, bbox_inches="tight", transparent=transparent)
    print(f"Saved PDF: {output_pdf}")
    if output_png is not None:
        print(f"Saved PNG: {output_png}")

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
        inner="quartile",
        rotation=45,
        show=False,
    )

    for ax in plt.gcf().axes:
        rasterize_axis_collections(ax)

    save_current_figure(output_pdf, output_png, transparent=True)
    plt.show()
    plt.close()

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
