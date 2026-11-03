# =============================================================================
# Figure 2 plotting script
# =============================================================================
# This script generates all Figure 2 panels from the integrated AnnData object.
# It preserves the original plotting parameters, color palette, output filenames,
# and figure appearance while removing duplicated code blocks.
# =============================================================================

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from adjustText import adjust_text


# =============================================================================
# 1. Global paths and plotting settings
# =============================================================================
PROJECT_DIR = Path(os.environ.get("NEU_PROJECT_DIR", "/home/lishaogang/NEU2026_July"))
INPUT_H5AD = Path(os.environ.get("FIGURE_INPUT_H5AD", PROJECT_DIR / "adata_final.h5ad"))
OUTPUT_DIR = Path(os.environ.get("FIGURE2_OUTPUT_DIR", PROJECT_DIR))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(OUTPUT_DIR)
ARIAL_FONT_PATH = Path("/home/lishaogang/.fonts/arial.ttf")

CELLTYPE_COL = "cell_subtype_short"
BASIS = "X_umap"
EXPRESSION_LAYER = "logcounts"

LINEAGE_INPUT_PATHS = {
    "Myeloid": os.environ.get("FIGURE2_MYELOID_H5AD"),
    "B_cells": os.environ.get("FIGURE2_B_H5AD"),
    "ILCs": os.environ.get("FIGURE2_ILCS_H5AD"),
    "T_cells": os.environ.get("FIGURE2_T_H5AD"),
}


def setup_arial_font(font_path=ARIAL_FONT_PATH):
    """Register Arial and configure editable vector-font settings."""
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_family = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        font_family = "Arial"
        print(f"Warning: Arial font file not found at {font_path}. Falling back to 'Arial'.")

    mpl.rcParams.update({
        "font.family": font_family,
        "font.sans-serif": [font_family],
        "font.weight": "regular",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.fontset": "custom",
        "mathtext.rm": font_family,
        "mathtext.bf": font_family,
        "mathtext.it": font_family,
        "mathtext.cal": font_family,
        "mathtext.sf": font_family,
        "mathtext.tt": font_family,
        "axes.titlesize": 16,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    print(f"Detected font family: {font_family}")
    return font_family


ARIAL_FAMILY = setup_arial_font()


# =============================================================================
# 2. Shared cell-type colors and ordering
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

    # CD8 T cells and MAIT cells
    "CD8 Naive": "#4F9A78",
    "MAIT": "#75B38A",
    "CD8 Memory": "#98C970",
    "CD8 CTL": "#C3D894",

    # Unconventional T cells
    "γδT": "#A6B85A",
    "NKT": "#BFB552",

    # Treg
    "Treg": "#3F6BA7",

    # NK / ILC lineage
    "Transit. NK": "#A995C1",
    "NK dim": "#C99235",
    "NK bright": "#D7BE66",
    "ILC2": "#DDD36A",

    # Monocytes and macrophages
    "cMonocytes": "#C97891",
    "ncMonocytes": "#9E5F73",
    "trMac": "#D6A196",
    "infMac": "#C17DB4",
    "mdMac": "#9568B1",

    # DCs and granulocytes
    "cDC2": "#B46AB0",
    "pDC": "#7F8CC1",
    "ASDC": "#B79BD0",
    "Basophil": "#91BD9E",
    "Neutrophils": "#B7AC5A",
}

CELL_TYPE_ORDER = [
    "Naive B", "Memory B", "Plasma cell",
    "CD4 Naive", "CD8 Naive", "CD4 Memory", "CD8 Memory",
    "Treg", "CD4 CTL", "CD8 CTL", "MAIT", "γδT", "NKT",
    "NK bright", "Transit. NK", "NK dim", "ILC2",
    "cMonocytes", "ncMonocytes", "mdMac", "trMac", "infMac",
    "cDC2", "pDC", "ASDC",
]

LABEL_NUDGE = {
    # NK / ILC lineage
    "NK bright": (0.15, 0.20),
    "NK dim": (-0.15, -0.12),
    "Transit. NK": (-0.25, 0.00),
    "ILC2": (0.18, -0.05),

    # Myeloid lineage
    "cMonocytes": (0.25, 0.00),
    "ncMonocytes": (0.10, 0.12),
    "trMac": (0.10, 0.05),
    "infMac": (0.12, -0.05),
    "mdMac": (0.00, -0.12),

    # DCs
    "ASDC": (-0.10, -0.05),
    "pDC": (0.00, -0.10),
    "cDC2": (0.08, 0.00),

    # B lineage
    "Naive B": (-0.10, -0.10),
    "Transitional B": (-0.10, 0.08),
    "Memory B": (0.12, 0.00),
    "Plasma cell": (0.10, 0.08),

    # T lineage
    "CD4 Naive": (-0.15, -0.03),
    "CD4 Memory": (-0.10, 0.00),
    "CD8 Naive": (0.08, -0.08),
    "CD8 Memory": (0.08, 0.06),
    "MAIT": (0.05, -0.08),
    "γδT": (0.08, 0.05),
    "NKT": (0.08, 0.05),
}


# =============================================================================
# 3. Shared helper functions
# =============================================================================

def make_ordered_categories(adata_obj, celltype_col, cell_type_order, color_map):
    """Set a stable cell-type order and apply the corresponding color list."""
    adata_obj.obs[celltype_col] = adata_obj.obs[celltype_col].astype(str)

    present_order = [
        cell_type for cell_type in cell_type_order
        if cell_type in adata_obj.obs[celltype_col].unique()
    ]

    extra_types = [
        cell_type for cell_type in sorted(adata_obj.obs[celltype_col].unique())
        if cell_type not in present_order
    ]

    final_order = present_order + extra_types

    adata_obj.obs[celltype_col] = pd.Categorical(
        adata_obj.obs[celltype_col],
        categories=final_order,
        ordered=True,
    )

    missing_colors = [cell_type for cell_type in final_order if cell_type not in color_map]
    if missing_colors:
        print("Warning: missing colors for:", missing_colors)

    adata_obj.uns[f"{celltype_col}_colors"] = [
        color_map.get(cell_type, "#BDBDBD") for cell_type in final_order
    ]

    return final_order


def get_group_label_positions(adata_obj, groupby, basis="X_umap"):
    """Return medoid-like label positions to reduce labels falling into empty space."""
    coords = pd.DataFrame(
        adata_obj.obsm[basis],
        columns=["UMAP1", "UMAP2"],
        index=adata_obj.obs_names,
    )
    coords[groupby] = adata_obj.obs[groupby].astype(str).values

    label_positions = {}

    for group, sub_df in coords.groupby(groupby, observed=True):
        xy = sub_df[["UMAP1", "UMAP2"]].values
        median_xy = np.median(xy, axis=0)
        dist = np.sum((xy - median_xy) ** 2, axis=1)
        label_positions[group] = xy[np.argmin(dist)]

    return label_positions


def get_group_center(adata_obj, groupby, basis):
    """Return median UMAP coordinates for each group."""
    coord_df = pd.DataFrame(
        adata_obj.obsm[basis],
        columns=["UMAP1", "UMAP2"],
        index=adata_obj.obs_names,
    )
    coord_df[groupby] = adata_obj.obs[groupby].astype(str).values

    center_dict = {}
    for group, sub_df in coord_df.groupby(groupby, observed=True):
        center_dict[group] = sub_df[["UMAP1", "UMAP2"]].median(axis=0).values

    return center_dict


def rasterize_scatter_points(ax):
    """Rasterize only the scatter layer while keeping text editable."""
    for collection in ax.collections:
        collection.set_rasterized(True)


def add_umap_axis_arrows(
    ax,
    arrow_x=0.06,
    arrow_y=0.06,
    arrow_len=0.15,
    lw=2,
    label_fontsize=10,
    fontfamily=ARIAL_FAMILY,
):
    """Add custom UMAP1/UMAP2 arrows in axes-fraction coordinates."""
    ax.annotate(
        "",
        xy=(arrow_x + arrow_len, arrow_y),
        xytext=(arrow_x, arrow_y),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=lw, color="black"),
    )

    ax.annotate(
        "",
        xy=(arrow_x, arrow_y + arrow_len),
        xytext=(arrow_x, arrow_y),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=lw, color="black"),
    )

    ax.text(
        arrow_x + arrow_len / 2,
        arrow_y - 0.025,
        "UMAP1",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=label_fontsize,
        fontfamily=fontfamily,
    )

    ax.text(
        arrow_x - 0.025,
        arrow_y + arrow_len / 2,
        "UMAP2",
        transform=ax.transAxes,
        ha="right",
        va="center",
        rotation=90,
        fontsize=label_fontsize,
        fontfamily=fontfamily,
    )


def remove_axis_traces(ax):
    """Remove default axes, ticks, and spines."""
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


def add_celltype_labels(
    ax,
    label_positions,
    final_order,
    label_nudge=None,
    fontsize=10,
    fontfamily=ARIAL_FAMILY,
    fontweight="normal",
    stroke_width=3.2,
    adjust_expand=(1.12, 1.22),
    adjust_force=(0.25, 0.45),
    arrow_lw=0.75,
    arrow_alpha=0.45,
):
    """Add cell-type labels with a white stroke and adjustText collision avoidance."""
    if label_nudge is None:
        label_nudge = {}

    texts = []

    for group in final_order:
        if group not in label_positions:
            continue

        x, y = label_positions[group]
        dx, dy = label_nudge.get(group, (0, 0))

        text = ax.text(
            x + dx,
            y + dy,
            group,
            fontsize=fontsize,
            fontfamily=fontfamily,
            fontweight=fontweight,
            ha="center",
            va="center",
            color="black",
            path_effects=[pe.withStroke(linewidth=stroke_width, foreground="white")],
        )
        texts.append(text)

    adjust_text(
        texts,
        ax=ax,
        expand_text=adjust_expand,
        force_text=adjust_force,
        arrowprops=dict(
            arrowstyle="-",
            color="#444444",
            lw=arrow_lw,
            alpha=arrow_alpha,
        ),
    )

    return texts


# =============================================================================
# 4. Load integrated AnnData object
# =============================================================================

adata = sc.read_h5ad(INPUT_H5AD)

# Use the integrated object as the default UMAP object for sample-count and tissue plots.
adata_umap = adata


# =============================================================================
# 5. Figure 2A: global UMAP colored by cell subtype
# =============================================================================

adata_fig2a = adata.copy()
final_order_fig2a = make_ordered_categories(
    adata_fig2a,
    celltype_col=CELLTYPE_COL,
    cell_type_order=CELL_TYPE_ORDER,
    color_map=SOFT_COLOR_MAP,
)

label_positions_fig2a = get_group_label_positions(
    adata_fig2a,
    groupby=CELLTYPE_COL,
    basis=BASIS,
)

fig, ax = plt.subplots(figsize=(8, 8))

sc.pl.umap(
    adata_fig2a,
    color=CELLTYPE_COL,
    size=3,
    alpha=0.75,
    ax=ax,
    show=False,
    frameon=False,
    legend_loc=None,
    title=None,
)

rasterize_scatter_points(ax)

add_celltype_labels(
    ax=ax,
    label_positions=label_positions_fig2a,
    final_order=final_order_fig2a,
    label_nudge=LABEL_NUDGE,
    fontsize=10,
    fontfamily=ARIAL_FAMILY,
    fontweight="normal",
    stroke_width=3.2,
    adjust_expand=(1.12, 1.22),
    adjust_force=(0.25, 0.45),
    arrow_lw=0.75,
    arrow_alpha=0.45,
)

ax.set_title(
    "UMAP by cell type",
    fontsize=16,
    pad=18,
    fontweight="normal",
    fontfamily=ARIAL_FAMILY,
)

add_umap_axis_arrows(
    ax=ax,
    arrow_x=0.06,
    arrow_y=0.06,
    arrow_len=0.15,
    lw=2,
    label_fontsize=10,
    fontfamily=ARIAL_FAMILY,
)

remove_axis_traces(ax)

fig.savefig(
    "UMAP_cell_type_Arial_editable.pdf",
    bbox_inches="tight",
    transparent=True,
)

plt.show()


# =============================================================================
# 6. Figure 2B-D: lineage-specific UMAPs
# =============================================================================


def create_lineage_subset(adata_obj, lineage_name):
    """Create a lineage-specific AnnData subset from the full object."""
    if "cell_type" not in adata_obj.obs.columns:
        raise ValueError("obs does not contain 'cell_type'.")

    if lineage_name == "Myeloid":
        return adata_obj[adata_obj.obs["cell_type"].isin(["Myeloid cell", "pDC"])].copy()
    if lineage_name == "B_cells":
        return adata_obj[adata_obj.obs["cell_type"] == "B cell"].copy()
    if lineage_name == "ILCs":
        return adata_obj[adata_obj.obs["cell_type"] == "Innate lymphoid cell"].copy()
    if lineage_name == "T_cells":
        return adata_obj[adata_obj.obs["cell_type"] == "T cell"].copy()

    raise ValueError(f"Unknown lineage name: {lineage_name}")


def load_lineage_or_subset(adata_obj, lineage_name):
    """Load a prepared lineage object, or fall back to a subset of the master."""
    configured_path = LINEAGE_INPUT_PATHS[lineage_name]
    if configured_path:
        input_path = Path(configured_path)
        if not input_path.is_file():
            raise FileNotFoundError(f"{lineage_name} input not found: {input_path}")
        lineage = sc.read_h5ad(input_path)
        if (
            "X_umap_logcounts_harmony" not in lineage.obsm
            and "X_umap" not in lineage.obsm
        ):
            raise KeyError(
                f"{lineage_name}: prepared object contains neither "
                "obsm['X_umap_logcounts_harmony'] nor obsm['X_umap']"
            )
        print(f"Loaded prepared {lineage_name} object: {input_path}")
        return lineage
    return create_lineage_subset(adata_obj, lineage_name)


try:
    adata_Myeloid
except NameError:
    adata_Myeloid = load_lineage_or_subset(adata, "Myeloid")

try:
    adata_B
except NameError:
    adata_B = load_lineage_or_subset(adata, "B_cells")

try:
    adata_ILCs
except NameError:
    adata_ILCs = load_lineage_or_subset(adata, "ILCs")

try:
    adata_T
except NameError:
    adata_T = load_lineage_or_subset(adata, "T_cells")


LINEAGE_PLOT_CONFIG = {
    "Myeloid": {
        "adata": adata_Myeloid,
        "celltype_order": [
            "cMonocytes", "ncMonocytes", "mdMac", "trMac", "infMac",
            "cDC2", "pDC", "ASDC",
        ],
        "exclude_celltypes": ["Basophil", "Neutrophils"],
        "output": "Myeloid_UMAP_logcounts_Harmony.pdf",
    },
    "B_cells": {
        "adata": adata_B,
        "celltype_order": ["Naive B", "Memory B", "Plasma cell"],
        "exclude_celltypes": [],
        "output": "B_cells_UMAP_logcounts_Harmony.pdf",
    },
    "ILCs": {
        "adata": adata_ILCs,
        "celltype_order": ["NK bright", "Transit. NK", "NK dim", "ILC2"],
        "exclude_celltypes": [],
        "output": "ILCs_UMAP_logcounts_Harmony.pdf",
    },
    "T_cells": {
        "adata": adata_T,
        "celltype_order": [
            "CD4 Naive", "CD8 Naive", "CD4 Memory", "CD8 Memory", "Treg",
            "CD4 CTL", "CD8 CTL", "MAIT", "γδT", "NKT",
        ],
        "exclude_celltypes": [],
        "output": "T_cells_UMAP_logcounts_Harmony.pdf",
    },
}


def plot_lineage_umap(
    adata_obj,
    lineage_name,
    celltype_order,
    output_file,
    celltype_col=CELLTYPE_COL,
    preferred_basis="X_umap_logcounts_harmony",
    fallback_basis="X_umap",
    exclude_celltypes=None,
    color_map=None,
    point_size=3,
    figsize=(8, 8),
    label_fontsize=9,
):
    """Plot one lineage-specific UMAP using the original visual settings."""
    if exclude_celltypes is None:
        exclude_celltypes = []
    if color_map is None:
        color_map = {}

    adata_plot = adata_obj.copy()

    if preferred_basis in adata_plot.obsm:
        basis = preferred_basis
    elif fallback_basis in adata_plot.obsm:
        basis = fallback_basis
        print(
            f"Warning: {lineage_name} does not contain {preferred_basis}. "
            f"Using {fallback_basis} instead."
        )
    else:
        raise ValueError(
            f"{lineage_name} does not contain either {preferred_basis} "
            f"or {fallback_basis} in .obsm."
        )

    if celltype_col not in adata_plot.obs.columns:
        raise ValueError(f"{lineage_name}: obs does not contain '{celltype_col}'.")

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
    final_order = [cell_type for cell_type in celltype_order if cell_type in observed_celltypes]

    if len(final_order) == 0:
        raise ValueError(f"{lineage_name}: none of the requested cell types were found.")

    adata_plot = adata_plot[
        adata_plot.obs[celltype_col].astype(str).isin(final_order), :
    ].copy()

    adata_plot.obs[celltype_col] = pd.Categorical(
        adata_plot.obs[celltype_col].astype(str),
        categories=final_order,
        ordered=True,
    )

    missing_colors = [cell_type for cell_type in final_order if cell_type not in color_map]
    if missing_colors:
        print(f"Warning: missing colors for {lineage_name}: {missing_colors}")

    adata_plot.uns[f"{celltype_col}_colors"] = [
        color_map.get(cell_type, "#BDBDBD") for cell_type in final_order
    ]

    label_center = get_group_label_positions(
        adata_plot,
        groupby=celltype_col,
        basis=basis,
    )

    fig, ax = plt.subplots(figsize=figsize)

    sc.pl.embedding(
        adata_plot,
        basis=basis,
        color=celltype_col,
        size=point_size,
        alpha=0.75,
        ax=ax,
        show=False,
        frameon=False,
        legend_loc=None,
        title=None,
    )

    rasterize_scatter_points(ax)

    texts = []
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
            weight="normal",
            ha="center",
            va="center",
            color="black",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        )
        texts.append(text)

    adjust_text(
        texts,
        ax=ax,
        expand_text=(1.10, 1.18),
        force_text=(0.22, 0.40),
        arrowprops=dict(
            arrowstyle="-",
            lw=0.65,
            color="#444444",
            alpha=0.40,
        ),
    )

    add_umap_axis_arrows(
        ax=ax,
        arrow_x=0.06,
        arrow_y=0.06,
        arrow_len=0.14,
        lw=1.8,
        label_fontsize=9,
        fontfamily=ARIAL_FAMILY,
    )

    remove_axis_traces(ax)
    ax.set_title("")

    fig.savefig(
        output_file,
        bbox_inches="tight",
        transparent=True,
        dpi=300,
    )

    plt.show()
    print(f"Saved {lineage_name} UMAP to: {output_file}")

    return adata_plot


lineage_umap_results = {}

for lineage_name, config in LINEAGE_PLOT_CONFIG.items():
    print("\n" + "=" * 80)
    print(f"Plotting lineage: {lineage_name}")
    print("=" * 80)

    lineage_umap_results[lineage_name] = plot_lineage_umap(
        adata_obj=config["adata"],
        lineage_name=lineage_name,
        celltype_order=config["celltype_order"],
        output_file=config["output"],
        celltype_col=CELLTYPE_COL,
        preferred_basis="X_umap_logcounts_harmony",
        fallback_basis="X_umap",
        exclude_celltypes=config["exclude_celltypes"],
        color_map=SOFT_COLOR_MAP,
        point_size=3,
        figsize=(8, 8),
        label_fontsize=9,
    )


# =============================================================================
# 7. Figure 2F-G: sample counts by disease in CSF and PBMC compartments
# =============================================================================


def split_csf_and_pbmc(adata_obj, tissue_col="tissue"):
    """Split AnnData into CSF and PBMC/BLD subsets using the original tissue rules."""
    if tissue_col not in adata_obj.obs.columns:
        raise KeyError(f"Column '{tissue_col}' not found in adata.obs")

    csf_mask = adata_obj.obs[tissue_col].astype(str).str.strip().str.upper() == "CSF"
    if csf_mask.sum() == 0:
        print("Warning: no cells found with tissue == 'CSF'")

    pbmc_mask = adata_obj.obs[tissue_col].isin(["PBMC", "PB", "BLD"])

    return adata_obj[csf_mask].copy(), adata_obj[:, :][pbmc_mask, :].copy()


def plot_disease_sample_count(
    adata_obj,
    disease_col="Disease",
    orig_ident_col="orig.ident",
    figsize=(12, 8),
    dpi=100,
    title="Total Sample Count by Disease",
    xlabel="Disease Type",
    ylabel="Number of Samples",
    min_label_threshold=2.5,
    name_max_length=12,
    color_map="tab20",
):
    """Plot the number of unique samples per disease group."""
    required_cols = [disease_col, orig_ident_col]
    missing_cols = [column for column in required_cols if column not in adata_obj.obs.columns]
    if missing_cols:
        raise ValueError(f"adata.obs is missing required columns: {', '.join(missing_cols)}")

    sample_metadata = adata_obj.obs[[disease_col, orig_ident_col]].drop_duplicates()
    plot_data = sample_metadata.groupby(disease_col, observed=False).size().sort_values(ascending=False)

    disease_list = plot_data.index.tolist()
    sample_counts = plot_data.values

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    colors = plt.get_cmap(color_map)(np.linspace(0, 1, len(disease_list)))

    bars = ax.bar(
        range(len(disease_list)),
        sample_counts,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )

    display_names = [
        disease[:name_max_length - 2] + ".." if len(disease) > name_max_length else disease
        for disease in disease_list
    ]

    ax.set_xticks(range(len(disease_list)))
    ax.set_xticklabels(display_names, rotation=40, ha="right", fontsize=11)

    for index, (count, bar) in enumerate(zip(sample_counts, bars)):
        if count >= min_label_threshold:
            brightness = np.mean(colors[index][:3])
            text_color = "white" if brightness < 0.6 else "#333333"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"n={int(count)}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=text_color,
            )

    ax.set_ylim(0, sample_counts.max() * 1.15 if sample_counts.max() > 0 else 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, color="#e0e0e0")
    ax.set_axisbelow(True)

    ax.set_title(title, fontsize=18, pad=20, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=14, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=15)
    ax.tick_params(axis="y", labelsize=11)

    plt.tight_layout()
    return fig, ax


adata_csf, adata_pbmc = split_csf_and_pbmc(adata_umap, tissue_col="tissue")

fig, ax = plot_disease_sample_count(
    adata_csf,
    disease_col="Disease",
    orig_ident_col="orig.ident",
    title="Custom Disease Sample Count Plot",
    min_label_threshold=0.0,
    color_map="Set3",
)
fig.savefig("disease_sample_count_csf.pdf", dpi=300, bbox_inches="tight")

fig, ax = plot_disease_sample_count(
    adata_pbmc,
    disease_col="Disease",
    orig_ident_col="orig.ident",
    title="Custom Disease Sample Count Plot",
    min_label_threshold=0.0,
    color_map="Set3",
)
fig.savefig("disease_sample_count_pbmc.pdf", dpi=300, bbox_inches="tight")


# =============================================================================
# 8. Figure 2H: marker-gene DotPlot
# =============================================================================

MARKER_DICT = {
    "B/Plasma": ["CD19", "MS4A1", "TCL1A", "CD27", "CD38", "JCHAIN"],
    "T_Naive": ["CD3E", "CD4", "CCR7", "TCF7", "IL2RA"],
    "Cytotoxic": ["CD8A", "GZMK", "GZMB", "PRF1", "GNLY", "NKG7", "SLC4A10"],
    "MAIT/γδT": ["KLRB1", "TRDC", "TRGC1", "IL7R"],
    "NK/ILC": ["KIT", "GATA3", "XCL1", "FCGR3A", "CX3CR1"],
    "Monocytes": ["CD14", "CCR2", "FCGR3B", "FCN1", "VCAN"],
    "Macrophages": ["S100A12", "C1QA", "C1QC", "APOE", "CD163", "F13A1", "LYZ"],
    "InfMac": ["CTSS", "CD44", "GBP1", "ISG15", "IFITM1"],
    "mdMac": ["FCN1", "VCAN", "S100A8", "S100A9", "CD36"],
    "DCs": ["CD1C", "CLEC10A", "FCER1A", "HLA-DRA", "AXL", "SIGLEC6"],
}

DOTPLOT_CELL_TYPE_ORDER = [
    "Naive B", "Memory B", "Plasma cell",
    "CD4 Naive", "CD8 Naive", "CD4 Memory", "CD8 Memory",
    "Treg", "CD4 CTL", "CD8 CTL", "MAIT", "γδT", "NKT",
    "NK bright", "Transit. NK", "NK dim", "ILC2",
    "cMonocytes", "ncMonocytes", "mdMac", "trMac", "infMac",
    "cDC2", "pDC", "ASDC",
]

DOTPLOT_EXCLUDE_CELLTYPES = ["Basophil", "Transitional B"]
DOTPLOT_OUTPUT_PREFIX = "dotplot_marker_genes_with_brackets"

LEGEND_Y_SHIFT = 0.18
CBAR_X_SHIFT = 0.00
SIZE_X_SHIFT = 0.03
CBAR_WIDTH = 0.012
CBAR_HEIGHT = 0.32
SIZE_WIDTH = 0.10
SIZE_HEIGHT = 0.35


def move_resize_ax(ax, width, height, x_shift=0, y_shift=0):
    """Move and resize an axis while preserving its center."""
    pos = ax.get_position()
    center_x = pos.x0 + pos.width / 2 + x_shift
    center_y = pos.y0 + pos.height / 2 + y_shift
    ax.set_position([
        center_x - width / 2,
        center_y - height / 2,
        width,
        height,
    ])


def set_axis_font(ax, fontsize=8):
    """Set font family and size for all labels and text in an axis."""
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(ARIAL_FAMILY)
        label.set_fontsize(fontsize)

    for text in ax.texts:
        text.set_fontfamily(ARIAL_FAMILY)
        text.set_fontsize(fontsize)


def filter_marker_dict_by_genes(marker_dict, var_names):
    """Keep only marker genes present in the AnnData object."""
    clean_dict = {}
    missing_gene_records = {}
    var_names = set(var_names)

    for group_name, genes in marker_dict.items():
        present_genes = [gene for gene in genes if gene in var_names]
        missing_genes = [gene for gene in genes if gene not in var_names]

        if present_genes:
            clean_dict[group_name] = present_genes
        if missing_genes:
            missing_gene_records[group_name] = missing_genes

    if missing_gene_records:
        print("Warning: the following marker genes were not found and were skipped:")
        for group_name, genes in missing_gene_records.items():
            print(f"  {group_name}: {genes}")

    return clean_dict


def plot_marker_gene_dotplot(
    adata_obj,
    marker_dict,
    cell_type_order,
    exclude_celltypes,
    celltype_col=CELLTYPE_COL,
    expression_layer=EXPRESSION_LAYER,
    output_prefix=DOTPLOT_OUTPUT_PREFIX,
):
    """Generate the marker-gene DotPlot using the original legend layout."""
    if celltype_col not in adata_obj.obs.columns:
        raise ValueError(f"obs does not contain '{celltype_col}'.")

    if expression_layer not in adata_obj.layers:
        raise ValueError(f"layers does not contain '{expression_layer}'.")

    adata_plot = adata_obj[
        ~adata_obj.obs[celltype_col].astype(str).isin(exclude_celltypes), :
    ].copy()

    existing_cell_types = adata_plot.obs[celltype_col].astype(str).unique().tolist()
    cell_type_order_use = [
        cell_type for cell_type in cell_type_order
        if cell_type in existing_cell_types
    ]

    if len(cell_type_order_use) == 0:
        raise ValueError("None of the specified cell types were found in the plotting object.")

    adata_plot.obs[celltype_col] = pd.Categorical(
        adata_plot.obs[celltype_col].astype(str),
        categories=cell_type_order_use,
        ordered=True,
    )

    clean_marker_dict = filter_marker_dict_by_genes(
        marker_dict=marker_dict,
        var_names=adata_plot.var_names,
    )

    dotplot = sc.pl.dotplot(
        adata_plot,
        var_names=clean_marker_dict,
        groupby=celltype_col,
        categories_order=cell_type_order_use,
        layer=expression_layer,
        standard_scale="var",
        smallest_dot=2,
        cmap="Reds",
        colorbar_title="Scaled mean expression",
        size_title="Fraction of cells\nin group (%)",
        var_group_rotation=0,
        figsize=(15, 7),
        return_fig=True,
        show=False,
    )

    dotplot.make_figure()
    axes_dict = dotplot.get_axes()
    main_ax = axes_dict["mainplot_ax"]

    main_ax.tick_params(axis="x", labelsize=8, labelrotation=65)
    main_ax.tick_params(axis="y", labelsize=9)

    for label in main_ax.get_xticklabels():
        label.set_fontfamily(ARIAL_FAMILY)
        label.set_horizontalalignment("right")

    for label in main_ax.get_yticklabels():
        label.set_fontfamily(ARIAL_FAMILY)

    if "color_legend_ax" in axes_dict:
        color_ax = axes_dict["color_legend_ax"]
        color_ax.clear()

        move_resize_ax(
            color_ax,
            width=CBAR_WIDTH,
            height=CBAR_HEIGHT,
            x_shift=CBAR_X_SHIFT,
            y_shift=LEGEND_Y_SHIFT,
        )

        scalar_mappable = ScalarMappable(
            norm=Normalize(vmin=0, vmax=1),
            cmap=plt.get_cmap("Reds"),
        )
        scalar_mappable.set_array([])

        colorbar = plt.colorbar(
            scalar_mappable,
            cax=color_ax,
            orientation="vertical",
        )

        colorbar.set_label(
            "Scaled mean expression",
            fontfamily=ARIAL_FAMILY,
            fontsize=9,
            labelpad=6,
        )
        colorbar.ax.tick_params(labelsize=8)
        colorbar.ax.yaxis.set_ticks_position("right")
        colorbar.ax.yaxis.set_label_position("right")
        set_axis_font(colorbar.ax, fontsize=8)

    if "size_legend_ax" in axes_dict:
        size_ax = axes_dict["size_legend_ax"]
        size_ax.clear()

        move_resize_ax(
            size_ax,
            width=SIZE_WIDTH,
            height=SIZE_HEIGHT,
            x_shift=SIZE_X_SHIFT,
            y_shift=LEGEND_Y_SHIFT,
        )

        legend_values = [1.00, 0.80, 0.60, 0.40, 0.20]
        legend_labels = ["100", "80", "60", "40", "20"]
        largest_dot = getattr(dotplot, "largest_dot", 200)
        smallest_dot = getattr(dotplot, "smallest_dot", 2)
        legend_sizes = [
            smallest_dot + value * (largest_dot - smallest_dot)
            for value in legend_values
        ]
        y_positions = np.array([3.2, 2.7, 2.2, 1.7, 1.2])

        size_ax.scatter(
            np.zeros(len(legend_values)),
            y_positions,
            s=legend_sizes,
            color="lightgray",
            edgecolor="black",
            linewidth=0.6,
        )

        for y_pos, label in zip(y_positions, legend_labels):
            size_ax.text(
                0.35,
                y_pos,
                label,
                va="center",
                ha="left",
                fontsize=8,
                fontfamily=ARIAL_FAMILY,
            )

        size_ax.text(
            0,
            3.75,
            "Fraction of cells\nin group (%)",
            ha="center",
            va="bottom",
            fontsize=8,
            fontfamily=ARIAL_FAMILY,
        )

        size_ax.set_xlim(-0.6, 1.5)
        size_ax.set_ylim(-0.1, 4.2)
        size_ax.axis("off")

    if "group_extra_ax" in axes_dict:
        for text in axes_dict["group_extra_ax"].texts:
            text.set_fontfamily(ARIAL_FAMILY)
            text.set_fontsize(9)

    dotplot.fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    dotplot.fig.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {output_prefix}.pdf")
    print(f"Saved: {output_prefix}.png")

    return adata_plot, dotplot


adata_dotplot, marker_dotplot = plot_marker_gene_dotplot(
    adata_obj=adata,
    marker_dict=MARKER_DICT,
    cell_type_order=DOTPLOT_CELL_TYPE_ORDER,
    exclude_celltypes=DOTPLOT_EXCLUDE_CELLTYPES,
    celltype_col=CELLTYPE_COL,
    expression_layer=EXPRESSION_LAYER,
    output_prefix=DOTPLOT_OUTPUT_PREFIX,
)


# =============================================================================
# 9. Figure 2I: CSF/PBMC side-by-side UMAPs
# =============================================================================

TISSUE_MAPPING = {
    "CSF": "CSF",
    "PB": "PBMC",
    "PBMC": "PBMC",
    "BLD": "PBMC",
}


def normalize_tissue_column(adata_obj, tissue_col="tissue", mapping_dict=None):
    """Map tissue labels to CSF and PBMC using the original rules."""
    if mapping_dict is None:
        mapping_dict = TISSUE_MAPPING

    adata_obj.obs[tissue_col] = adata_obj.obs[tissue_col].map(mapping_dict)

    if isinstance(adata_obj.obs[tissue_col].dtype, pd.CategoricalDtype):
        adata_obj.obs[tissue_col] = adata_obj.obs[tissue_col].cat.remove_unused_categories()

    return adata_obj


def plot_umap_one_tissue_on_ax(
    adata_obj,
    tissue_name,
    ax,
    tissue_col="tissue",
    celltype_col=CELLTYPE_COL,
    cell_type_order=None,
    color_map=None,
    basis=BASIS,
    fontfamily=ARIAL_FAMILY,
    global_xlim=None,
    global_ylim=None,
    title=None,
):
    """Plot one tissue-specific UMAP panel on a provided axis."""
    if cell_type_order is None:
        cell_type_order = []
    if color_map is None:
        color_map = {}

    adata_sub = adata_obj[adata_obj.obs[tissue_col].astype(str) == tissue_name].copy()

    if adata_sub.n_obs == 0:
        print(f"Warning: no cells found for tissue = {tissue_name}")
        ax.set_visible(False)
        return ax

    present_order = [
        cell_type for cell_type in cell_type_order
        if cell_type in adata_sub.obs[celltype_col].astype(str).unique()
    ]

    extra_types = [
        cell_type for cell_type in sorted(adata_sub.obs[celltype_col].astype(str).unique())
        if cell_type not in present_order
    ]

    final_order = present_order + extra_types

    adata_sub.obs[celltype_col] = pd.Categorical(
        adata_sub.obs[celltype_col].astype(str),
        categories=final_order,
        ordered=True,
    )

    adata_sub.uns[f"{celltype_col}_colors"] = [
        color_map.get(cell_type, "#BDBDBD") for cell_type in final_order
    ]

    label_positions = get_group_label_positions(
        adata_sub,
        groupby=celltype_col,
        basis=basis,
    )

    sc.pl.umap(
        adata_sub,
        color=celltype_col,
        size=3,
        alpha=0.75,
        ax=ax,
        show=False,
        frameon=False,
        legend_loc=None,
        title=None,
    )

    if global_xlim is not None:
        ax.set_xlim(global_xlim)
    if global_ylim is not None:
        ax.set_ylim(global_ylim)

    rasterize_scatter_points(ax)

    add_celltype_labels(
        ax=ax,
        label_positions=label_positions,
        final_order=final_order,
        label_nudge=LABEL_NUDGE,
        fontsize=9,
        fontfamily=fontfamily,
        fontweight="normal",
        stroke_width=3.0,
        adjust_expand=(1.10, 1.18),
        adjust_force=(0.22, 0.40),
        arrow_lw=0.65,
        arrow_alpha=0.40,
    )

    ax.set_title(
        title if title is not None else tissue_name,
        fontsize=15,
        pad=12,
        fontweight="normal",
        fontfamily=fontfamily,
    )

    add_umap_axis_arrows(
        ax=ax,
        arrow_x=0.06,
        arrow_y=0.06,
        arrow_len=0.14,
        lw=1.8,
        label_fontsize=9,
        fontfamily=fontfamily,
    )

    return ax


def plot_csf_pbmc_umap_side_by_side(
    adata_obj,
    tissue_col="tissue",
    tissue_order=("CSF", "PBMC"),
    celltype_col=CELLTYPE_COL,
    cell_type_order=CELL_TYPE_ORDER,
    color_map=SOFT_COLOR_MAP,
    basis=BASIS,
    output_file="UMAP_cell_type_CSF_PBMC_side_by_side_Arial_editable.pdf",
):
    """Plot CSF and PBMC UMAPs side by side with shared UMAP limits."""
    adata_plot = adata_obj.copy()
    adata_plot = normalize_tissue_column(adata_plot, tissue_col=tissue_col)

    print(adata_plot.obs[tissue_col].unique())

    adata_plot.obs[celltype_col] = adata_plot.obs[celltype_col].astype(str)
    adata_plot.obs[tissue_col] = adata_plot.obs[tissue_col].astype(str)

    print(adata_plot.obs[tissue_col].value_counts())

    global_xy = adata_plot.obsm[basis]
    x_min, x_max = global_xy[:, 0].min(), global_xy[:, 0].max()
    y_min, y_max = global_xy[:, 1].min(), global_xy[:, 1].max()
    x_pad = (x_max - x_min) * 0.04
    y_pad = (y_max - y_min) * 0.04
    global_xlim = (x_min - x_pad, x_max + x_pad)
    global_ylim = (y_min - y_pad, y_max + y_pad)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.5, 7.5),
        sharex=True,
        sharey=True,
    )

    for ax, tissue_name in zip(axes, tissue_order):
        plot_umap_one_tissue_on_ax(
            adata_obj=adata_plot,
            tissue_name=tissue_name,
            ax=ax,
            tissue_col=tissue_col,
            celltype_col=celltype_col,
            cell_type_order=cell_type_order,
            color_map=color_map,
            basis=basis,
            fontfamily=ARIAL_FAMILY,
            global_xlim=global_xlim,
            global_ylim=global_ylim,
            title=tissue_name,
        )

    plt.subplots_adjust(
        wspace=0.04,
        left=0.02,
        right=0.98,
        top=0.92,
        bottom=0.04,
    )

    fig.savefig(
        output_file,
        bbox_inches="tight",
        transparent=True,
    )

    plt.show()
    print(f"Saved: {output_file}")

    return fig, axes


# The original code used the DotPlot-filtered object for the CSF/PBMC UMAP,
# which removes Basophil and Transitional B before the tissue split.
plot_csf_pbmc_umap_side_by_side(
    adata_obj=adata_dotplot,
    tissue_col="tissue",
    tissue_order=("CSF", "PBMC"),
    celltype_col=CELLTYPE_COL,
    cell_type_order=CELL_TYPE_ORDER,
    color_map=SOFT_COLOR_MAP,
    basis=BASIS,
    output_file="UMAP_cell_type_CSF_PBMC_side_by_side_Arial_editable.pdf",
)
