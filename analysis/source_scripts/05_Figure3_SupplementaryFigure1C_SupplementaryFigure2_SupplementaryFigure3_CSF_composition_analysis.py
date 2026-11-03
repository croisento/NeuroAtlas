
# =============================================================================
# Figure 3, Supplementary Figure 1C, Supplementary Figure 2, and Supplementary Figure 3: CSF immune composition analysis
# =============================================================================
# This script performs the following analyses:
#   1. Generate Supplementary Figure 1C: disease-wise PCA outlier detection.
#   2. Remove detected outlier samples from the CSF AnnData object.
#   3. Generate overall and lineage-specific composition heatmaps.
#   4. Generate a 100% stacked barplot of cell-subtype proportions by disease.
#   5. Generate the Supplementary Figure 2 CSF ROE heatmap with age and disease annotations.
#   6. Generate dataset-stratified sample-level boxplots for Figure 3G-O.
#
# Input:
#   /home/lishaogang/NEU2026_July/adata_final_csf.h5ad
#
# Main outputs:
#   SupplementaryFigure1C_PCA_Results/SupplementaryFigure1C_CSF_PCA_outlier_detection.pdf
#   SupplementaryFigure1C_PCA_Results/SupplementaryFigure1C_CSF_PCA_outlier_detection.png
#   SupplementaryFigure1C_PCA_Results/SupplementaryFigure1C_CSF_PCA_outlier_summary.csv
#   Figure3_and_SuppFigure3_custom_layout_final/Figure3_final.pdf
#   Figure3_and_SuppFigure3_custom_layout_final/Figure3_final.png
#   Figure3_and_SuppFigure3_custom_layout_final/SupplementaryFigure3_final.pdf
#   Figure3_and_SuppFigure3_custom_layout_final/SupplementaryFigure3_final.png
#   Pairwise_Heatmaps/Cell_Type_Counts_by_Disease.csv
#   Pairwise_Heatmaps/Cell_Type_Proportion_StackedBar_Grouped_NoFrame_OneLegendColumn.pdf
#   Pairwise_Heatmaps/Cell_Type_Proportion_StackedBar_Grouped_NoFrame_OneLegendColumn.png
#   Figure3G_O_sample_level_cell_proportions/Figure3G_O_sample_level_proportions_combined.pdf
#   Figure3G_O_sample_level_cell_proportions/Figure3G_O_sample_level_proportions_combined.png
#   SupplementaryFigure2_ROE_Heatmap_Results/SupplementaryFigure2_CSF_ROE_heatmap.pdf
#   SupplementaryFigure2_ROE_Heatmap_Results/SupplementaryFigure2_CSF_ROE_heatmap.png
#   SupplementaryFigure2_ROE_Heatmap_Results/SupplementaryFigure2_CSF_ROE_raw_matrix.csv
#   SupplementaryFigure2_ROE_Heatmap_Results/SupplementaryFigure2_CSF_ROE_plotted_matrix.csv
#   SupplementaryFigure2_ROE_Heatmap_Results/SupplementaryFigure2_CSF_ROE_sample_metadata_plotted_order.csv
#   SupplementaryFigure2_ROE_Heatmap_Results/SupplementaryFigure2_CSF_ROE_group_layout.csv
#   SupplementaryFigure2_ROE_Heatmap_Results/SupplementaryFigure2_CSF_ROE_celltype_order.csv
# =============================================================================

import os
import re
from pathlib import Path

import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager

from scipy.stats import ranksums, zscore
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter, MaxNLocator
from matplotlib.transforms import blended_transform_factory
from matplotlib.gridspec import GridSpec


# =============================================================================
# 1. Global configuration
# =============================================================================

PROJECT_DIR = Path(os.environ.get("NEU_PROJECT_DIR", "/home/lishaogang/NEU2026_July"))
INPUT_H5AD = os.environ.get(
    "CSF_INPUT_H5AD", str(PROJECT_DIR / "adata_final_csf.h5ad")
)
RESULT_ROOT = os.environ.get("FIGURE_RESULT_ROOT", "Result")

SAMPLE_COL = "orig.ident"
DISEASE_COL = "Disease"
DATASET_COL = "dataset"
CELLTYPE_COL = "cell_subtype_short"
BROAD_TYPE_COL = "cell_type"
AGE_COL = "age_numeric"
REF_GROUP = "Younger control"

DISEASE_RENAME_MAP = {
    "MCVAD": "MCI/AD",
    "MCI_AD": "MCI/AD",
    "MCI-AD": "MCI/AD",
    "CAR-T/DIPG": "CAR-T-DIPG",
}

CELLTYPE_RENAME_MAP = {
    "CD4 Naïve": "CD4 Naive",
    "CD8 Naïve": "CD8 Naive",
    "maMac": "mdMac",
}

PCA_OUTLIER_Z_THRESHOLD = 1.5
PCA_MIN_SAMPLES_PER_DISEASE = 5

HEATMAP_OUTDIR = os.path.join(RESULT_ROOT, "Figure3_and_Supplement Figure3", "Figure3_and_SuppFigure3_custom_layout_final")
STACKED_BAR_OUTDIR = os.path.join(RESULT_ROOT, "Figure3_and_Supplement Figure3")
DATASET_BOXPLOT_OUTDIR = os.path.join(RESULT_ROOT, "Figure3G_O")
PCA_OUTDIR = os.path.join(RESULT_ROOT, "Supplement_Figure1", "Supplement_Figure1C")
ROE_OUTDIR = os.path.join(RESULT_ROOT, "Supplement_Figure2")

PCA_OUTPUT_PDF = os.path.join(PCA_OUTDIR, "SupplementaryFigure1C_CSF_PCA_outlier_detection.pdf")
PCA_OUTPUT_PNG = os.path.join(PCA_OUTDIR, "SupplementaryFigure1C_CSF_PCA_outlier_detection.png")
PCA_OUTLIER_TABLE = os.path.join(PCA_OUTDIR, "SupplementaryFigure1C_CSF_PCA_outlier_summary.csv")

FIGURE3_PDF = os.path.join(HEATMAP_OUTDIR, "Figure3_final.pdf")
FIGURE3_PNG = os.path.join(HEATMAP_OUTDIR, "Figure3_final.png")
SUPPLEMENTARY_FIGURE3_PDF = os.path.join(HEATMAP_OUTDIR, "SupplementaryFigure3_final.pdf")
SUPPLEMENTARY_FIGURE3_PNG = os.path.join(HEATMAP_OUTDIR, "SupplementaryFigure3_final.png")

FIGURE3GO_COMBINED_PDF = os.path.join(
    DATASET_BOXPLOT_OUTDIR,
    "Figure3G_O_sample_level_proportions_combined.pdf",
)
FIGURE3GO_COMBINED_PNG = os.path.join(
    DATASET_BOXPLOT_OUTDIR,
    "Figure3G_O_sample_level_proportions_combined.png",
)

ROE_OUTPUT_PDF = os.path.join(ROE_OUTDIR, "SupplementaryFigure2_CSF_ROE_heatmap.pdf")
ROE_OUTPUT_PNG = os.path.join(ROE_OUTDIR, "SupplementaryFigure2_CSF_ROE_heatmap.png")
ROE_TABLE_PREFIX = "SupplementaryFigure2_CSF"

# Figure assignment index:
#   Supplementary Figure 1C: PCA outlier overview
#   Supplementary Figure 2: CSF ROE heatmap
#   Figure 3: raw-P composition heatmap and Figure 3G-O boxplots
#   Supplementary Figure 3: FDR-adjusted composition heatmap

DEFAULT_ROE_COLORS = [
    "#244773",
    "#73A9D0",
    "#F8F8F8",
    "#D4887A",
    "#752C36",
]

for directory in [HEATMAP_OUTDIR, STACKED_BAR_OUTDIR, DATASET_BOXPLOT_OUTDIR, PCA_OUTDIR, ROE_OUTDIR]:
    os.makedirs(directory, exist_ok=True)

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

sns.set_style("white")


DISEASE_ORDER_FULL = [
    "Younger control",
    "Older control",
    "Unknown",
    "MCI/AD",
    "PD",
    "IIH",
    "MS",
    "Ab-mediated IDD",
    "VE",
    "Neuro-COVID",
    "CAR-T-DIPG",
    "NT1",
]

DISEASE_ORDER_STACKED_BAR = [
    "Younger control",
    "Older control",
    "Unknown",
    "MCI/AD",
    "PD",
    "IIH",
    "MS",
]

BROAD_TO_SUBTYPES = {
    "T_cell": [
        "MAIT",
        "CD4 CTL",
        "NKT",
        "CD8 Naive",
        "CD4 Memory",
        "CD8 CTL",
        "CD8 Memory",
        "Treg",
        "γδT",
        "CD4 Naive",
    ],
    "B_cell": [
        "Plasma cell",
        "Naive B",
        "Memory B",
    ],
    "ILCs": [
        "ILC2",
        "NK bright",
        "NK dim",
        "Transit. NK",
    ],
    "Myeloid": [
        "cMonocytes",
        "cDC2",
        "infMac",
        "trMac",
        "ASDC",
        "mdMac",
        "maMac",
        "ncMonocytes",
        "Basophil",
    ],
}

MANUAL_ROW_ORDER = (
    BROAD_TO_SUBTYPES["B_cell"]
    + BROAD_TO_SUBTYPES["T_cell"]
    + BROAD_TO_SUBTYPES["ILCs"]
    + BROAD_TO_SUBTYPES["Myeloid"]
)

STACKED_CELL_ORDER = [
    "Naive B", "Memory B", "Plasma cell",
    "CD4 Naive", "CD8 Naive", "CD4 Memory", "CD8 Memory",
    "Treg", "CD4 CTL", "CD8 CTL", "MAIT", "γδT", "NKT",
    "NK bright", "Transit. NK", "NK dim", "ILC2",
    "cMonocytes", "ncMonocytes", "mdMac", "trMac", "infMac",
    "cDC2", "pDC", "ASDC",
]

STACKED_COLOR_MAP = {
    "Naive B": "#C96F6F", "Memory B": "#B85C68", "Plasma cell": "#B77AB5",
    "CD4 Naive": "#7DA7CF", "CD8 Naive": "#4F9A78", "CD4 Memory": "#AEC9E4",
    "CD8 Memory": "#98C970", "Treg": "#3F6BA7", "CD4 CTL": "#3E7C9D",
    "CD8 CTL": "#C3D894", "MAIT": "#75B38A", "γδT": "#A6B85A",
    "NKT": "#BFB552", "NK bright": "#D7BE66", "Transit. NK": "#A995C1",
    "NK dim": "#C99235", "ILC2": "#DDD36A", "cMonocytes": "#C97891",
    "ncMonocytes": "#9E5F73", "mdMac": "#9568B1", "trMac": "#D6A196",
    "infMac": "#C17DB4", "cDC2": "#B46AB0", "pDC": "#7F8CC1",
    "ASDC": "#B79BD0",
}

FIGURE3GO_TARGET_PANELS = [
    {"panel": "G", "disease": "MS", "cell_type": "Naive B", "display_cell_type": "Naive B"},
    {"panel": "H", "disease": "MS", "cell_type": "Plasma cell", "display_cell_type": "Plasma cell"},
    {"panel": "I", "disease": "MS", "cell_type": "ASDC", "display_cell_type": "ASDC"},
    {"panel": "J", "disease": "MS", "cell_type": "CD4 Naive", "display_cell_type": "CD4 Naïve"},
    {"panel": "K", "disease": "MS", "cell_type": "Treg", "display_cell_type": "Treg"},
    {"panel": "L", "disease": "PD", "cell_type": "NK dim", "display_cell_type": "NK dim"},
    {"panel": "M", "disease": "MCI/AD", "cell_type": "NK dim", "display_cell_type": "NK dim"},
    {"panel": "N", "disease": "Older control", "cell_type": "ASDC", "display_cell_type": "ASDC"},
    {"panel": "O", "disease": "IIH", "cell_type": "Naive B", "display_cell_type": "Naive B"},
]

DATASET_ORDER_BY_DISEASE = {
    "MS": ["GSE133028", "GSE138266", "GSE163005", "GSE194078", "GSE200164", "GSE247134"],
    "PD": ["GSE134577", "GSE141578"],
    "MCI/AD": ["GSE134577", "GSE200164"],
    "Older control": ["GSE134577", "GSE200164"],
    "IIH": ["GSE138266", "GSE163005", "GSE247134"],
}


# =============================================================================
# 2. Shared helper functions
# =============================================================================

def clean_filename(name):
    """Replace characters that are unsafe for file names."""
    return re.sub(r'[\/:*?"<>|]', "_", str(name))


def standardize_metadata(adata, disease_col=DISEASE_COL, celltype_col=CELLTYPE_COL):
    """Standardize disease and cell-subtype labels in AnnData metadata."""
    adata = adata.copy()

    if disease_col in adata.obs.columns:
        adata.obs[disease_col] = (
            adata.obs[disease_col]
            .astype(str)
            .str.strip()
            .replace({
                "MCVAD": "MCI/AD",
                "MCI_AD": "MCI/AD",
                "MCI-AD": "MCI/AD",
                "CAR-T/DIPG": "CAR-T-DIPG",
            })
        )

    if celltype_col in adata.obs.columns:
        adata.obs[celltype_col] = (
            adata.obs[celltype_col]
            .astype(str)
            .str.strip()
            .replace({
                "CD4 Naïve": "CD4 Naive",
                "CD8 Naïve": "CD8 Naive",
                "maMac": "mdMac",
            })
        )

    return adata


def validate_obs_columns(adata, required_cols):
    """Raise an error if any required AnnData obs columns are missing."""
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in adata.obs: {missing_cols}")


def p_to_stars(p_value):
    """Convert a P value to significance stars."""
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def format_raw_p_value_or_stars(p_value):
    """Show significant raw P values as stars and non-significant values as raw P values."""
    if pd.isna(p_value):
        return "p=NA"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return f"p={p_value:.2g}"


def bh_adjust_within_group(stats_df, p_col="p_value", group_col="cell_type"):
    """Apply Benjamini-Hochberg correction within each cell type."""
    stats_df = stats_df.copy()
    stats_df["p_adj"] = np.nan

    for _, idx in stats_df.groupby(group_col).groups.items():
        p_values = stats_df.loc[idx, p_col].values.astype(float)
        valid_mask = np.isfinite(p_values)

        if valid_mask.sum() == 0:
            continue

        _, adjusted_values, _, _ = multipletests(
            p_values[valid_mask],
            method="fdr_bh",
        )

        full_adjusted = np.full(len(p_values), np.nan)
        full_adjusted[valid_mask] = adjusted_values
        stats_df.loc[idx, "p_adj"] = full_adjusted

    return stats_df


def get_robust_absmax(matrix, q=0.98):
    """Calculate a robust symmetric color-scale limit from one matrix."""
    values = matrix.values.flatten()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 1.0
    return float(np.quantile(np.abs(values), q))


def get_global_absmax(matrices, q=0.98):
    """Calculate a robust symmetric color-scale limit across multiple matrices."""
    values = np.concatenate([matrix.values.flatten() for matrix in matrices])
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 1.0
    return float(np.quantile(np.abs(values), q))


def ensure_disease_categories(df, disease_col=DISEASE_COL, disease_order=None):
    """Convert the disease column to an ordered categorical variable."""
    if disease_order is None:
        disease_order = DISEASE_ORDER_FULL

    df = df.copy()
    present_order = [
        disease for disease in disease_order
        if disease in df[disease_col].astype(str).unique()
    ]

    df[disease_col] = pd.Categorical(
        df[disease_col],
        categories=present_order,
        ordered=True,
    )

    return df, present_order


def order_celltypes(observed_celltypes, manual_order):
    """Order observed cell types using a manual order and append any remaining types."""
    observed_celltypes = list(pd.unique(observed_celltypes))
    ordered = [cell_type for cell_type in manual_order if cell_type in observed_celltypes]
    remaining = [cell_type for cell_type in observed_celltypes if cell_type not in ordered]
    return ordered + sorted(remaining)


# =============================================================================
# 3. Supplementary Figure 1C: PCA-based sample outlier detection
# =============================================================================

def build_sample_subtype_matrix(
    adata,
    sample_col=SAMPLE_COL,
    subtype_col=CELLTYPE_COL,
):
    """Build a sample-by-cell-subtype proportion matrix."""
    validate_obs_columns(adata, [sample_col, subtype_col])

    count_df = (
        adata.obs
        .groupby([sample_col, subtype_col], observed=True)
        .size()
        .reset_index(name="count")
    )

    matrix = (
        count_df
        .pivot(index=sample_col, columns=subtype_col, values="count")
        .fillna(0)
    )

    matrix = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)
    return matrix


def detect_outliers_from_pca_coordinates(pca_coords, z_threshold=PCA_OUTLIER_Z_THRESHOLD):
    """Detect outlier samples using the Z-score of the distance from the PCA origin."""
    pca_distance = np.sqrt(pca_coords["PC1"] ** 2 + pca_coords["PC2"] ** 2)
    pca_z = zscore(pca_distance)

    outliers = pca_coords.index[pca_z > z_threshold].tolist()
    normals = pca_coords.index[pca_z <= z_threshold].tolist()

    return outliers, normals, pca_z


def run_pca_outlier_detection_by_disease(
    adata,
    sample_col=SAMPLE_COL,
    disease_col=DISEASE_COL,
    subtype_col=CELLTYPE_COL,
    min_samples=PCA_MIN_SAMPLES_PER_DISEASE,
    z_threshold=PCA_OUTLIER_Z_THRESHOLD,
    output_pdf=PCA_OUTPUT_PDF,
    output_png=PCA_OUTPUT_PNG,
    output_outlier_csv=PCA_OUTLIER_TABLE,
    n_cols=3,
):
    """Generate Supplementary Figure 1C: disease-wise PCA outlier detection in one combined figure."""
    validate_obs_columns(adata, [sample_col, disease_col, subtype_col])

    sample_subtype_matrix = build_sample_subtype_matrix(
        adata,
        sample_col=sample_col,
        subtype_col=subtype_col,
    )

    sample_disease_map = (
        adata.obs[[sample_col, disease_col]]
        .drop_duplicates()
        .set_index(sample_col)[disease_col]
        .to_dict()
    )

    valid_diseases = []
    pca_plot_data = {}
    all_outliers = {}

    observed_diseases = adata.obs[disease_col].astype(str).unique().tolist()
    ordered_diseases = [d for d in DISEASE_ORDER_FULL if d in observed_diseases]
    ordered_diseases += sorted([d for d in observed_diseases if d not in ordered_diseases])

    for disease in ordered_diseases:
        disease_samples = [
            sample for sample in sample_subtype_matrix.index
            if str(sample_disease_map.get(sample)) == str(disease)
        ]

        if len(disease_samples) < min_samples:
            print(f"Skipped {disease}: fewer than {min_samples} samples.")
            continue

        disease_matrix = sample_subtype_matrix.loc[disease_samples]
        disease_matrix_scaled = StandardScaler().fit_transform(disease_matrix)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(disease_matrix_scaled)

        pca_coords = pd.DataFrame(
            pca_result,
            index=disease_samples,
            columns=["PC1", "PC2"],
        )

        outliers, normals, pca_z = detect_outliers_from_pca_coordinates(
            pca_coords,
            z_threshold=z_threshold,
        )

        valid_diseases.append(disease)
        all_outliers[disease] = outliers

        pca_plot_data[disease] = {
            "pca_coords": pca_coords,
            "normals": normals,
            "outliers": outliers,
            "explained_var": pca.explained_variance_ratio_,
            "pca_z": pca_z,
        }

        print(f"{disease}: {len(outliers)} outlier samples detected: {outliers}")

    if len(valid_diseases) == 0:
        raise ValueError("No disease group has enough samples for PCA outlier detection.")

    n_rows = int(np.ceil(len(valid_diseases) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        dpi=100,
        squeeze=False,
    )
    axes = axes.flatten()

    for index, disease in enumerate(valid_diseases):
        ax = axes[index]
        item = pca_plot_data[disease]
        coords = item["pca_coords"]

        ax.scatter(
            coords.loc[item["normals"], "PC1"],
            coords.loc[item["normals"], "PC2"],
            c="skyblue",
            label="Normal",
            s=60,
            alpha=0.8,
        )

        if len(item["outliers"]) > 0:
            ax.scatter(
                coords.loc[item["outliers"], "PC1"],
                coords.loc[item["outliers"], "PC2"],
                c="red",
                label="Outlier",
                s=80,
                edgecolors="black",
            )

            for sample in item["outliers"]:
                ax.text(
                    coords.loc[sample, "PC1"] + 0.05,
                    coords.loc[sample, "PC2"] + 0.05,
                    sample,
                    fontsize=8,
                    color="red",
                )

        ev1, ev2 = item["explained_var"]

        ax.set_title(f"{disease}\nPC1: {ev1:.1%}, PC2: {ev2:.1%}", fontsize=10)
        ax.set_xlabel(f"PC1 ({ev1:.1%})", fontsize=8)
        ax.set_ylabel(f"PC2 ({ev2:.1%})", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(
            fontsize=8,
            frameon=True,
            framealpha=1,
            edgecolor="black",
            facecolor="white",
            borderpad=1.2,
            labelspacing=0.8,
            fancybox=False,
        )

    for index in range(len(valid_diseases), len(axes)):
        axes[index].set_visible(False)

    fig.suptitle(
        "Supplementary Figure 1C",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    if output_png is not None:
        fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

    outlier_rows = []
    for disease, item in pca_plot_data.items():
        coords = item["pca_coords"]
        pca_z = item["pca_z"]
        for sample_idx, sample in enumerate(coords.index):
            outlier_rows.append({
                "Disease": disease,
                "sample": sample,
                "PC1": coords.loc[sample, "PC1"],
                "PC2": coords.loc[sample, "PC2"],
                "pca_distance_z": pca_z[sample_idx],
                "is_outlier": sample in item["outliers"],
            })

    if output_outlier_csv is not None:
        pd.DataFrame(outlier_rows).to_csv(output_outlier_csv, index=False)

    remove_list = sorted({
        sample
        for outlier_samples in all_outliers.values()
        for sample in outlier_samples
    })

    print("\nOutlier samples by disease:")
    for disease, outliers in all_outliers.items():
        print(f"{disease}: {outliers if outliers else 'None'}")

    print("\nFinal sample removal list:")
    print(f"Number of samples to remove: {len(remove_list)}")
    print(f"remove_list_csf = {remove_list}")
    print(f"Saved Supplementary Figure 1C PCA PDF: {output_pdf}")
    if output_png is not None:
        print(f"Saved Supplementary Figure 1C PCA PNG: {output_png}")
    if output_outlier_csv is not None:
        print(f"Saved Supplementary Figure 1C PCA outlier table: {output_outlier_csv}")

    return remove_list, all_outliers, pca_plot_data


# =============================================================================
# 4. Heatmap analysis
# =============================================================================

def calculate_overall_fraction_table(adata):
    """Calculate sample-level cell-subtype fractions relative to all cells in each sample."""
    validate_obs_columns(adata, [SAMPLE_COL, DISEASE_COL, CELLTYPE_COL])

    df = adata.obs[[SAMPLE_COL, DISEASE_COL, CELLTYPE_COL]].copy()

    count_df = (
        df.groupby([SAMPLE_COL, DISEASE_COL, CELLTYPE_COL], observed=True)
        .size()
        .unstack(fill_value=0)
        .stack()
        .reset_index(name="n")
    )

    sample_totals = (
        count_df.groupby(SAMPLE_COL, observed=False)["n"]
        .sum()
        .rename("sample_total")
        .reset_index()
    )

    fraction_df = count_df.merge(sample_totals, on=SAMPLE_COL, how="left")
    fraction_df["fraction"] = fraction_df["n"] / fraction_df["sample_total"] * 100.0
    fraction_df, present_order = ensure_disease_categories(fraction_df)

    return fraction_df, present_order


def calculate_within_lineage_fraction_table(adata, lineage_name, subtype_list):
    """Calculate sample-level subtype fractions relative to a major lineage in each sample."""
    validate_obs_columns(adata, [SAMPLE_COL, DISEASE_COL, CELLTYPE_COL])

    df = adata.obs[[SAMPLE_COL, DISEASE_COL, CELLTYPE_COL]].copy()
    df = df[df[CELLTYPE_COL].isin(subtype_list)].copy()

    count_df = (
        df.groupby([SAMPLE_COL, DISEASE_COL, CELLTYPE_COL], observed=True)
        .size()
        .unstack(fill_value=0)
        .stack()
        .reset_index(name="n")
    )

    lineage_totals = (
        count_df.groupby([SAMPLE_COL, DISEASE_COL], observed=False)["n"]
        .sum()
        .rename("lineage_total")
        .reset_index()
    )

    fraction_df = count_df.merge(lineage_totals, on=[SAMPLE_COL, DISEASE_COL], how="left")
    fraction_df["fraction"] = fraction_df["n"] / fraction_df["lineage_total"] * 100.0
    fraction_df["lineage"] = lineage_name
    fraction_df, present_order = ensure_disease_categories(fraction_df)

    return fraction_df, present_order


def compare_each_disease_vs_healthy(fraction_df, ref_group=REF_GROUP):
    """Compare disease sample-level fractions against the configured control."""
    rows = []

    diseases = [
        disease for disease in fraction_df[DISEASE_COL].cat.categories
        if disease != ref_group
    ]

    cell_types = list(fraction_df[CELLTYPE_COL].dropna().unique())

    for cell_type in cell_types:
        sub_df = fraction_df[fraction_df[CELLTYPE_COL] == cell_type].copy()

        ref_values = sub_df.loc[sub_df[DISEASE_COL] == ref_group, "fraction"].values
        ref_median = np.median(ref_values) if len(ref_values) > 0 else np.nan

        for disease in diseases:
            disease_values = sub_df.loc[sub_df[DISEASE_COL] == disease, "fraction"].values
            disease_median = np.median(disease_values) if len(disease_values) > 0 else np.nan

            if len(ref_values) >= 2 and len(disease_values) >= 2:
                _, p_value = ranksums(ref_values, disease_values)
            else:
                p_value = np.nan

            if np.isfinite(disease_median) and np.isfinite(ref_median):
                median_diff = disease_median - ref_median
            else:
                median_diff = np.nan

            rows.append({
                "cell_type": cell_type,
                "Disease": disease,
                "ref_group": ref_group,
                "ref_median": ref_median,
                "disease_median": disease_median,
                "median_diff": median_diff,
                "p_value": p_value,
            })

    stats_df = pd.DataFrame(rows)
    stats_df = bh_adjust_within_group(stats_df, p_col="p_value", group_col="cell_type")
    stats_df["signif_raw"] = stats_df["p_value"].apply(p_to_stars)
    stats_df["signif_fdr"] = stats_df["p_adj"].apply(p_to_stars)

    return stats_df


def convert_stats_to_heatmap_matrices(stats_df, row_order, col_order, signif_col="signif_raw"):
    """Convert a statistics table to effect-size and significance matrices."""
    effect_matrix = (
        stats_df
        .pivot(index="cell_type", columns="Disease", values="median_diff")
        .reindex(index=row_order, columns=col_order)
    )

    star_matrix = (
        stats_df
        .pivot(index="cell_type", columns="Disease", values=signif_col)
        .reindex(index=row_order, columns=col_order)
    )

    return effect_matrix, star_matrix


def build_overall_heatmap_bundle(adata):
    """Build all tables and matrices required for the overall heatmap."""
    fraction_df, present_order = calculate_overall_fraction_table(adata)
    stats_df = compare_each_disease_vs_healthy(fraction_df, ref_group=REF_GROUP)

    disease_cols = [disease for disease in present_order if disease != REF_GROUP]
    row_order = order_celltypes(stats_df["cell_type"].unique(), MANUAL_ROW_ORDER)

    effect_raw, star_raw = convert_stats_to_heatmap_matrices(
        stats_df, row_order, disease_cols, signif_col="signif_raw"
    )

    effect_fdr, star_fdr = convert_stats_to_heatmap_matrices(
        stats_df, row_order, disease_cols, signif_col="signif_fdr"
    )

    return {
        "frac_df": fraction_df,
        "stats_df": stats_df,
        "effect_raw": effect_raw,
        "star_raw": star_raw,
        "effect_fdr": effect_fdr,
        "star_fdr": star_fdr,
    }


def build_lineage_heatmap_bundle(adata, lineage_name, subtype_list):
    """Build all tables and matrices required for one lineage-specific heatmap."""
    fraction_df, present_order = calculate_within_lineage_fraction_table(
        adata,
        lineage_name,
        subtype_list,
    )

    stats_df = compare_each_disease_vs_healthy(fraction_df, ref_group=REF_GROUP)

    disease_cols = [disease for disease in present_order if disease != REF_GROUP]
    row_order = [cell_type for cell_type in subtype_list if cell_type in stats_df["cell_type"].unique()]

    effect_raw, star_raw = convert_stats_to_heatmap_matrices(
        stats_df, row_order, disease_cols, signif_col="signif_raw"
    )

    effect_fdr, star_fdr = convert_stats_to_heatmap_matrices(
        stats_df, row_order, disease_cols, signif_col="signif_fdr"
    )

    return {
        "frac_df": fraction_df,
        "stats_df": stats_df,
        "effect_raw": effect_raw,
        "star_raw": star_raw,
        "effect_fdr": effect_fdr,
        "star_fdr": star_fdr,
    }


def plot_custom_A_BC_DE_layout(
    overall_item,
    sub_items,
    savepath,
    page_title="Figure",
    show_in_notebook=True,
    overall_absmax=None,
    sub_absmax=None,
    overall_cbar_label="Median difference vs Younger control (%)",
    sub_cbar_label="Median difference vs Younger control (%)",
    output_png=None,
):
    """Plot one overall heatmap and four lineage-specific heatmaps in an A/B-C/D-E layout."""
    if overall_absmax is None:
        overall_absmax = get_robust_absmax(overall_item["effect_mat"], q=0.98)

    if sub_absmax is None:
        sub_absmax = get_global_absmax([item["effect_mat"] for item in sub_items], q=0.98)

    fig = plt.figure(figsize=(11.69, 8.27), dpi=300)

    grid_spec = gridspec.GridSpec(
        nrows=20,
        ncols=24,
        figure=fig,
        left=0.04,
        right=0.86,
        top=0.91,
        bottom=0.08,
        wspace=1.55,
        hspace=1.95,
    )

    fig.text(
        0.02,
        0.965,
        page_title,
        fontsize=14,
        fontweight="bold",
        color="white",
        ha="left",
        va="top",
        bbox=dict(facecolor="black", edgecolor="black", boxstyle="square,pad=0.15"),
    )

    ax_a = fig.add_subplot(grid_spec[3:18, 0:5])

    sns.heatmap(
        overall_item["effect_mat"],
        cmap="RdBu_r",
        center=0,
        vmin=-overall_absmax,
        vmax=overall_absmax,
        linewidths=0.22,
        linecolor="white",
        annot=overall_item["star_mat"],
        fmt="",
        annot_kws={"fontsize": 6, "fontweight": "bold"},
        cbar=False,
        ax=ax_a,
    )

    ax_a.set_title(overall_item["title"], fontsize=9, pad=4)
    ax_a.set_xticklabels(ax_a.get_xticklabels(), rotation=45, ha="right", fontsize=5.5)
    ax_a.set_yticklabels(ax_a.get_yticklabels(), fontsize=5.5)
    ax_a.set_xlabel("")
    ax_a.set_ylabel("")

    panel_positions = [
        (3, 8, 10, 15),
        (3, 16, 10, 23),
        (12, 8, 16, 15),
        (12, 16, 16, 23),
    ]

    for item, position in zip(sub_items, panel_positions):
        row_start, col_start, row_end, col_end = position
        ax = fig.add_subplot(grid_spec[row_start:row_end, col_start:col_end])

        sns.heatmap(
            item["effect_mat"],
            cmap="RdBu_r",
            center=0,
            vmin=-sub_absmax,
            vmax=sub_absmax,
            linewidths=0.22,
            linecolor="white",
            annot=item["star_mat"],
            fmt="",
            annot_kws={"fontsize": 6.5, "fontweight": "bold"},
            cbar=False,
            ax=ax,
        )

        ax.set_title(item["title"], fontsize=9, pad=4)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=5.5)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=5.5)
        ax.set_xlabel("")
        ax.set_ylabel("")

    panel_positions_text = {
        "A": (0.045, 0.83),
        "B": (0.30, 0.83),
        "C": (0.565, 0.83),
        "D": (0.30, 0.43),
        "E": (0.565, 0.43),
    }

    for label, (x_pos, y_pos) in panel_positions_text.items():
        fig.text(x_pos, y_pos, label, fontsize=15, fontweight="bold")

    cbar_y = 0.29
    cbar_height = 0.39
    cbar_width = 0.008

    cbar_ax_a = fig.add_axes([0.207, cbar_y, cbar_width, cbar_height])
    sm_a = ScalarMappable(
        cmap="RdBu_r",
        norm=Normalize(vmin=-overall_absmax, vmax=overall_absmax),
    )
    sm_a.set_array([])
    cbar_a = fig.colorbar(sm_a, cax=cbar_ax_a)
    cbar_a.ax.set_title("A", fontsize=7, pad=4)
    cbar_a.set_label(overall_cbar_label, fontsize=7)
    cbar_a.ax.tick_params(labelsize=6)

    cbar_ax_sub = fig.add_axes([0.835, cbar_y, cbar_width, cbar_height])
    sm_sub = ScalarMappable(
        cmap="RdBu_r",
        norm=Normalize(vmin=-sub_absmax, vmax=sub_absmax),
    )
    sm_sub.set_array([])
    cbar_sub = fig.colorbar(sm_sub, cax=cbar_ax_sub)
    cbar_sub.ax.set_title("B–E", fontsize=7, pad=4)
    cbar_sub.set_label(sub_cbar_label, fontsize=7)
    cbar_sub.ax.tick_params(labelsize=6)

    fig.savefig(savepath, bbox_inches="tight")
    if output_png is not None:
        fig.savefig(output_png, dpi=300, bbox_inches="tight")

    if show_in_notebook:
        plt.show()
    else:
        plt.close()

    print(f"Saved PDF: {savepath}")
    if output_png is not None:
        print(f"Saved PNG: {output_png}")


def run_heatmap_analysis(adata):
    """Build composition heatmaps and save raw and FDR-adjusted figures."""
    overall_bundle = build_overall_heatmap_bundle(adata)
    t_bundle = build_lineage_heatmap_bundle(adata, "T_cell", BROAD_TO_SUBTYPES["T_cell"])
    b_bundle = build_lineage_heatmap_bundle(adata, "B_cell", BROAD_TO_SUBTYPES["B_cell"])
    ilc_bundle = build_lineage_heatmap_bundle(adata, "ILCs", BROAD_TO_SUBTYPES["ILCs"])
    myeloid_bundle = build_lineage_heatmap_bundle(adata, "Myeloid", BROAD_TO_SUBTYPES["Myeloid"])

    bundle_by_name = {
        "overall": overall_bundle,
        "T_lineage": t_bundle,
        "B_lineage": b_bundle,
        "ILCs_lineage": ilc_bundle,
        "Myeloid_lineage": myeloid_bundle,
    }

    for name, bundle in bundle_by_name.items():
        bundle["stats_df"].to_csv(
            os.path.join(HEATMAP_OUTDIR, f"{name}_stats.csv"),
            index=False,
        )

    sub_items_raw = [
        {"title": "", "effect_mat": t_bundle["effect_raw"], "star_mat": t_bundle["star_raw"]},
        {"title": "", "effect_mat": myeloid_bundle["effect_raw"], "star_mat": myeloid_bundle["star_raw"]},
        {"title": "", "effect_mat": ilc_bundle["effect_raw"], "star_mat": ilc_bundle["star_raw"]},
        {"title": "", "effect_mat": b_bundle["effect_raw"], "star_mat": b_bundle["star_raw"]},
    ]

    sub_items_fdr = [
        {"title": "", "effect_mat": t_bundle["effect_fdr"], "star_mat": t_bundle["star_fdr"]},
        {"title": "", "effect_mat": myeloid_bundle["effect_fdr"], "star_mat": myeloid_bundle["star_fdr"]},
        {"title": "", "effect_mat": ilc_bundle["effect_fdr"], "star_mat": ilc_bundle["star_fdr"]},
        {"title": "", "effect_mat": b_bundle["effect_fdr"], "star_mat": b_bundle["star_fdr"]},
    ]

    plot_custom_A_BC_DE_layout(
        overall_item={
            "title": "",
            "effect_mat": overall_bundle["effect_raw"],
            "star_mat": overall_bundle["star_raw"],
        },
        sub_items=sub_items_raw,
        savepath=FIGURE3_PDF,
        output_png=FIGURE3_PNG,
        page_title="Figure 3",
    )

    plot_custom_A_BC_DE_layout(
        overall_item={
            "title": "",
            "effect_mat": overall_bundle["effect_fdr"],
            "star_mat": overall_bundle["star_fdr"],
        },
        sub_items=sub_items_fdr,
        savepath=SUPPLEMENTARY_FIGURE3_PDF,
        output_png=SUPPLEMENTARY_FIGURE3_PNG,
        page_title="Supplementary Figure 3",
    )

    return bundle_by_name


# =============================================================================
# 5. 100% stacked barplot by disease
# =============================================================================

def plot_stacked_cell_proportion_barplot(
    adata,
    outdir=STACKED_BAR_OUTDIR,
    disease_order=DISEASE_ORDER_STACKED_BAR,
    cell_order=STACKED_CELL_ORDER,
    color_map=STACKED_COLOR_MAP,
):
    """Plot a 100% stacked barplot of cell-subtype proportions by disease."""
    validate_obs_columns(adata, [DISEASE_COL, CELLTYPE_COL])

    output_csv_path = os.path.join(outdir, "Cell_Type_Counts_by_Disease.csv")
    output_pdf_path = os.path.join(
        outdir,
        "Cell_Type_Proportion_StackedBar_Grouped_NoFrame_OneLegendColumn.pdf",
    )
    output_png_path = os.path.join(
        outdir,
        "Cell_Type_Proportion_StackedBar_Grouped_NoFrame_OneLegendColumn.png",
    )

    meta_df = adata.obs[[DISEASE_COL, CELLTYPE_COL]].copy()
    meta_df[DISEASE_COL] = meta_df[DISEASE_COL].astype(str).str.strip()
    meta_df[CELLTYPE_COL] = meta_df[CELLTYPE_COL].astype(str).str.strip()

    meta_df = meta_df[
        (meta_df[DISEASE_COL] != "")
        & (meta_df[DISEASE_COL] != "nan")
        & (meta_df[CELLTYPE_COL] != "")
        & (meta_df[CELLTYPE_COL] != "nan")
    ].copy()

    meta_df = standardize_metadata_from_dataframe(meta_df)

    print("Detected disease groups:")
    print(sorted(meta_df[DISEASE_COL].unique()))

    missing_diseases = [
        disease for disease in disease_order
        if disease not in meta_df[DISEASE_COL].unique()
    ]

    if missing_diseases:
        print(f"Warning: these disease groups are not present: {missing_diseases}")

    meta_plot = meta_df[meta_df[DISEASE_COL].isin(disease_order)].copy()

    unmatched_celltypes = sorted(set(meta_plot[CELLTYPE_COL].unique()) - set(cell_order))
    if unmatched_celltypes:
        print("The following cell subtypes are not in the plotting order and will be removed:")
        print(unmatched_celltypes)

    meta_plot = meta_plot[meta_plot[CELLTYPE_COL].isin(cell_order)].copy()

    if meta_plot.empty:
        raise ValueError("No cells remain after disease and cell-subtype filtering.")

    cell_counts_table = (
        meta_df
        .groupby([DISEASE_COL, CELLTYPE_COL], observed=True)
        .size()
        .reset_index(name="Count")
    )
    cell_counts_table.to_csv(output_csv_path, index=False)

    plot_counts = (
        meta_plot
        .groupby([DISEASE_COL, CELLTYPE_COL], observed=True)
        .size()
        .reset_index(name="Count")
    )

    full_index = pd.MultiIndex.from_product(
        [disease_order, cell_order],
        names=[DISEASE_COL, CELLTYPE_COL],
    )

    plot_data = (
        plot_counts
        .set_index([DISEASE_COL, CELLTYPE_COL])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    plot_data[DISEASE_COL] = pd.Categorical(
        plot_data[DISEASE_COL],
        categories=disease_order,
        ordered=True,
    )
    plot_data[CELLTYPE_COL] = pd.Categorical(
        plot_data[CELLTYPE_COL],
        categories=cell_order,
        ordered=True,
    )

    plot_data["Total"] = (
        plot_data
        .groupby(DISEASE_COL, observed=True)["Count"]
        .transform("sum")
    )
    plot_data["Proportion"] = np.where(
        plot_data["Total"] > 0,
        plot_data["Count"] / plot_data["Total"],
        0,
    )

    print("Total number of cells included for each disease group:")
    print(
        plot_data[[DISEASE_COL, "Total"]]
        .drop_duplicates()
        .sort_values(DISEASE_COL)
    )

    plot_matrix = (
        plot_data
        .pivot(index=DISEASE_COL, columns=CELLTYPE_COL, values="Proportion")
        .reindex(index=disease_order, columns=cell_order)
        .fillna(0)
    )

    legend_labels = {cell_type: cell_type for cell_type in cell_order}
    legend_labels["γδT"] = "gdT"

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    x_positions = np.arange(len(disease_order))
    bottom = np.zeros(len(disease_order))

    for cell_type in cell_order:
        values = plot_matrix[cell_type].values

        ax.bar(
            x_positions,
            values,
            bottom=bottom,
            width=0.7,
            color=color_map[cell_type],
            edgecolor="none",
            label=legend_labels[cell_type],
        )

        bottom += values

    ax.set_xticks(x_positions)
    totals_by_disease = (
        plot_data[[DISEASE_COL, "Total"]]
        .drop_duplicates()
        .set_index(DISEASE_COL)["Total"]
    )
    ax.set_xticklabels(
        [
            f"{disease}\n(n={int(totals_by_disease[disease]):,} cells)"
            for disease in disease_order
        ],
        rotation=45,
        ha="right",
        va="top",
        fontweight="bold",
        color="black",
    )

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Disease Group", fontweight="bold", color="black")
    ax.set_ylabel("Cell Type Proportion (%)", fontweight="bold", color="black")
    ax.tick_params(axis="y", colors="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    legend = ax.legend(
        title="Cell Type",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        ncol=1,
        fontsize=8.2,
        title_fontsize=11,
        handlelength=1.0,
        handleheight=0.8,
        borderaxespad=0.0,
        labelspacing=0.28,
    )
    legend.get_title().set_fontweight("bold")

    plt.subplots_adjust(left=0.12, right=0.74, bottom=0.18, top=0.95)

    fig.savefig(output_pdf_path, bbox_inches="tight", transparent=True)
    fig.savefig(output_png_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.show()

    print(f"Cell-count table saved to: {output_csv_path}")
    print(f"PDF saved to: {output_pdf_path}")
    print(f"PNG saved to: {output_png_path}")

    return plot_data, cell_counts_table


def standardize_metadata_from_dataframe(df):
    """Standardize disease and cell-subtype labels in a metadata DataFrame."""
    df = df.copy()

    if DISEASE_COL in df.columns:
        df[DISEASE_COL] = (
            df[DISEASE_COL]
            .astype(str)
            .str.strip()
            .replace({
                "MCVAD": "MCI/AD",
                "MCI_AD": "MCI/AD",
                "MCI-AD": "MCI/AD",
                "CAR-T/DIPG": "CAR-T-DIPG",
            })
        )

    if CELLTYPE_COL in df.columns:
        df[CELLTYPE_COL] = (
            df[CELLTYPE_COL]
            .astype(str)
            .str.strip()
            .replace({
                "CD4 Naïve": "CD4 Naive",
                "CD8 Naïve": "CD8 Naive",
                "maMac": "mdMac",
            })
        )

    return df


# =============================================================================
# 6. Sample-level dataset-stratified boxplots
# =============================================================================

def build_sample_level_proportion_table(
    adata,
    sample_col=SAMPLE_COL,
    disease_col=DISEASE_COL,
    dataset_col=DATASET_COL,
    celltype_col=CELLTYPE_COL,
):
    """Calculate sample-level cell-subtype proportions using total sample cell count as denominator."""
    validate_obs_columns(adata, [sample_col, disease_col, dataset_col, celltype_col])

    obs_df = adata.obs[[sample_col, disease_col, dataset_col, celltype_col]].copy()
    obs_df = standardize_metadata_from_dataframe(obs_df)

    for col in [sample_col, disease_col, dataset_col, celltype_col]:
        obs_df[col] = obs_df[col].astype(str).str.strip()

    obs_df = obs_df[
        (obs_df[sample_col] != "")
        & (obs_df[disease_col] != "")
        & (obs_df[dataset_col] != "")
        & (obs_df[celltype_col] != "")
    ].copy()

    obs_df["sample_uid"] = (
        obs_df[dataset_col].astype(str)
        + "|"
        + obs_df[sample_col].astype(str)
    )

    sample_meta = (
        obs_df[["sample_uid", sample_col, disease_col, dataset_col]]
        .drop_duplicates()
        .copy()
    )

    duplicated_samples = sample_meta["sample_uid"].duplicated().sum()
    if duplicated_samples > 0:
        raise ValueError("Duplicated sample_uid values were detected.")

    sample_totals = (
        obs_df
        .groupby("sample_uid", observed=True)
        .size()
        .rename("sample_total")
        .reset_index()
    )

    count_df = (
        obs_df
        .groupby(["sample_uid", celltype_col], observed=True)
        .size()
        .rename("cell_count")
        .reset_index()
    )

    full_index = pd.MultiIndex.from_product(
        [sample_meta["sample_uid"].unique(), sorted(obs_df[celltype_col].unique())],
        names=["sample_uid", celltype_col],
    )

    count_df = (
        count_df
        .set_index(["sample_uid", celltype_col])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    prop_df = (
        count_df
        .merge(sample_meta, on="sample_uid", how="left")
        .merge(sample_totals, on="sample_uid", how="left")
    )

    prop_df["proportion"] = prop_df["cell_count"] / prop_df["sample_total"] * 100.0

    return prop_df


def get_multidataset_diseases(prop_df, min_datasets=2):
    """Return diseases represented by at least min_datasets datasets."""
    dataset_counts = (
        prop_df[[DISEASE_COL, DATASET_COL]]
        .drop_duplicates()
        .groupby(DISEASE_COL, observed=True)[DATASET_COL]
        .nunique()
        .sort_values(ascending=False)
    )

    eligible_diseases = dataset_counts[
        (dataset_counts >= min_datasets)
        & (dataset_counts.index != REF_GROUP)
    ].index.tolist()

    return eligible_diseases, dataset_counts


def compare_one_disease_celltype_vs_global_healthy(
    prop_df,
    disease,
    cell_type,
    dataset_order=None,
    min_samples_per_group=2,
):
    """Compare each dataset within one disease against pooled younger controls."""
    df = prop_df[prop_df[CELLTYPE_COL].astype(str) == cell_type].copy()

    reference_df = df[df[DISEASE_COL].astype(str) == REF_GROUP].copy()
    disease_df = df[df[DISEASE_COL].astype(str) == disease].copy()

    if reference_df.empty:
        raise ValueError(f"No {REF_GROUP} samples were found for {cell_type}.")

    if disease_df.empty:
        raise ValueError(f"No {disease} samples were found for {cell_type}.")

    if dataset_order is None:
        dataset_order = sorted(disease_df[DATASET_COL].astype(str).unique().tolist())
    else:
        dataset_order = [
            dataset for dataset in dataset_order
            if dataset in disease_df[DATASET_COL].astype(str).unique().tolist()
        ]

    reference_values = reference_df["proportion"].values.astype(float)
    rows = []

    for dataset in dataset_order:
        sub_df = disease_df[disease_df[DATASET_COL].astype(str) == str(dataset)].copy()
        disease_values = sub_df["proportion"].values.astype(float)

        if len(reference_values) >= min_samples_per_group and len(disease_values) >= min_samples_per_group:
            _, p_value = ranksums(disease_values, reference_values)
        else:
            p_value = np.nan

        rows.append({
            "disease": disease,
            "cell_type": cell_type,
            "dataset": dataset,
            "reference_group": REF_GROUP,
            "n_reference_samples": len(reference_values),
            "n_disease_samples": len(disease_values),
            "reference_median": np.median(reference_values) if len(reference_values) > 0 else np.nan,
            "disease_dataset_median": np.median(disease_values) if len(disease_values) > 0 else np.nan,
            "median_diff_vs_reference": (
                np.median(disease_values) - np.median(reference_values)
                if len(reference_values) > 0 and len(disease_values) > 0
                else np.nan
            ),
            "p_value_raw": p_value,
            "p_label": format_raw_p_value_or_stars(p_value),
        })

    return pd.DataFrame(rows), dataset_order


def prepare_boxplot_data(
    prop_df,
    disease,
    cell_type,
    dataset_order=None,
):
    """Prepare plot data and dataset-wise statistics for one disease-cell-type panel."""
    df = prop_df[prop_df[CELLTYPE_COL].astype(str) == cell_type].copy()

    reference_df = df[df[DISEASE_COL].astype(str) == REF_GROUP].copy()
    disease_df = df[df[DISEASE_COL].astype(str) == disease].copy()

    stats_df, dataset_order = compare_one_disease_celltype_vs_global_healthy(
        prop_df,
        disease=disease,
        cell_type=cell_type,
        dataset_order=dataset_order,
    )

    if len(dataset_order) == 0:
        raise ValueError(f"No requested datasets were found for disease={disease}, cell_type={cell_type}.")

    reference_df["plot_group"] = REF_GROUP

    disease_df = disease_df[disease_df[DATASET_COL].astype(str).isin(dataset_order)].copy()
    disease_df["plot_group"] = disease_df[DISEASE_COL].astype(str) + "\n" + disease_df[DATASET_COL].astype(str)

    plot_order = [REF_GROUP] + [f"{disease}\n{dataset}" for dataset in dataset_order]

    plot_df = pd.concat([reference_df, disease_df], axis=0, ignore_index=True)
    plot_df = plot_df[plot_df["plot_group"].isin(plot_order)].copy()
    plot_df["plot_group"] = pd.Categorical(plot_df["plot_group"], categories=plot_order, ordered=True)

    return plot_df, stats_df, plot_order, dataset_order


def add_pvalue_bracket_axes_fraction(
    ax,
    x1,
    x2,
    y_frac,
    h_frac,
    text,
    fontsize=5.8,
    linewidth=0.65,
):
    """Add a P-value bracket using data coordinates for x and axis-fraction coordinates for y."""
    transform = blended_transform_factory(ax.transData, ax.transAxes)

    ax.plot(
        [x1, x1, x2, x2],
        [y_frac, y_frac + h_frac, y_frac + h_frac, y_frac],
        lw=linewidth,
        color="black",
        transform=transform,
        clip_on=False,
    )

    ax.text(
        (x1 + x2) / 2,
        y_frac + h_frac + 0.006,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="black",
        transform=transform,
        clip_on=False,
    )


def set_compact_y_axis_for_boxplot(ax, plot_df, n_comparisons):
    """Set a high y-axis range so stacked P-value brackets have enough vertical room."""
    y_values = plot_df["proportion"].values.astype(float)
    y_values = y_values[np.isfinite(y_values)]

    if len(y_values) == 0:
        y_top = 1.0
    else:
        y_max = float(np.max(y_values))
        if y_max <= 0:
            y_top = 1.0
        elif n_comparisons >= 5:
            y_top = max(y_max * 2.85, y_max + 0.75)
        elif n_comparisons >= 3:
            y_top = max(y_max * 2.45, y_max + 0.55)
        else:
            y_top = max(y_max * 2.05, y_max + 0.35)

    ax.set_ylim(0, y_top)


def draw_dataset_boxplot_panel(
    ax,
    prop_df,
    panel_label,
    disease,
    cell_type,
    display_cell_type=None,
    dataset_order=None,
):
    """Draw one Figure 3G-O boxplot panel."""
    if display_cell_type is None:
        display_cell_type = cell_type

    plot_df, stats_df, plot_order, dataset_order = prepare_boxplot_data(
        prop_df,
        disease=disease,
        cell_type=cell_type,
        dataset_order=dataset_order,
    )

    palette = {REF_GROUP: "#F2F2F2"}
    for dataset in dataset_order:
        palette[f"{disease}\n{dataset}"] = "#E8B9B4"

    sns.boxplot(
        data=plot_df,
        x="plot_group",
        y="proportion",
        order=plot_order,
        palette=palette,
        width=0.55,
        linewidth=0.8,
        fliersize=0,
        ax=ax,
        boxprops={"edgecolor": "#4D4D4D", "alpha": 0.85},
        whiskerprops={"color": "#4D4D4D", "linewidth": 0.8},
        capprops={"color": "#4D4D4D", "linewidth": 0.8},
        medianprops={"color": "#4D4D4D", "linewidth": 1.0},
    )

    sns.stripplot(
        data=plot_df,
        x="plot_group",
        y="proportion",
        order=plot_order,
        color="#8C2D23",
        edgecolor="black",
        linewidth=0.25,
        size=2.4,
        jitter=0.22,
        alpha=0.85,
        ax=ax,
        zorder=10,
    )

    x_labels = []
    for group in plot_order:
        if group == REF_GROUP:
            x_labels.append(REF_GROUP)
        else:
            disease_name, dataset_name = group.split("\n")
            x_labels.append(f"{disease_name}\n{dataset_name}")

    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=5.7)

    n_comparisons = len(dataset_order)
    set_compact_y_axis_for_boxplot(ax, plot_df, n_comparisons=n_comparisons)

    if n_comparisons >= 5:
        y_fracs = np.linspace(0.46, 0.92, n_comparisons)
        bracket_h_frac = 0.014
        p_label_fontsize = 5.2
    elif n_comparisons >= 3:
        y_fracs = np.linspace(0.54, 0.90, n_comparisons)
        bracket_h_frac = 0.014
        p_label_fontsize = 5.3
    else:
        y_fracs = np.linspace(0.62, 0.86, n_comparisons)
        bracket_h_frac = 0.012
        p_label_fontsize = 5.5

    for index, dataset in enumerate(dataset_order, start=1):
        row = stats_df[stats_df["dataset"].astype(str) == str(dataset)]
        if row.empty:
            continue

        add_pvalue_bracket_axes_fraction(
            ax=ax,
            x1=0,
            x2=index,
            y_frac=y_fracs[index - 1],
            h_frac=bracket_h_frac,
            text=row.iloc[0]["p_label"],
            fontsize=p_label_fontsize,
            linewidth=0.65,
        )

    ax.set_title(f"{display_cell_type}\n({disease})", fontsize=7.2, pad=4, fontweight="normal")
    ax.set_xlabel("")
    ax.set_ylabel("Fraction (%)", fontsize=6.5)
    ax.tick_params(axis="y", labelsize=6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.grid(True, linestyle="--", linewidth=0.55, alpha=0.65, color="#A9A9A9")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")

    ax.text(
        -0.18,
        1.12,
        panel_label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color="#003366",
        ha="left",
        va="top",
    )

    return stats_df


def plot_disease_celltype_dataset_boxplot(
    prop_df,
    disease,
    cell_type,
    dataset_order=None,
    save=True,
    outdir=DATASET_BOXPLOT_OUTDIR,
):
    """Plot one dataset-stratified sample-level boxplot for a selected disease and cell type."""
    plot_df, stats_df, plot_order, dataset_order = prepare_boxplot_data(
        prop_df,
        disease=disease,
        cell_type=cell_type,
        dataset_order=dataset_order,
    )

    figsize = (max(5.0, 1.25 * len(plot_order)), 4.8)
    fig, ax = plt.subplots(figsize=figsize)

    draw_dataset_boxplot_panel(
        ax=ax,
        prop_df=prop_df,
        panel_label="",
        disease=disease,
        cell_type=cell_type,
        dataset_order=dataset_order,
    )

    ax.set_title(f"{cell_type} proportion in {disease}", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    if save:
        safe_disease = clean_filename(disease.replace(" ", "_"))
        safe_cell_type = clean_filename(cell_type.replace(" ", "_").replace("γδ", "gd"))

        output_pdf = os.path.join(outdir, f"{safe_disease}_{safe_cell_type}_sample_proportion_rawP.pdf")
        output_png = os.path.join(outdir, f"{safe_disease}_{safe_cell_type}_sample_proportion_rawP.png")

        fig.savefig(output_pdf, bbox_inches="tight", transparent=True)
        fig.savefig(output_png, dpi=300, bbox_inches="tight", transparent=True)

        print(f"Saved: {output_pdf}")
        print(f"Saved: {output_png}")

    return fig, ax, stats_df


def plot_figure3G_O_combined(
    prop_df,
    target_panels=FIGURE3GO_TARGET_PANELS,
    dataset_order_by_disease=DATASET_ORDER_BY_DISEASE,
    output_pdf=FIGURE3GO_COMBINED_PDF,
    output_png=FIGURE3GO_COMBINED_PNG,
):
    """Create all selected dataset-stratified boxplots in one combined PDF and PNG."""
    fig = plt.figure(figsize=(16, 5.4), dpi=300)

    grid = fig.add_gridspec(
        nrows=2,
        ncols=12,
        left=0.035,
        right=0.995,
        top=0.93,
        bottom=0.10,
        wspace=1.05,
        hspace=0.92,
    )

    axes_positions = {
        "G": grid[0, 0:2],
        "H": grid[0, 2:4],
        "I": grid[0, 4:6],
        "J": grid[0, 6:8],
        "K": grid[0, 8:10],
        "L": grid[0, 10:12],
        "M": grid[1, 3:5],
        "N": grid[1, 5:7],
        "O": grid[1, 7:9],
    }

    all_stats = []

    for panel_cfg in target_panels:
        panel_label = panel_cfg["panel"]
        ax = fig.add_subplot(axes_positions[panel_label])

        disease = panel_cfg["disease"]
        cell_type = panel_cfg["cell_type"]
        display_cell_type = panel_cfg.get("display_cell_type", cell_type)
        dataset_order = dataset_order_by_disease.get(disease, None)

        stats_df = draw_dataset_boxplot_panel(
            ax=ax,
            prop_df=prop_df,
            panel_label=panel_label,
            disease=disease,
            cell_type=cell_type,
            display_cell_type=display_cell_type,
            dataset_order=dataset_order,
        )

        stats_df["panel"] = panel_label
        all_stats.append(stats_df)

    all_stats_df = pd.concat(all_stats, axis=0, ignore_index=True)
    stats_output = os.path.join(DATASET_BOXPLOT_OUTDIR, "Figure3G_O_dataset_vs_global_younger_control_stats.csv")
    all_stats_df.to_csv(stats_output, index=False)

    fig.savefig(output_pdf, bbox_inches="tight", transparent=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", transparent=True)
    plt.show()

    print(f"Saved combined PDF: {output_pdf}")
    print(f"Saved combined PNG: {output_png}")
    print(f"Saved statistics table: {stats_output}")

    return fig, all_stats_df


def run_dataset_boxplot_analysis(adata):
    """Calculate sample-level proportions and save all target boxplots in one combined figure."""
    sample_prop_df = build_sample_level_proportion_table(adata)

    eligible_diseases, disease_dataset_counts = get_multidataset_diseases(sample_prop_df)
    print("Number of datasets per disease:")
    print(disease_dataset_counts)
    print("\nDiseases with at least two datasets:")
    print(eligible_diseases)

    sample_prop_df.to_csv(
        os.path.join(DATASET_BOXPLOT_OUTDIR, "sample_level_celltype_proportions.csv"),
        index=False,
    )

    fig, figure3go_stats_df = plot_figure3G_O_combined(sample_prop_df)

    return sample_prop_df, figure3go_stats_df



# =============================================================================
# 7. ROE heatmap data preparation
# =============================================================================

def clean_axis(ax):
    """Remove axis background, frame, spines, ticks, and grid."""
    ax.set_facecolor("none")
    ax.patch.set_alpha(0)
    ax.patch.set_visible(False)
    ax.set_frame_on(False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def clean_obs_metadata_for_roe(
    adata,
    sample_col=SAMPLE_COL,
    group_col=DISEASE_COL,
    celltype_col=CELLTYPE_COL,
    broad_type_col=BROAD_TYPE_COL,
    age_col=AGE_COL,
):
    """Extract and clean the AnnData metadata required for ROE analysis."""
    required_cols = [sample_col, group_col, celltype_col, broad_type_col, age_col]
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in adata.obs: {missing_cols}")

    obs_df = adata.obs[required_cols].copy()

    for col in [sample_col, group_col, celltype_col, broad_type_col]:
        obs_df[col] = obs_df[col].astype(str).str.strip()

    obs_df[group_col] = obs_df[group_col].replace(DISEASE_RENAME_MAP)
    obs_df[celltype_col] = obs_df[celltype_col].replace(CELLTYPE_RENAME_MAP)
    obs_df[age_col] = pd.to_numeric(obs_df[age_col], errors="coerce")

    obs_df = obs_df[
        (obs_df[sample_col] != "")
        & (obs_df[sample_col] != "nan")
        & (obs_df[group_col] != "")
        & (obs_df[group_col] != "nan")
        & (obs_df[celltype_col] != "")
        & (obs_df[celltype_col] != "nan")
        & (obs_df[broad_type_col] != "")
        & (obs_df[broad_type_col] != "nan")
    ].copy()

    return obs_df


def calculate_roe_matrix(obs_df, sample_col=SAMPLE_COL, celltype_col=CELLTYPE_COL, pseudocount=0.1):
    """Calculate log2 ROE from the cell-subtype-by-sample contingency table."""
    count_table = pd.crosstab(obs_df[celltype_col], obs_df[sample_col])

    row_sums = count_table.sum(axis=1).values.reshape(-1, 1)
    col_sums = count_table.sum(axis=0).values.reshape(1, -1)
    total_sum = count_table.values.sum()

    expected = (row_sums @ col_sums) / total_sum
    return np.log2((count_table / expected) + pseudocount)


def cluster_groups(roe_df, sample_meta, group_col=DISEASE_COL):
    """Cluster disease groups based on the mean ROE profile."""
    group_means = roe_df.T.groupby(sample_meta[group_col]).mean()

    if group_means.shape[0] <= 1:
        return group_means.index.tolist(), None, 1.0

    group_linkage = linkage(group_means, method="ward")
    group_dendro = dendrogram(group_linkage, no_plot=True)
    group_order = group_means.index[group_dendro["leaves"]].tolist()
    max_height = float(np.array(group_dendro["dcoord"]).max())

    return group_order, group_dendro, max_height


def summarize_group_layout(sample_meta, group_order, group_col=DISEASE_COL):
    """Calculate group widths and group-center positions after sample ordering."""
    group_widths = [int(np.sum(sample_meta[group_col].astype(str) == group)) for group in group_order]

    group_centers = []
    x_pos = 0

    for width in group_widths:
        group_centers.append(x_pos + width / 2)
        x_pos += width

    return group_widths, group_centers


def calculate_roe_heatmap_data(
    adata,
    sample_col=SAMPLE_COL,
    group_col=DISEASE_COL,
    celltype_col=CELLTYPE_COL,
    broad_type_col=BROAD_TYPE_COL,
    age_col=AGE_COL,
):
    """Prepare all matrices and metadata required by the ROE heatmap function."""
    obs_df = clean_obs_metadata_for_roe(
        adata=adata,
        sample_col=sample_col,
        group_col=group_col,
        celltype_col=celltype_col,
        broad_type_col=broad_type_col,
        age_col=age_col,
    )

    roe_df = calculate_roe_matrix(obs_df, sample_col=sample_col, celltype_col=celltype_col)

    sample_meta = (
        obs_df[[sample_col, group_col, age_col]]
        .drop_duplicates(subset=[sample_col])
        .set_index(sample_col)
        .loc[roe_df.columns]
    )

    group_order, group_dendro, max_height = cluster_groups(
        roe_df=roe_df,
        sample_meta=sample_meta,
        group_col=group_col,
    )

    sample_meta[group_col] = pd.Categorical(sample_meta[group_col], categories=group_order, ordered=True)
    sample_meta = sample_meta.sort_values(group_col)
    roe_df = roe_df.loc[:, sample_meta.index]

    group_widths, group_centers = summarize_group_layout(
        sample_meta=sample_meta,
        group_order=group_order,
        group_col=group_col,
    )

    return {
        "roe_df": roe_df,
        "sample_meta": sample_meta,
        "group_order": group_order,
        "group_widths": group_widths,
        "group_centers": group_centers,
        "group_dendro": group_dendro,
        "max_height": max_height,
        "age_values": sample_meta[age_col].copy(),
        "age_min": float(obs_df[age_col].min()),
        "age_max": float(obs_df[age_col].max()),
        "sample_col": sample_col,
        "group_col": group_col,
        "celltype_col": celltype_col,
        "broad_type_col": broad_type_col,
        "age_col": age_col,
    }


# =============================================================================
# 8. ROE heatmap plotting helpers
# =============================================================================

def apply_manual_group_order(
    roe_df,
    sample_meta,
    group_col,
    disease_order=None,
    add_unlisted_groups=True,
):
    """Apply a manual disease order, or keep the current order in sample_meta."""
    existing_groups = sample_meta[group_col].astype(str).unique().tolist()

    if disease_order is None:
        group_order = existing_groups
    else:
        ordered_groups = [group for group in disease_order if group in existing_groups]
        missing_groups = [group for group in disease_order if group not in existing_groups]

        if missing_groups:
            print("Warning: groups in disease_order not found:", missing_groups)

        if add_unlisted_groups:
            group_order = ordered_groups + [group for group in existing_groups if group not in ordered_groups]
        else:
            group_order = ordered_groups

    sample_order = []
    for group in group_order:
        sample_order.extend(sample_meta.index[sample_meta[group_col].astype(str) == group].tolist())

    sample_meta = sample_meta.loc[sample_order].copy()
    roe_df = roe_df.loc[:, sample_order]
    group_widths, group_centers = summarize_group_layout(sample_meta, group_order, group_col)

    return roe_df, sample_meta, group_order, group_widths, group_centers


def reorder_age_values(age_values, old_order, new_order):
    """Reorder age values to match the final sample order."""
    if isinstance(age_values, pd.Series):
        return age_values.loc[new_order].to_frame().T

    arr = np.asarray(age_values)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    old_position = {sample: idx for idx, sample in enumerate(old_order)}
    reorder_idx = [old_position[sample] for sample in new_order]

    if arr.shape[1] == len(old_order):
        return arr[:, reorder_idx]

    if arr.shape[0] == len(old_order):
        return arr[reorder_idx, :].T

    raise ValueError("Cannot align age values with the requested sample order.")


def get_celltype_order(
    roe_df,
    sample_order,
    cell_order=None,
    add_unlisted_celltypes=True,
    row_cluster=True,
    row_cluster_method="ward",
):
    """Apply manual cell-type ordering and optionally cluster rows."""
    existing_celltypes = roe_df.index.unique().tolist()

    if cell_order is None:
        base_celltypes = existing_celltypes
    else:
        ordered_celltypes = [cell_type for cell_type in cell_order if cell_type in existing_celltypes]
        missing_celltypes = [cell_type for cell_type in cell_order if cell_type not in existing_celltypes]

        if missing_celltypes:
            print("Warning: cell types in cell_order not found:", missing_celltypes)

        if add_unlisted_celltypes:
            base_celltypes = ordered_celltypes + [
                cell_type for cell_type in existing_celltypes
                if cell_type not in ordered_celltypes
            ]
        else:
            base_celltypes = ordered_celltypes

    row_data = roe_df.loc[base_celltypes, sample_order]

    if row_cluster and row_data.shape[0] > 1:
        row_linkage = linkage(row_data.values, method=row_cluster_method)
        row_leaves = dendrogram(row_linkage, no_plot=True)["leaves"]
        row_order = row_data.index[row_leaves[::-1]].tolist()
    else:
        row_linkage = None
        row_order = row_data.index.tolist()

    return row_order, row_linkage


def build_roe_colormap(roe_colors=None):
    """Create the custom diverging ROE colormap."""
    if roe_colors is None:
        roe_colors = DEFAULT_ROE_COLORS

    return LinearSegmentedColormap.from_list(
        "roe_cmap",
        list(zip([0.0, 0.35, 0.5, 0.65, 1.0], roe_colors)),
    )


def draw_row_dendrogram(ax, row_linkage, n_rows, color="#333333"):
    """Draw the left-side row dendrogram."""
    clean_axis(ax)

    if row_linkage is None or n_rows <= 1:
        ax.axis("off")
        return

    dendro_data = dendrogram(
        row_linkage,
        orientation="left",
        ax=ax,
        color_threshold=0,
        above_threshold_color=color,
        no_labels=True,
    )

    max_distance = np.max(dendro_data["dcoord"])
    ax.set_xlim(max_distance * 1.05, 0)
    ax.set_ylim(0, 10 * n_rows)

    for collection in ax.collections:
        collection.set_linewidth(0.55)
        collection.set_color(color)
        collection.set_zorder(2)

    ax.axis("off")


def draw_group_separators(ax, sample_meta, group_order, group_col, color="#333333"):
    """Draw optional vertical separators between disease groups."""
    current_position = 0

    for group in group_order[:-1]:
        current_position += np.sum(sample_meta[group_col].astype(str) == group)
        ax.axvline(current_position, color=color, linewidth=0.4, alpha=0.5, zorder=3)


def draw_disease_annotation_bar(ax, sample_meta, group_order, group_widths, group_col, n_cols, group_palette="Set2"):
    """Draw the disease annotation bar and rotated disease labels."""
    clean_axis(ax)

    group_colors = sns.color_palette(group_palette, len(group_order))
    color_dict = dict(zip(group_order, group_colors))

    x_pos = 0
    for group, width in zip(group_order, group_widths):
        ax.barh(
            0,
            width,
            left=x_pos,
            color=color_dict[group],
            height=0.8,
            edgecolor="none",
            linewidth=0,
            zorder=2,
        )
        ax.text(
            x_pos + width / 2,
            -0.8,
            group,
            ha="right",
            va="top",
            rotation=45,
            fontweight="bold",
            fontsize=14,
            zorder=3,
        )
        x_pos += width

    ax.set_xlim(0, n_cols)
    ax.set_ylim(-1.5, 1.0)
    ax.axis("off")


def draw_group_dendrogram(ax, group_dendro, group_order, group_centers, n_cols, max_height, color="#333333"):
    """Draw the bottom disease-group dendrogram."""
    clean_axis(ax)

    if group_dendro is None or len(group_order) <= 1:
        ax.axis("off")
        return

    icoord = np.array(group_dendro["icoord"])
    dcoord = np.array(group_dendro["dcoord"])
    default_leaf_position = np.array([idx * 10 + 5 for idx in range(len(group_order))])

    for xs, ys in zip(icoord, dcoord):
        mapped_x = np.interp(xs, default_leaf_position, group_centers)
        ax.plot(mapped_x, max_height - ys, color=color, lw=0.55, zorder=2)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(-max_height * 0.1, max_height * 1.1)
    ax.axis("off")


def add_vertical_colorbar(fig, cmap, vmin, vmax, cbar_rect, title, ticks=None, title_pad=10):
    """Add a clean vertical colorbar to the figure."""
    cbar_ax = fig.add_axes(cbar_rect)
    clean_axis(cbar_ax)

    scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    colorbar = plt.colorbar(scalar_mappable, cax=cbar_ax, ticks=ticks)

    colorbar.ax.set_title(title, fontweight="bold", fontsize=12, pad=title_pad)
    colorbar.ax.tick_params(labelsize=10, length=0, width=0)
    colorbar.outline.set_visible(False)
    clean_axis(colorbar.ax)

    return colorbar


def save_roe_tables(
    data_dict,
    heatmap_data,
    sample_meta,
    group_order,
    group_widths,
    group_centers,
    row_order,
    outdir=ROE_OUTDIR,
    prefix=ROE_TABLE_PREFIX,
):
    """Save ROE matrices, plotted row order, sample metadata, and disease-group layout tables."""
    os.makedirs(outdir, exist_ok=True)

    raw_roe_path = os.path.join(outdir, f"{prefix}_ROE_raw_matrix.csv")
    plotted_roe_path = os.path.join(outdir, f"{prefix}_ROE_plotted_matrix.csv")
    sample_meta_path = os.path.join(outdir, f"{prefix}_ROE_sample_metadata_plotted_order.csv")
    group_layout_path = os.path.join(outdir, f"{prefix}_ROE_group_layout.csv")
    celltype_order_path = os.path.join(outdir, f"{prefix}_ROE_celltype_order.csv")

    data_dict["roe_df"].to_csv(raw_roe_path)
    heatmap_data.to_csv(plotted_roe_path)
    sample_meta.to_csv(sample_meta_path)

    group_layout_df = pd.DataFrame({
        "disease": group_order,
        "n_samples": group_widths,
        "group_center": group_centers,
    })
    group_layout_df.to_csv(group_layout_path, index=False)

    celltype_order_df = pd.DataFrame({
        "row_index": np.arange(1, len(row_order) + 1),
        "cell_type": row_order,
    })
    celltype_order_df.to_csv(celltype_order_path, index=False)

    print(f"Saved ROE raw matrix: {raw_roe_path}")
    print(f"Saved plotted ROE matrix: {plotted_roe_path}")
    print(f"Saved ROE sample metadata: {sample_meta_path}")
    print(f"Saved ROE group layout: {group_layout_path}")
    print(f"Saved ROE cell-type order: {celltype_order_path}")


def plot_roe_heatmap(
    data_dict,
    cell_size=0.35,
    vmax=10,
    roe_colors=None,
    age_cmap="YlGnBu",
    group_palette="Set2",
    disease_order=None,
    cell_order=None,
    add_unlisted_groups=True,
    add_unlisted_celltypes=True,
    row_cluster=True,
    row_cluster_method="ward",
    show_row_dendrogram=True,
    show_group_dendrogram_if_clustered=True,
    save_path=None,
    output_png=None,
    table_outdir=None,
    table_prefix=ROE_TABLE_PREFIX,
    dpi=300,
    cbar_x=0.80,
    grey_color="#333333",
    show_group_separators=False,
    transparent=True,
    pad_inches=0.02,
):
    """Draw a transparent ROE heatmap with row dendrogram, age bar, disease bar, and optional group dendrogram."""
    roe_df = data_dict["roe_df"].copy()
    sample_meta = data_dict["sample_meta"].copy()
    group_col = data_dict["group_col"]
    age_min = data_dict["age_min"]
    age_max = data_dict["age_max"]
    original_sample_order = sample_meta.index.tolist()

    if disease_order is None:
        base_group_order = list(data_dict["group_order"])
        group_dendro = data_dict["group_dendro"]
        max_height = data_dict["max_height"]
        show_group_dendro = show_group_dendrogram_if_clustered
    else:
        base_group_order = disease_order
        group_dendro = None
        max_height = 1.0
        show_group_dendro = False

    roe_df, sample_meta, group_order, group_widths, group_centers = apply_manual_group_order(
        roe_df=roe_df,
        sample_meta=sample_meta,
        group_col=group_col,
        disease_order=base_group_order,
        add_unlisted_groups=add_unlisted_groups,
    )

    age_values = reorder_age_values(
        age_values=data_dict["age_values"],
        old_order=original_sample_order,
        new_order=sample_meta.index.tolist(),
    )

    row_order, row_linkage = get_celltype_order(
        roe_df=roe_df,
        sample_order=sample_meta.index,
        cell_order=cell_order,
        add_unlisted_celltypes=add_unlisted_celltypes,
        row_cluster=row_cluster,
        row_cluster_method=row_cluster_method,
    )

    heatmap_data = roe_df.loc[row_order, sample_meta.index]
    n_rows, n_cols = heatmap_data.shape
    show_cell_dendro = show_row_dendrogram and row_linkage is not None and n_rows > 1

    fig_width = n_cols * cell_size + 0.6 + (1.0 if show_cell_dendro else 0.25) + 3.2
    fig_height = n_rows * cell_size + (5.0 if show_group_dendro else 3.4) + 0.35

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="none")
    fig.patch.set_alpha(0)
    fig.patch.set_visible(False)

    main_gs = GridSpec(
        4,
        4,
        height_ratios=[n_rows, 0.35, 0.55, 2.5 if show_group_dendro else 0.5],
        width_ratios=[0.04, 0.09 if show_cell_dendro else 0.01, 0.74, 0.13],
        hspace=0.10,
        wspace=0.01,
    )

    roe_cmap = build_roe_colormap(roe_colors)

    ax_tree = fig.add_subplot(main_gs[0, 1])
    draw_row_dendrogram(ax_tree, row_linkage if show_cell_dendro else None, n_rows, color=grey_color)

    ax_heatmap = fig.add_subplot(main_gs[0, 2])
    clean_axis(ax_heatmap)
    sns.heatmap(
        heatmap_data,
        ax=ax_heatmap,
        cmap=roe_cmap,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        cbar=False,
        xticklabels=False,
        yticklabels=True,
        linewidths=0,
        linecolor="none",
        square=False,
    )
    clean_axis(ax_heatmap)
    ax_heatmap.yaxis.tick_right()
    ax_heatmap.yaxis.set_label_position("right")
    ax_heatmap.set_ylim(n_rows, 0)
    ax_heatmap.set_xlabel("")
    ax_heatmap.set_ylabel("")
    ax_heatmap.tick_params(axis="both", length=0, width=0)
    plt.setp(
        ax_heatmap.get_yticklabels(),
        rotation=0,
        horizontalalignment="left",
        fontsize=12,
        fontweight="medium",
    )

    if show_group_separators:
        draw_group_separators(ax_heatmap, sample_meta, group_order, group_col, color=grey_color)

    ax_age = fig.add_subplot(main_gs[1, 2])
    clean_axis(ax_age)
    sns.heatmap(
        age_values,
        ax=ax_age,
        cmap=age_cmap,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        linewidths=0,
        linecolor="none",
    )
    clean_axis(ax_age)
    ax_age.tick_params(axis="both", length=0, width=0)

    if show_group_separators:
        draw_group_separators(ax_age, sample_meta, group_order, group_col, color=grey_color)

    ax_bar = fig.add_subplot(main_gs[2, 2])
    draw_disease_annotation_bar(
        ax=ax_bar,
        sample_meta=sample_meta,
        group_order=group_order,
        group_widths=group_widths,
        group_col=group_col,
        n_cols=n_cols,
        group_palette=group_palette,
    )

    ax_bottom_tree = fig.add_subplot(main_gs[3, 2])
    draw_group_dendrogram(
        ax=ax_bottom_tree,
        group_dendro=group_dendro if show_group_dendro else None,
        group_order=group_order,
        group_centers=group_centers,
        n_cols=n_cols,
        max_height=max_height,
        color=grey_color,
    )

    add_vertical_colorbar(
        fig=fig,
        cmap=roe_cmap,
        vmin=-vmax,
        vmax=vmax,
        cbar_rect=[cbar_x, 0.57, 0.013, 0.22],
        title="Log2ROE",
        ticks=np.linspace(-vmax, vmax, 5),
        title_pad=15,
    )

    add_vertical_colorbar(
        fig=fig,
        cmap=age_cmap,
        vmin=age_min,
        vmax=age_max,
        cbar_rect=[cbar_x, 0.34, 0.013, 0.16],
        title="Age",
        ticks=None,
        title_pad=10,
    )

    for ax in fig.axes:
        clean_axis(ax)

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=pad_inches,
            transparent=transparent,
            facecolor="none",
            edgecolor="none",
        )
        print(f"ROE heatmap PDF saved to: {save_path}")

    if output_png is not None:
        fig.savefig(
            output_png,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=pad_inches,
            transparent=transparent,
            facecolor="none",
            edgecolor="none",
        )
        print(f"ROE heatmap PNG saved to: {output_png}")

    if table_outdir is not None:
        save_roe_tables(
            data_dict=data_dict,
            heatmap_data=heatmap_data,
            sample_meta=sample_meta,
            group_order=group_order,
            group_widths=group_widths,
            group_centers=group_centers,
            row_order=row_order,
            outdir=table_outdir,
            prefix=table_prefix,
        )

    plt.show()
    return fig




# =============================================================================
# 9. Load AnnData object
# =============================================================================
# Run this cell first. All downstream cells assume that adata_csf has been loaded
# and that disease / cell-subtype labels have been standardized.

adata_csf = sc.read_h5ad(INPUT_H5AD)
adata_csf = standardize_metadata(adata_csf)

print(adata_csf)



# =============================================================================
# 10. Generate Supplementary Figure 1C PCA outlier overview
# =============================================================================
# This cell saves Supplementary Figure 1C as one combined PCA figure and creates remove_list_csf.

remove_list_csf, all_outliers, pca_plot_data = run_pca_outlier_detection_by_disease(
    adata_csf,
    output_pdf=PCA_OUTPUT_PDF,
    output_png=PCA_OUTPUT_PNG,
    output_outlier_csv=PCA_OUTLIER_TABLE,
)

print("Number of outlier samples to remove:", len(remove_list_csf))
print("remove_list_csf =", remove_list_csf)



# =============================================================================
# 11. Remove outlier samples
# =============================================================================
# Run this cell after checking remove_list_csf.
# adata_csf_clean is used for the following heatmaps and composition plots.

adata_csf_clean = adata_csf[
    ~adata_csf.obs[SAMPLE_COL].isin(remove_list_csf)
].copy()

print(adata_csf_clean)



# =============================================================================
# 12. Generate Supplementary Figure 2 CSF ROE heatmap
# =============================================================================
# This cell saves:
#   SupplementaryFigure2_ROE_Heatmap_Results/SupplementaryFigure2_CSF_ROE_heatmap.pdf
#   SupplementaryFigure2_ROE_Heatmap_Results/SupplementaryFigure2_CSF_ROE_heatmap.png
#   SupplementaryFigure2_ROE_Heatmap_Results/SupplementaryFigure2_CSF_ROE_raw_matrix.csv
#   SupplementaryFigure2_ROE_Heatmap_Results/SupplementaryFigure2_CSF_ROE_plotted_matrix.csv
#   SupplementaryFigure2_ROE_Heatmap_Results/SupplementaryFigure2_CSF_ROE_sample_metadata_plotted_order.csv
#   SupplementaryFigure2_ROE_Heatmap_Results/SupplementaryFigure2_CSF_ROE_group_layout.csv
#   SupplementaryFigure2_ROE_Heatmap_Results/SupplementaryFigure2_CSF_ROE_celltype_order.csv

data_dict_csf_roe = calculate_roe_heatmap_data(
    adata=adata_csf_clean,
    sample_col=SAMPLE_COL,
    group_col=DISEASE_COL,
    celltype_col=CELLTYPE_COL,
    broad_type_col=BROAD_TYPE_COL,
    age_col=AGE_COL,
)

fig_csf_roe = plot_roe_heatmap(
    data_dict=data_dict_csf_roe,
    cell_size=0.35,
    vmax=10,
    roe_colors=DEFAULT_ROE_COLORS,
    save_path=ROE_OUTPUT_PDF,
    output_png=ROE_OUTPUT_PNG,
    table_outdir=ROE_OUTDIR,
    table_prefix=ROE_TABLE_PREFIX,
    dpi=300,
    cbar_x=0.875,
)



# =============================================================================
# 13. Generate Figure 3 and Supplementary Figure 3 heatmaps
# =============================================================================
# This cell saves:
#   Figure3_and_SuppFigure3_custom_layout_final/Figure3_final.pdf
#   Figure3_and_SuppFigure3_custom_layout_final/Figure3_final.png
#   Figure3_and_SuppFigure3_custom_layout_final/SupplementaryFigure3_final.pdf
#   Figure3_and_SuppFigure3_custom_layout_final/SupplementaryFigure3_final.png
#   Figure3_and_SuppFigure3_custom_layout_final/*.csv

heatmap_bundles = run_heatmap_analysis(adata_csf_clean)



# =============================================================================
# 14. Generate 100% stacked barplot of cell-subtype proportions
# =============================================================================
# This cell saves:
#   Pairwise_Heatmaps/Cell_Type_Counts_by_Disease.csv
#   Pairwise_Heatmaps/Cell_Type_Proportion_StackedBar_Grouped_NoFrame_OneLegendColumn.pdf
#   Pairwise_Heatmaps/Cell_Type_Proportion_StackedBar_Grouped_NoFrame_OneLegendColumn.png

stacked_plot_data, stacked_counts = plot_stacked_cell_proportion_barplot(
    adata_csf_clean
)



# =============================================================================
# 15. Generate Figure 3G-O dataset-stratified sample-level boxplots
# =============================================================================
# This cell saves:
#   Figure3G_O_sample_level_cell_proportions/sample_level_celltype_proportions.csv
#   Figure3G_O_sample_level_cell_proportions/Figure3G_O_sample_level_proportions_combined.pdf
#   Figure3G_O_sample_level_cell_proportions/Figure3G_O_sample_level_proportions_combined.png
#   Figure3G_O_sample_level_cell_proportions/Figure3G_O_dataset_vs_global_healthy_stats.csv

sample_prop_df, figure3go_stats_df = run_dataset_boxplot_analysis(
    adata_csf_clean
)



# =============================================================================
# 16. Optional: plot one custom dataset-stratified boxplot
# =============================================================================
# Modify disease and cell_type as needed.

# fig, ax, stats_df = plot_disease_celltype_dataset_boxplot(
#     prop_df=sample_prop_df,
#     disease="MS",
#     cell_type="Plasma cell",
#     dataset_order=DATASET_ORDER_BY_DISEASE.get("MS"),
#     save=True,
# )
# print(stats_df)
