# %%
# =============================================================================
# PBMC composition QC, ROE heatmap, dataset-rich disease statistics, and Naive B boxplot
# =============================================================================
# This notebook-style script performs four analyses:
#   1. Generate a PBMC ROE heatmap for Figure 5A.
#   2. Detect sample-level composition outliers by disease using PCA.
#   3. For diseases represented by at least two independent datasets, test every
#      cell subtype against the global Healthy group and export summary tables.
#   4. Generate a sample-level Naive B proportion boxplot comparing MS with Healthy.
# =============================================================================

import os
import re

import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

from scipy.stats import ranksums, zscore
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator


# =============================================================================
# 1. Global configuration
# =============================================================================
os.chdir("/home/lishaogang/NEU2026_July")
INPUT_H5AD = os.environ.get(
    "PBMC_INPUT_H5AD", "/home/lishaogang/NEU2026_July/adata_final_pbmc.h5ad"
)
RESULT_ROOT = os.environ.get("FIGURE_RESULT_ROOT", "Result")

SAMPLE_COL = "orig.ident"
DISEASE_COL = "Disease"
DATASET_COL = "dataset"
CELLTYPE_COL = "cell_subtype_short"
BROAD_TYPE_COL = "cell_type"
AGE_COL = "age_numeric"

REF_GROUP = "Younger control"
TEST_GROUP = "MS"
TARGET_CELLTYPE = "Naive B"

PCA_OUTLIER_Z_THRESHOLD = 1.5
PCA_MIN_SAMPLES_PER_DISEASE = 5
PCA_DISEASES_TO_PLOT = None  # None means automatically plot every disease with sample count >= PCA_MIN_SAMPLES_PER_DISEASE.
PCA_N_COLS = None  # None means automatically choose a compact grid layout from the number of eligible diseases.
PCA_OUTDIR = os.path.join(RESULT_ROOT, "Supplement_Figure1", "Supplement_Figure1D")
PCA_OUTPUT_PDF = os.path.join(PCA_OUTDIR, "combined_disease_pca_more_outliers.pdf")
PCA_OUTPUT_PNG = os.path.join(PCA_OUTDIR, "combined_disease_pca_more_outliers.png")
PCA_OUTLIER_TABLE = os.path.join(PCA_OUTDIR, "PBMC_PCA_outlier_samples_by_disease.csv")
PCA_SAMPLE_COUNT_TABLE = os.path.join(PCA_OUTDIR, "PBMC_PCA_sample_counts_by_disease.csv")

ROE_OUTDIR = "Result/Figure5/Figure5A"
ROE_OUTPUT_PDF = os.path.join(ROE_OUTDIR, "roe_heatmap_adata_pbmc.pdf")
ROE_OUTPUT_PNG = os.path.join(ROE_OUTDIR, "roe_heatmap_adata_pbmc.png")
ROE_TABLE_PREFIX = "PBMC"

BOXPLOT_OUTDIR = "Result/Figure5/Figure5B"
ALL_CELLTYPE_STATS_OUTDIR = "Resulr/Figure5/PBMC_DatasetGE2_All_Celltype_vs_Healthy_Stats"

PBMC_DISEASE_ORDER = [
    "Younger control",
    "Older control",
    "Unknown",
    "MS",
    "IIH",
    "Ab-mediated IDD",
    "CIS",
]

DEFAULT_ROE_COLORS = [
    "#244773",
    "#73A9D0",
    "#F8F8F8",
    "#D4887A",
    "#752C36",
]

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

for output_dir in [PCA_OUTDIR, ROE_OUTDIR, BOXPLOT_OUTDIR, ALL_CELLTYPE_STATS_OUTDIR]:
    os.makedirs(output_dir, exist_ok=True)

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


# =============================================================================
# 2. General helpers
# =============================================================================

def clean_filename(name):
    """Replace unsafe filename characters with underscores."""
    return re.sub(r'[\\/:*?"<>|]', "_", str(name))


def validate_obs_columns(adata, required_cols):
    """Raise an error if required AnnData obs columns are missing."""
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in adata.obs: {missing_cols}")


def standardize_obs_labels(adata):
    """Return a copy of AnnData with standardized disease and cell-subtype labels."""
    adata = adata.copy()

    if DISEASE_COL in adata.obs.columns:
        adata.obs[DISEASE_COL] = (
            adata.obs[DISEASE_COL]
            .astype(str)
            .str.strip()
            .replace(DISEASE_RENAME_MAP)
        )

    if CELLTYPE_COL in adata.obs.columns:
        adata.obs[CELLTYPE_COL] = (
            adata.obs[CELLTYPE_COL]
            .astype(str)
            .str.strip()
            .replace(CELLTYPE_RENAME_MAP)
        )

    return adata


def clean_obs_table(adata, columns):
    """Extract selected obs columns, standardize labels, and remove empty rows."""
    validate_obs_columns(adata, columns)
    df = adata.obs[columns].copy()

    if DISEASE_COL in df.columns:
        df[DISEASE_COL] = df[DISEASE_COL].astype(str).str.strip().replace(DISEASE_RENAME_MAP)

    if CELLTYPE_COL in df.columns:
        df[CELLTYPE_COL] = df[CELLTYPE_COL].astype(str).str.strip().replace(CELLTYPE_RENAME_MAP)

    for col in columns:
        df[col] = df[col].astype(str).str.strip()

    keep_mask = np.ones(df.shape[0], dtype=bool)
    for col in columns:
        keep_mask &= df[col].notna() & (df[col] != "") & (df[col] != "nan")

    return df.loc[keep_mask].copy()


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


def benjamini_hochberg(p_values):
    """Return Benjamini-Hochberg adjusted P values while preserving NaNs."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    valid_mask = np.isfinite(p_values)

    if valid_mask.sum() == 0:
        return adjusted

    valid_p = p_values[valid_mask]
    order = np.argsort(valid_p)
    ranked_p = valid_p[order]
    n_tests = len(ranked_p)

    ranked_adj = ranked_p * n_tests / np.arange(1, n_tests + 1)
    ranked_adj = np.minimum.accumulate(ranked_adj[::-1])[::-1]
    ranked_adj = np.clip(ranked_adj, 0, 1)

    valid_adjusted = np.empty_like(valid_p)
    valid_adjusted[order] = ranked_adj
    adjusted[valid_mask] = valid_adjusted

    return adjusted


def add_pvalue_bracket(ax, x1, x2, y, h, text, fontsize=10, linewidth=0.9):
    """Add one P-value bracket to a matplotlib axis."""
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        lw=linewidth,
        color="black",
        clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2,
        y + h,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="black",
    )


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


def save_pdf_png(fig, output_pdf, output_png=None, dpi=300, transparent=True):
    """Save a matplotlib figure as PDF and optionally PNG."""
    fig.savefig(output_pdf, bbox_inches="tight", transparent=transparent)
    if output_png is not None:
        fig.savefig(output_png, dpi=dpi, bbox_inches="tight", transparent=transparent)


# =============================================================================
# 3. PCA-based composition outlier detection
# =============================================================================

def build_sample_subtype_matrix(adata, sample_col=SAMPLE_COL, subtype_col=CELLTYPE_COL):
    """Build a sample-by-cell-subtype proportion matrix."""
    obs_df = clean_obs_table(adata, [sample_col, subtype_col])

    count_df = (
        obs_df
        .groupby([sample_col, subtype_col], observed=True)
        .size()
        .reset_index(name="count")
    )

    matrix = (
        count_df
        .pivot(index=sample_col, columns=subtype_col, values="count")
        .fillna(0)
    )

    return matrix.div(matrix.sum(axis=1), axis=0).fillna(0)


def detect_outliers_from_pca_coordinates(pca_coords, z_threshold=PCA_OUTLIER_Z_THRESHOLD):
    """Detect outlier samples from the Z-score of PC1-PC2 Euclidean distance."""
    pca_distance = np.sqrt(pca_coords["PC1"] ** 2 + pca_coords["PC2"] ** 2)
    pca_z = zscore(pca_distance)

    if np.all(~np.isfinite(pca_z)):
        pca_z = np.zeros_like(pca_distance, dtype=float)

    outliers = pca_coords.index[pca_z > z_threshold].tolist()
    normals = pca_coords.index[pca_z <= z_threshold].tolist()

    return outliers, normals, pca_z



def choose_pca_grid_ncols(n_panels):
    """Choose a compact number of columns for the combined PCA figure."""
    if n_panels <= 4:
        return 2
    if n_panels <= 9:
        return 3
    return 4


def get_sample_count_by_disease(adata, sample_col=SAMPLE_COL, disease_col=DISEASE_COL):
    """Count unique samples per disease."""
    validate_obs_columns(adata, [sample_col, disease_col])

    sample_count_df = (
        adata.obs[[sample_col, disease_col]]
        .drop_duplicates()
        .assign(**{
            sample_col: lambda x: x[sample_col].astype(str).str.strip(),
            disease_col: lambda x: x[disease_col].astype(str).str.strip(),
        })
        .groupby(disease_col, observed=True)[sample_col]
        .nunique()
        .rename("n_samples")
        .reset_index()
        .sort_values(["n_samples", disease_col], ascending=[False, True])
        .reset_index(drop=True)
    )

    return sample_count_df

def run_pca_outlier_detection_by_disease(
    adata,
    sample_col=SAMPLE_COL,
    disease_col=DISEASE_COL,
    subtype_col=CELLTYPE_COL,
    min_samples=PCA_MIN_SAMPLES_PER_DISEASE,
    z_threshold=PCA_OUTLIER_Z_THRESHOLD,
    diseases_to_plot=None,
    output_pdf=PCA_OUTPUT_PDF,
    output_png=PCA_OUTPUT_PNG,
    output_outlier_csv=PCA_OUTLIER_TABLE,
    output_sample_count_csv=PCA_SAMPLE_COUNT_TABLE,
    n_cols=PCA_N_COLS,
):
    """Run PCA outlier detection for every disease with enough unique samples."""
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
        .astype(str)
        .to_dict()
    )

    valid_diseases = []
    pca_plot_data = {}
    all_outliers = {}

    sample_count_df = get_sample_count_by_disease(
        adata,
        sample_col=sample_col,
        disease_col=disease_col,
    )

    if output_sample_count_csv is not None:
        sample_count_df.to_csv(output_sample_count_csv, index=False)
        print(f"Saved PCA sample-count table: {output_sample_count_csv}")

    eligible_by_sample_count = sample_count_df.loc[
        sample_count_df["n_samples"] >= min_samples,
        disease_col,
    ].astype(str).tolist()

    if diseases_to_plot is None:
        candidate_diseases = eligible_by_sample_count
    else:
        requested_diseases = [str(disease) for disease in diseases_to_plot]
        candidate_diseases = [
            disease for disease in requested_diseases
            if disease in eligible_by_sample_count
        ]
        missing_or_ineligible = [
            disease for disease in requested_diseases
            if disease not in eligible_by_sample_count
        ]
        if missing_or_ineligible:
            print(
                "Warning: requested PCA diseases were not found or had fewer "
                f"than {min_samples} samples:",
                missing_or_ineligible,
            )

    print("Diseases selected for PCA plotting:", candidate_diseases)

    for disease in candidate_diseases:
        disease_samples = [
            sample for sample in sample_subtype_matrix.index
            if sample_disease_map.get(sample) == disease
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

    if n_cols is None:
        n_cols = choose_pca_grid_ncols(len(valid_diseases))

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

    plt.tight_layout()
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight", facecolor="white")

    if output_png is not None:
        fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")

    plt.show()

    remove_list = sorted({
        sample
        for outlier_samples in all_outliers.values()
        for sample in outlier_samples
    })

    outlier_table = pd.DataFrame([
        {
            "disease": disease,
            "n_samples": int(
                sample_count_df.loc[
                    sample_count_df[disease_col].astype(str) == str(disease),
                    "n_samples",
                ].iloc[0]
            ),
            "n_outliers": len(outliers),
            "outlier_samples": ";".join(map(str, outliers)),
        }
        for disease, outliers in all_outliers.items()
    ])

    if output_outlier_csv is not None:
        outlier_table.to_csv(output_outlier_csv, index=False)

    print("\nOutlier samples by disease:")
    for disease, outliers in all_outliers.items():
        print(f"{disease}: {outliers if outliers else 'None'}")

    print("\nFinal sample removal list:")
    print(f"Number of samples to remove: {len(remove_list)}")
    print(f"remove_list_pbmc = {remove_list}")
    print(f"Saved combined PCA PDF: {output_pdf}")

    if output_png is not None:
        print(f"Saved combined PCA PNG: {output_png}")

    if output_outlier_csv is not None:
        print(f"Saved PCA outlier table: {output_outlier_csv}")

    return remove_list, all_outliers, pca_plot_data


# =============================================================================
# 4. Sample-level proportion tables and dataset-rich disease statistics
# =============================================================================

def build_sample_level_celltype_proportion_table(
    adata,
    sample_col=SAMPLE_COL,
    disease_col=DISEASE_COL,
    dataset_col=DATASET_COL,
    celltype_col=CELLTYPE_COL,
):
    """Calculate sample-level cell-subtype proportions using total sample cell count as denominator."""
    columns = [sample_col, disease_col, celltype_col]
    has_dataset = dataset_col in adata.obs.columns

    if has_dataset:
        columns.insert(2, dataset_col)

    obs_df = clean_obs_table(adata, columns)

    if has_dataset:
        obs_df["sample_uid"] = obs_df[dataset_col].astype(str) + "|" + obs_df[sample_col].astype(str)
    else:
        obs_df["sample_uid"] = obs_df[sample_col].astype(str)
        obs_df[dataset_col] = "Unknown"

    sample_meta = obs_df[["sample_uid", sample_col, disease_col, dataset_col]].drop_duplicates().copy()

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

    prop_df = count_df.merge(sample_meta, on="sample_uid", how="left").merge(
        sample_totals,
        on="sample_uid",
        how="left",
    )

    prop_df["proportion"] = prop_df["cell_count"] / prop_df["sample_total"] * 100.0
    return prop_df


def summarize_datasets_per_disease(prop_df, ref_group=REF_GROUP):
    """Summarize how many independent datasets are available for each disease."""
    dataset_summary = (
        prop_df[[DISEASE_COL, DATASET_COL]]
        .drop_duplicates()
        .groupby(DISEASE_COL, observed=True)[DATASET_COL]
        .agg(
            n_datasets="nunique",
            dataset_list=lambda values: ";".join(sorted(map(str, pd.unique(values)))),
        )
        .reset_index()
        .sort_values(["n_datasets", DISEASE_COL], ascending=[False, True])
    )

    dataset_summary["is_reference"] = dataset_summary[DISEASE_COL].astype(str) == ref_group
    dataset_summary["eligible_dataset_ge2"] = (
        (dataset_summary["n_datasets"] >= 2)
        & (~dataset_summary["is_reference"])
    )

    return dataset_summary


def compare_one_disease_celltype(prop_df, disease, cell_type, ref_group=REF_GROUP, min_samples_per_group=2):
    """Compare one disease-cell-type combination against global Healthy."""
    sub_df = prop_df[prop_df[CELLTYPE_COL].astype(str) == str(cell_type)].copy()

    ref_values = (
        sub_df.loc[sub_df[DISEASE_COL].astype(str) == str(ref_group), "proportion"]
        .astype(float)
        .to_numpy()
    )
    disease_values = (
        sub_df.loc[sub_df[DISEASE_COL].astype(str) == str(disease), "proportion"]
        .astype(float)
        .to_numpy()
    )

    if len(ref_values) >= min_samples_per_group and len(disease_values) >= min_samples_per_group:
        _, p_value = ranksums(disease_values, ref_values)
    else:
        p_value = np.nan

    return {
        "n_healthy_samples": int(len(ref_values)),
        "n_disease_samples": int(len(disease_values)),
        "healthy_mean": float(np.mean(ref_values)) if len(ref_values) > 0 else np.nan,
        "disease_mean": float(np.mean(disease_values)) if len(disease_values) > 0 else np.nan,
        "mean_diff_disease_minus_healthy": (
            float(np.mean(disease_values) - np.mean(ref_values))
            if len(ref_values) > 0 and len(disease_values) > 0
            else np.nan
        ),
        "healthy_median": float(np.median(ref_values)) if len(ref_values) > 0 else np.nan,
        "disease_median": float(np.median(disease_values)) if len(disease_values) > 0 else np.nan,
        "median_diff_disease_minus_healthy": (
            float(np.median(disease_values) - np.median(ref_values))
            if len(ref_values) > 0 and len(disease_values) > 0
            else np.nan
        ),
        "p_value_raw": p_value,
        "p_label_raw": format_raw_p_value_or_stars(p_value),
    }


def calculate_dataset_ge2_disease_celltype_stats(
    prop_df,
    ref_group=REF_GROUP,
    min_datasets=2,
    min_samples_per_group=2,
):
    """Compare every cell subtype in dataset-rich diseases against global Healthy."""
    dataset_summary = summarize_datasets_per_disease(prop_df, ref_group=ref_group)

    eligible_diseases = dataset_summary.loc[
        dataset_summary["eligible_dataset_ge2"],
        DISEASE_COL,
    ].astype(str).tolist()

    if len(eligible_diseases) == 0:
        raise ValueError(f"No non-{ref_group} disease has at least {min_datasets} datasets.")

    cell_types = sorted(prop_df[CELLTYPE_COL].astype(str).unique().tolist())
    rows = []

    for disease in eligible_diseases:
        disease_info = dataset_summary.loc[
            dataset_summary[DISEASE_COL].astype(str) == str(disease)
        ].iloc[0]

        for cell_type in cell_types:
            row = compare_one_disease_celltype(
                prop_df=prop_df,
                disease=disease,
                cell_type=cell_type,
                ref_group=ref_group,
                min_samples_per_group=min_samples_per_group,
            )
            row.update({
                "disease": disease,
                "cell_type": cell_type,
                "n_datasets": int(disease_info["n_datasets"]),
                "dataset_list": disease_info["dataset_list"],
            })
            rows.append(row)

    stats_df = pd.DataFrame(rows)
    front_cols = ["disease", "cell_type", "n_datasets", "dataset_list"]
    stats_df = stats_df[front_cols + [col for col in stats_df.columns if col not in front_cols]]

    stats_df["p_adj_global_fdr"] = benjamini_hochberg(stats_df["p_value_raw"].to_numpy())
    stats_df["signif_global_fdr"] = stats_df["p_adj_global_fdr"].apply(format_raw_p_value_or_stars)

    stats_df["p_adj_within_celltype_fdr"] = np.nan
    for _, idx in stats_df.groupby("cell_type", observed=True).groups.items():
        stats_df.loc[idx, "p_adj_within_celltype_fdr"] = benjamini_hochberg(
            stats_df.loc[idx, "p_value_raw"].to_numpy()
        )

    stats_df["signif_within_celltype_fdr"] = stats_df["p_adj_within_celltype_fdr"].apply(
        format_raw_p_value_or_stars
    )

    stats_df = stats_df.sort_values(
        ["cell_type", "p_value_raw", "disease"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    return stats_df, dataset_summary


def save_dataset_ge2_disease_celltype_stats(
    prop_df,
    outdir=ALL_CELLTYPE_STATS_OUTDIR,
    ref_group=REF_GROUP,
    min_datasets=2,
):
    """Calculate and save all-celltype statistics for dataset-rich diseases."""
    os.makedirs(outdir, exist_ok=True)

    stats_df, dataset_summary = calculate_dataset_ge2_disease_celltype_stats(
        prop_df=prop_df,
        ref_group=ref_group,
        min_datasets=min_datasets,
    )

    output_paths = {
        "sample_level_proportions": os.path.join(outdir, "PBMC_cleaned_sample_level_celltype_proportions.csv"),
        "dataset_summary": os.path.join(outdir, "PBMC_disease_dataset_counts.csv"),
        "stats": os.path.join(outdir, "PBMC_datasetGE2_diseases_all_celltypes_vs_global_Healthy_stats.csv"),
    }

    prop_df.to_csv(output_paths["sample_level_proportions"], index=False)
    dataset_summary.to_csv(output_paths["dataset_summary"], index=False)
    stats_df.to_csv(output_paths["stats"], index=False)

    for label, path in output_paths.items():
        print(f"Saved {label}: {path}")

    return stats_df, dataset_summary


# =============================================================================
# 5. MS vs Healthy Naive B boxplot
# =============================================================================

def plot_two_group_celltype_boxplot(
    prop_df,
    cell_type=TARGET_CELLTYPE,
    ref_group=REF_GROUP,
    test_group=TEST_GROUP,
    outdir=BOXPLOT_OUTDIR,
):
    """Plot one sample-level cell-subtype proportion boxplot comparing two disease groups."""
    os.makedirs(outdir, exist_ok=True)

    df = prop_df[
        (prop_df[CELLTYPE_COL].astype(str) == cell_type)
        & (prop_df[DISEASE_COL].astype(str).isin([ref_group, test_group]))
    ].copy()

    if df.empty:
        raise ValueError(f"No data found for {cell_type} in {ref_group} and {test_group}.")

    plot_order = [ref_group, test_group]
    df[DISEASE_COL] = pd.Categorical(df[DISEASE_COL], categories=plot_order, ordered=True)

    ref_values = df.loc[df[DISEASE_COL] == ref_group, "proportion"].astype(float).to_numpy()
    test_values = df.loc[df[DISEASE_COL] == test_group, "proportion"].astype(float).to_numpy()

    if len(ref_values) >= 2 and len(test_values) >= 2:
        _, p_value = ranksums(test_values, ref_values)
    else:
        p_value = np.nan

    stats_df = pd.DataFrame([{
        "cell_type": cell_type,
        "ref_group": ref_group,
        "test_group": test_group,
        "n_ref_samples": len(ref_values),
        "n_test_samples": len(test_values),
        "ref_median": np.median(ref_values) if len(ref_values) > 0 else np.nan,
        "test_median": np.median(test_values) if len(test_values) > 0 else np.nan,
        "median_diff_test_minus_ref": (
            np.median(test_values) - np.median(ref_values)
            if len(ref_values) > 0 and len(test_values) > 0
            else np.nan
        ),
        "p_value_raw": p_value,
        "p_label": format_raw_p_value_or_stars(p_value),
    }])

    group_counts = (
        df.groupby(DISEASE_COL, observed=True)["sample_uid"]
        .nunique()
        .reindex(plot_order)
        .fillna(0)
        .astype(int)
    )

    fig, ax = plt.subplots(figsize=(4.2, 4.6))

    sns.boxplot(
        data=df,
        x=DISEASE_COL,
        y="proportion",
        order=plot_order,
        palette={ref_group: "#F2F2F2", test_group: "#E8B9B4"},
        width=0.55,
        linewidth=0.9,
        fliersize=0,
        ax=ax,
        boxprops={"edgecolor": "#4D4D4D", "alpha": 0.90},
        whiskerprops={"color": "#4D4D4D", "linewidth": 0.9},
        capprops={"color": "#4D4D4D", "linewidth": 0.9},
        medianprops={"color": "#4D4D4D", "linewidth": 1.1},
    )

    sns.stripplot(
        data=df,
        x=DISEASE_COL,
        y="proportion",
        order=plot_order,
        color="#8C2D23",
        edgecolor="black",
        linewidth=0.30,
        size=3.2,
        jitter=0.18,
        alpha=0.85,
        ax=ax,
        zorder=10,
    )

    y_values = df["proportion"].astype(float).to_numpy()
    y_values = y_values[np.isfinite(y_values)]
    y_max = float(np.max(y_values)) if len(y_values) > 0 else 1.0
    y_range = max(y_max, 0.5)
    bracket_y = y_max + y_range * 0.12
    bracket_h = y_range * 0.05

    add_pvalue_bracket(
        ax=ax,
        x1=0,
        x2=1,
        y=bracket_y,
        h=bracket_h,
        text=stats_df.loc[0, "p_label"],
        fontsize=10,
        linewidth=0.9,
    )

    ax.set_ylim(0, bracket_y + bracket_h + y_range * 0.18)
    ax.set_xticklabels([f"{group}\n(n={group_counts.loc[group]})" for group in plot_order])
    ax.set_xlabel("")
    ax.set_ylabel(f"{cell_type} fraction (%)", fontsize=11)
    ax.set_title(f"{cell_type}: {test_group} vs {ref_group}", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.45, color="#A9A9A9")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    safe_ref = clean_filename(ref_group)
    safe_test = clean_filename(test_group)
    safe_cell = clean_filename(cell_type.replace(" ", "_"))

    output_pdf = os.path.join(outdir, f"{safe_test}_vs_{safe_ref}_{safe_cell}_sample_proportion_boxplot.pdf")
    output_png = os.path.join(outdir, f"{safe_test}_vs_{safe_ref}_{safe_cell}_sample_proportion_boxplot.png")
    output_stats = os.path.join(outdir, f"{safe_test}_vs_{safe_ref}_{safe_cell}_sample_proportion_stats.csv")
    output_data = os.path.join(outdir, f"{safe_test}_vs_{safe_ref}_{safe_cell}_sample_proportion_values.csv")

    stats_df.to_csv(output_stats, index=False)
    df.to_csv(output_data, index=False)
    save_pdf_png(fig, output_pdf, output_png, transparent=True)
    plt.show()

    print(f"Saved boxplot PDF: {output_pdf}")
    print(f"Saved boxplot PNG: {output_png}")
    print(f"Saved statistics table: {output_stats}")
    print(f"Saved plotting data: {output_data}")

    return fig, ax, stats_df, df


# =============================================================================
# 6. ROE heatmap data preparation
# =============================================================================

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
# 7. ROE heatmap plotting helpers
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


# %%
# =============================================================================
# 8. Load PBMC AnnData
# =============================================================================

adata_pbmc = sc.read_h5ad(INPUT_H5AD)
adata_pbmc = standardize_obs_labels(adata_pbmc)
print(adata_pbmc)




# %%
# =============================================================================
# 9. Detect sample-level composition outliers by PCA
# =============================================================================

remove_list_pbmc, all_outliers, pca_plot_data = run_pca_outlier_detection_by_disease(
    adata_pbmc,
    diseases_to_plot=PCA_DISEASES_TO_PLOT,
    output_pdf=PCA_OUTPUT_PDF,
    output_png=PCA_OUTPUT_PNG,
    output_outlier_csv=PCA_OUTLIER_TABLE,
    output_sample_count_csv=PCA_SAMPLE_COUNT_TABLE,
    n_cols=PCA_N_COLS,
)

print("Number of outlier samples to remove:", len(remove_list_pbmc))
print("remove_list_pbmc =", remove_list_pbmc)

if os.environ.get("SUPPLEMENT1D_ONLY") == "1":
    print("SUPPLEMENT1D_ONLY=1: Supplementary Figure 1D completed; skipping Figure 5 analyses.")
    raise SystemExit(0)


# %%
# =============================================================================
# 10. Remove outlier samples
# =============================================================================
# Review remove_list_pbmc before running this cell.
# The cleaned object is used for the statistics table and the Naive B boxplot.

adata_pbmc_clean = adata_pbmc[
    ~adata_pbmc.obs[SAMPLE_COL].isin(remove_list_pbmc)
].copy()

print(adata_pbmc_clean)



# %%
# =============================================================================
# 11. Generate Figure 5A PBMC ROE heatmap
# =============================================================================

data_dict_pbmc = calculate_roe_heatmap_data(
    adata=adata_pbmc_clean,
    sample_col=SAMPLE_COL,
    group_col=DISEASE_COL,
    celltype_col=CELLTYPE_COL,
    broad_type_col=BROAD_TYPE_COL,
    age_col=AGE_COL,
)

fig_pbmc_roe = plot_roe_heatmap(
    data_dict=data_dict_pbmc,
    cell_size=0.35,
    vmax=10,
    # disease_order=PBMC_DISEASE_ORDER,
    roe_colors=DEFAULT_ROE_COLORS,
    save_path=ROE_OUTPUT_PDF,
    output_png=ROE_OUTPUT_PNG,
    table_outdir=ROE_OUTDIR,
    table_prefix=ROE_TABLE_PREFIX,
    dpi=300,
    cbar_x=0.875,
)


# %%
# =============================================================================
# 12. Generate dataset-rich disease statistics and the MS vs Healthy Naive B boxplot
# =============================================================================

sample_prop_df = build_sample_level_celltype_proportion_table(adata_pbmc_clean)

dataset_ge2_stats_df, disease_dataset_summary_df = save_dataset_ge2_disease_celltype_stats(
    prop_df=sample_prop_df,
    outdir=ALL_CELLTYPE_STATS_OUTDIR,
    ref_group=REF_GROUP,
    min_datasets=2,
)

print(disease_dataset_summary_df)
print(dataset_ge2_stats_df.head(20))

fig_naive_b, ax_naive_b, naive_b_stats_df, naive_b_plot_df = plot_two_group_celltype_boxplot(
    prop_df=sample_prop_df,
    cell_type=TARGET_CELLTYPE,
    ref_group=REF_GROUP,
    test_group=TEST_GROUP,
    outdir=BOXPLOT_OUTDIR,
)

print(naive_b_stats_df)
