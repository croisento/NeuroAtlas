"""Plotting functions extracted from 07_Figure5_AB_and_Supplement_Figure1D.py; top-level data reads omitted."""
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
SAMPLE_COL = 'orig.ident'
DISEASE_COL = 'Disease'
DATASET_COL = 'dataset'
CELLTYPE_COL = 'cell_subtype_short'
BROAD_TYPE_COL = 'cell_type'
AGE_COL = 'age_numeric'
REF_GROUP = 'Younger control'
TEST_GROUP = 'MS'
TARGET_CELLTYPE = 'Naive B'
PCA_OUTLIER_Z_THRESHOLD = 1.5
PCA_MIN_SAMPLES_PER_DISEASE = 5
PCA_DISEASES_TO_PLOT = None
PCA_N_COLS = None
ROE_OUTDIR = 'Result/Figure5/Figure5A'
ROE_TABLE_PREFIX = 'PBMC'
BOXPLOT_OUTDIR = 'Result/Figure5/Figure5B'
ALL_CELLTYPE_STATS_OUTDIR = 'Resulr/Figure5/PBMC_DatasetGE2_All_Celltype_vs_Healthy_Stats'
PBMC_DISEASE_ORDER = ['Younger control', 'Older control', 'Unknown', 'MS', 'IIH', 'Ab-mediated IDD', 'CIS']
DEFAULT_ROE_COLORS = ['#244773', '#73A9D0', '#F8F8F8', '#D4887A', '#752C36']
DISEASE_RENAME_MAP = {'MCVAD': 'MCI/AD', 'MCI_AD': 'MCI/AD', 'MCI-AD': 'MCI/AD', 'CAR-T/DIPG': 'CAR-T-DIPG'}
CELLTYPE_RENAME_MAP = {'CD4 Naïve': 'CD4 Naive', 'CD8 Naïve': 'CD8 Naive', 'maMac': 'mdMac'}
ARIAL_FONT_PATH = '/home/lishaogang/.fonts/arial.ttf'
ARIAL_FAMILY = 'DejaVu Sans'
FONT_FAMILY = 'DejaVu Sans'
PCA_SAMPLE_COUNT_TABLE = None
PCA_OUTPUT_PNG = None
PCA_OUTLIER_TABLE = None
PCA_OUTPUT_PDF = None
def validate_obs_columns(adata, required_cols):
    """Raise an error if required AnnData obs columns are missing."""
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in adata.obs: {missing_cols}")

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

    print("\nPCA-flagged samples (retained):")
    print(f"Number of flagged samples: {len(remove_list)}")
    print(f"remove_list_pbmc = {remove_list}")
    print(f"Saved combined PCA PDF: {output_pdf}")

    if output_png is not None:
        print(f"Saved combined PCA PNG: {output_png}")

    if output_outlier_csv is not None:
        print(f"Saved PCA outlier table: {output_outlier_csv}")

    return remove_list, all_outliers, pca_plot_data

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
