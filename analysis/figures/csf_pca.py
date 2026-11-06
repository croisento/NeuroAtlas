"""Plotting functions extracted from 05_Figure3_SupplementaryFigure1C_SupplementaryFigure2_SupplementaryFigure3_CSF_composition_analysis.py; top-level data reads omitted."""
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
DISEASE_RENAME_MAP = {'MCVAD': 'MCI/AD', 'MCI_AD': 'MCI/AD', 'MCI-AD': 'MCI/AD', 'CAR-T/DIPG': 'CAR-T-DIPG'}
CELLTYPE_RENAME_MAP = {'CD4 Naïve': 'CD4 Naive', 'CD8 Naïve': 'CD8 Naive', 'maMac': 'mdMac'}
PCA_OUTLIER_Z_THRESHOLD = 1.5
PCA_MIN_SAMPLES_PER_DISEASE = 5
ROE_TABLE_PREFIX = 'SupplementaryFigure2_CSF'
DEFAULT_ROE_COLORS = ['#244773', '#73A9D0', '#F8F8F8', '#D4887A', '#752C36']
ARIAL_FONT_PATH = '/home/lishaogang/.fonts/arial.ttf'
DISEASE_ORDER_FULL = ['Younger control', 'Older control', 'Unknown', 'MCI/AD', 'PD', 'IIH', 'MS', 'Ab-mediated IDD', 'VE', 'Neuro-COVID', 'CAR-T-DIPG', 'NT1']
DISEASE_ORDER_STACKED_BAR = ['Younger control', 'Older control', 'Unknown', 'MCI/AD', 'PD', 'IIH', 'MS']
BROAD_TO_SUBTYPES = {'T_cell': ['MAIT', 'CD4 CTL', 'NKT', 'CD8 Naive', 'CD4 Memory', 'CD8 CTL', 'CD8 Memory', 'Treg', 'γδT', 'CD4 Naive'], 'B_cell': ['Plasma cell', 'Naive B', 'Memory B'], 'ILCs': ['ILC2', 'NK bright', 'NK dim', 'Transit. NK'], 'Myeloid': ['cMonocytes', 'cDC2', 'infMac', 'trMac', 'ASDC', 'mdMac', 'maMac', 'ncMonocytes', 'Basophil']}
STACKED_CELL_ORDER = ['Naive B', 'Memory B', 'Plasma cell', 'CD4 Naive', 'CD8 Naive', 'CD4 Memory', 'CD8 Memory', 'Treg', 'CD4 CTL', 'CD8 CTL', 'MAIT', 'γδT', 'NKT', 'NK bright', 'Transit. NK', 'NK dim', 'ILC2', 'cMonocytes', 'ncMonocytes', 'mdMac', 'trMac', 'infMac', 'cDC2', 'pDC', 'ASDC']
STACKED_COLOR_MAP = {'Naive B': '#C96F6F', 'Memory B': '#B85C68', 'Plasma cell': '#B77AB5', 'CD4 Naive': '#7DA7CF', 'CD8 Naive': '#4F9A78', 'CD4 Memory': '#AEC9E4', 'CD8 Memory': '#98C970', 'Treg': '#3F6BA7', 'CD4 CTL': '#3E7C9D', 'CD8 CTL': '#C3D894', 'MAIT': '#75B38A', 'γδT': '#A6B85A', 'NKT': '#BFB552', 'NK bright': '#D7BE66', 'Transit. NK': '#A995C1', 'NK dim': '#C99235', 'ILC2': '#DDD36A', 'cMonocytes': '#C97891', 'ncMonocytes': '#9E5F73', 'mdMac': '#9568B1', 'trMac': '#D6A196', 'infMac': '#C17DB4', 'cDC2': '#B46AB0', 'pDC': '#7F8CC1', 'ASDC': '#B79BD0'}
FIGURE3GO_TARGET_PANELS = [{'panel': 'G', 'disease': 'MS', 'cell_type': 'Naive B', 'display_cell_type': 'Naive B'}, {'panel': 'H', 'disease': 'MS', 'cell_type': 'Plasma cell', 'display_cell_type': 'Plasma cell'}, {'panel': 'I', 'disease': 'MS', 'cell_type': 'ASDC', 'display_cell_type': 'ASDC'}, {'panel': 'J', 'disease': 'MS', 'cell_type': 'CD4 Naive', 'display_cell_type': 'CD4 Naïve'}, {'panel': 'K', 'disease': 'MS', 'cell_type': 'Treg', 'display_cell_type': 'Treg'}, {'panel': 'L', 'disease': 'PD', 'cell_type': 'NK dim', 'display_cell_type': 'NK dim'}, {'panel': 'M', 'disease': 'MCI/AD', 'cell_type': 'NK dim', 'display_cell_type': 'NK dim'}, {'panel': 'N', 'disease': 'Older control', 'cell_type': 'ASDC', 'display_cell_type': 'ASDC'}, {'panel': 'O', 'disease': 'IIH', 'cell_type': 'Naive B', 'display_cell_type': 'Naive B'}]
DATASET_ORDER_BY_DISEASE = {'MS': ['GSE133028', 'GSE138266', 'GSE163005', 'GSE194078', 'GSE200164', 'GSE247134'], 'PD': ['GSE134577', 'GSE141578'], 'MCI/AD': ['GSE134577', 'GSE200164'], 'Older control': ['GSE134577', 'GSE200164'], 'IIH': ['GSE138266', 'GSE163005', 'GSE247134']}
ARIAL_FAMILY = 'DejaVu Sans'
FONT_FAMILY = 'DejaVu Sans'
PCA_OUTPUT_PNG = None
PCA_OUTLIER_TABLE = None
PCA_OUTPUT_PDF = None
def validate_obs_columns(adata, required_cols):
    """Raise an error if any required AnnData obs columns are missing."""
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in adata.obs: {missing_cols}")

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
    """Generate Figure S1a: disease-wise PCA outlier detection in one combined figure."""
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
        "Figure S1a",
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

    print("\nPCA-flagged samples (retained):")
    print(f"Number of flagged samples: {len(remove_list)}")
    print(f"remove_list_csf = {remove_list}")
    print(f"Saved Figure S1a PCA PDF: {output_pdf}")
    if output_png is not None:
        print(f"Saved Figure S1a PCA PNG: {output_png}")
    if output_outlier_csv is not None:
        print(f"Saved Figure S1a PCA outlier table: {output_outlier_csv}")

    return remove_list, all_outliers, pca_plot_data
