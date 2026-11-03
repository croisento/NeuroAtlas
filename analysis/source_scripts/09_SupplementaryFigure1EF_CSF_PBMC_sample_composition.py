# %%
# =============================================================================
# Supplementary Figure 1E-F: CSF and PBMC sample composition across datasets
# =============================================================================
# This notebook-style script generates:
#   Supplementary Figure 1E: CSF sample composition by dataset and disease
#   Supplementary Figure 1F: PBMC sample composition by dataset and disease
#
# Main outputs:
#   SupplementaryFigure1E_F_Sample_Composition/
#   ├── SupplementaryFigure1E_CSF_sample_composition.pdf
#   ├── SupplementaryFigure1E_CSF_sample_composition.png
#   ├── SupplementaryFigure1F_PBMC_sample_composition.pdf
#   ├── SupplementaryFigure1F_PBMC_sample_composition.png
#   └── tables/
#       ├── SupplementaryFigure1E_CSF_sample_composition_counts.csv
#       ├── SupplementaryFigure1E_CSF_sample_composition_matrix.csv
#       ├── SupplementaryFigure1F_PBMC_sample_composition_counts.csv
#       ├── SupplementaryFigure1F_PBMC_sample_composition_matrix.csv
#       └── SupplementaryFigure1E_F_combined_sample_composition_counts.csv
# =============================================================================

import logging
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns


# =============================================================================
# 1. Global configuration
# =============================================================================
PROJECT_DIR = os.environ.get("NEU_PROJECT_DIR", "/home/lishaogang/NEU2026_July")
os.chdir(PROJECT_DIR)
INPUT_CSF_H5AD = os.environ.get(
    "CSF_INPUT_H5AD", os.path.join(PROJECT_DIR, "adata_final_csf.h5ad")
)
INPUT_PBMC_H5AD = os.environ.get(
    "PBMC_INPUT_H5AD", os.path.join(PROJECT_DIR, "adata_final_pbmc.h5ad")
)
RESULT_ROOT = os.environ.get("FIGURE_RESULT_ROOT", "Result")

OUTPUT_DIR = os.path.join(RESULT_ROOT, "Supplement_Figure1", "Supplement_Figure1E_F")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")

DATASET_COL = "dataset"
DISEASE_COL = "Disease"
SAMPLE_COL = "orig.ident"

FIGURE_DPI = 300
BASE_DPI = 120
BAR_WIDTH = 0.60
FIXED_HEIGHT = 7.0
SPACE_PER_BAR = 1.20
FIXED_YLIM = (0, 60)
MIN_LABEL_THRESHOLD = 3.0
NAME_MAX_LENGTH = 12
FORCE_BLACK_LABEL = "Ab-mediated IDD"

DISEASE_RENAME_MAP = {
    "MCVAD": "MCI/AD",
    "MCI_AD": "MCI/AD",
    "MCI-AD": "MCI/AD",
    "CAR-T/DIPG": "CAR-T-DIPG",
}

DISEASE_ORDER = [
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
    "CIS",
    "RIS",
    "DLB",
    "PHN",
    "Uveitis",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

ARIAL_FONT_PATH = "/home/lishaogang/.fonts/arial.ttf"
if os.path.exists(ARIAL_FONT_PATH):
    font_manager.fontManager.addfont(ARIAL_FONT_PATH)
    FONT_FAMILY = font_manager.FontProperties(fname=ARIAL_FONT_PATH).get_name()
else:
    FONT_FAMILY = "Arial"

logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

mpl.rcParams.update({
    "font.family": FONT_FAMILY,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

sns.set_style("white")


# =============================================================================
# 2. Shared helpers
# =============================================================================

def clean_filename(name: str) -> str:
    """Replace unsafe characters in a file name."""
    return re.sub(r'[\\/:*?"<>|]', "_", str(name))


def validate_obs_columns(adata, required_cols: Sequence[str]) -> None:
    """Raise an informative error if required AnnData obs columns are missing."""
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in adata.obs: {missing_cols}")


def standardize_metadata_df(
    df: pd.DataFrame,
    disease_col: str = DISEASE_COL,
) -> pd.DataFrame:
    """Standardize disease labels and remove empty metadata values."""
    df = df.copy()

    if disease_col in df.columns:
        df[disease_col] = (
            df[disease_col]
            .astype(str)
            .str.strip()
            .replace(DISEASE_RENAME_MAP)
        )

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df = df[(df[col] != "") & (df[col] != "nan")].copy()

    return df


def make_consistent_disease_palette(diseases: Iterable[str], cmap_name: str = "tab20") -> Dict[str, Tuple[float, float, float, float]]:
    """Create a stable disease-color mapping shared by CSF and PBMC figures."""
    diseases = list(dict.fromkeys(map(str, diseases)))
    ordered = [d for d in DISEASE_ORDER if d in diseases]
    ordered += sorted([d for d in diseases if d not in ordered])

    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, max(len(ordered), 1)))
    return dict(zip(ordered, colors))


def shorten_label(label: str, max_length: int = NAME_MAX_LENGTH) -> str:
    """Shorten long labels used inside stacked bars."""
    label = str(label)
    if len(label) > max_length:
        return label[: max_length - 2] + ".."
    return label


def choose_text_color(color, disease: str, force_black_label: str = FORCE_BLACK_LABEL) -> str:
    """Choose readable text color for labels inside colored bars."""
    if disease == force_black_label:
        return "black"
    return "white" if np.mean(color[:3]) < 0.60 else "#333333"


# =============================================================================
# 3. Data preparation
# =============================================================================

def build_sample_composition_table(
    adata,
    dataset_col: str = DATASET_COL,
    disease_col: str = DISEASE_COL,
    sample_col: str = SAMPLE_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Count unique samples for each dataset-disease combination."""
    validate_obs_columns(adata, [dataset_col, disease_col, sample_col])

    sample_metadata = adata.obs[[dataset_col, disease_col, sample_col]].drop_duplicates().copy()
    sample_metadata = standardize_metadata_df(sample_metadata, disease_col=disease_col)

    count_df = (
        sample_metadata
        .groupby([dataset_col, disease_col], observed=True)[sample_col]
        .nunique()
        .rename("n_samples")
        .reset_index()
    )

    matrix_df = (
        count_df
        .pivot(index=dataset_col, columns=disease_col, values="n_samples")
        .fillna(0)
        .astype(int)
    )

    disease_order = [disease for disease in DISEASE_ORDER if disease in matrix_df.columns]
    disease_order += sorted([disease for disease in matrix_df.columns if disease not in disease_order])
    matrix_df = matrix_df.reindex(columns=disease_order)

    dataset_order = matrix_df.sum(axis=1).sort_values(ascending=False).index.tolist()
    matrix_df = matrix_df.loc[dataset_order]

    count_df = count_df.merge(
        matrix_df.sum(axis=1).rename("dataset_total_samples").reset_index(),
        on=dataset_col,
        how="left",
    )

    count_df[disease_col] = pd.Categorical(
        count_df[disease_col],
        categories=disease_order,
        ordered=True,
    )
    count_df[dataset_col] = pd.Categorical(
        count_df[dataset_col],
        categories=dataset_order,
        ordered=True,
    )
    count_df = count_df.sort_values([dataset_col, disease_col]).reset_index(drop=True)

    return count_df, matrix_df


def save_sample_composition_tables(
    count_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    prefix: str,
    table_dir: str = TABLE_DIR,
) -> Dict[str, str]:
    """Save long-format and matrix-format sample composition tables."""
    os.makedirs(table_dir, exist_ok=True)

    count_path = os.path.join(table_dir, f"{prefix}_sample_composition_counts.csv")
    matrix_path = os.path.join(table_dir, f"{prefix}_sample_composition_matrix.csv")

    count_df.to_csv(count_path, index=False)
    matrix_df.to_csv(matrix_path)

    print(f"Saved count table: {count_path}")
    print(f"Saved matrix table: {matrix_path}")

    return {"counts": count_path, "matrix": matrix_path}


# =============================================================================
# 4. Plotting
# =============================================================================

def plot_dataset_sample_composition(
    matrix_df: pd.DataFrame,
    title: str,
    output_pdf: str,
    output_png: str,
    disease_palette: Dict[str, Tuple[float, float, float, float]],
    xlabel: str = "Dataset Source",
    ylabel: str = "Total Number of Samples",
    min_label_threshold: float = MIN_LABEL_THRESHOLD,
    name_max_length: int = NAME_MAX_LENGTH,
    force_black_label: str = FORCE_BLACK_LABEL,
    bar_width: float = BAR_WIDTH,
    fixed_height: float = FIXED_HEIGHT,
    space_per_bar: float = SPACE_PER_BAR,
    fixed_ylim: Optional[Tuple[float, float]] = FIXED_YLIM,
    dpi: int = BASE_DPI,
):
    """Plot a stacked bar chart showing disease composition across datasets."""
    if matrix_df.empty:
        raise ValueError("The sample-composition matrix is empty.")

    diseases = matrix_df.columns.tolist()
    datasets = matrix_df.index.tolist()
    totals = matrix_df.sum(axis=1)

    n_datasets = len(datasets)
    figsize = (max(5.0, n_datasets * space_per_bar), fixed_height)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x_positions = np.arange(n_datasets)
    bottom = np.zeros(n_datasets)

    for disease in diseases:
        values = matrix_df[disease].astype(float).to_numpy()
        color = disease_palette.get(disease, "#BDBDBD")

        ax.bar(
            x_positions,
            values,
            bottom=bottom,
            label=disease,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            width=bar_width,
        )

        for idx, value in enumerate(values):
            if value > 0:
                ax.text(
                    x_positions[idx],
                    bottom[idx] + value / 2,
                    f"n={int(value)}",
                    ha="center",
                    va="center",
                    color=choose_text_color(color, disease, force_black_label),
                    fontweight="bold",
                    fontsize=7,
                )

        bottom += values

    y_offset = 0.5 if fixed_ylim is not None else max(float(totals.max()) * 0.015, 0.5)
    for idx, total in enumerate(totals):
        ax.text(
            x_positions[idx],
            total + y_offset,
            f"n={int(total)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#444444",
        )

    if fixed_ylim is not None:
        ax.set_ylim(*fixed_ylim)
    else:
        ax.set_ylim(0, float(totals.max()) * 1.15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, color="#E0E0E0")
    ax.set_axisbelow(True)

    ax.set_title(title, fontsize=18, pad=20, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=14, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=15)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(datasets, rotation=40, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)

    legend = ax.legend(
        title=r"$\mathbf{Disease\ Status}$",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=False,
        fontsize=10,
    )
    legend.get_title().set_fontsize(11)

    plt.tight_layout()

    fig.savefig(output_pdf, dpi=FIGURE_DPI, bbox_inches="tight", transparent=True)
    fig.savefig(output_png, dpi=FIGURE_DPI, bbox_inches="tight", transparent=True)

    print(f"Saved PDF: {output_pdf}")
    print(f"Saved PNG: {output_png}")

    plt.show()
    return fig, ax


# =============================================================================
# 5. Main workflow
# =============================================================================

def run_sample_composition_analysis():
    """Generate Supplementary Figure 1E-F and save related tables."""
    adata_csf = sc.read_h5ad(INPUT_CSF_H5AD)
    adata_pbmc = sc.read_h5ad(INPUT_PBMC_H5AD)

    csf_count_df, csf_matrix_df = build_sample_composition_table(adata_csf)
    pbmc_count_df, pbmc_matrix_df = build_sample_composition_table(adata_pbmc)

    csf_count_df.insert(0, "tissue", "CSF")
    pbmc_count_df.insert(0, "tissue", "PBMC")

    save_sample_composition_tables(
        count_df=csf_count_df,
        matrix_df=csf_matrix_df,
        prefix="SupplementaryFigure1E_CSF",
    )
    save_sample_composition_tables(
        count_df=pbmc_count_df,
        matrix_df=pbmc_matrix_df,
        prefix="SupplementaryFigure1F_PBMC",
    )

    combined_count_df = pd.concat([csf_count_df, pbmc_count_df], axis=0, ignore_index=True)
    combined_path = os.path.join(TABLE_DIR, "SupplementaryFigure1E_F_combined_sample_composition_counts.csv")
    combined_count_df.to_csv(combined_path, index=False)
    print(f"Saved combined count table: {combined_path}")

    all_diseases = list(csf_matrix_df.columns) + list(pbmc_matrix_df.columns)
    disease_palette = make_consistent_disease_palette(all_diseases)

    fig_csf, ax_csf = plot_dataset_sample_composition(
        matrix_df=csf_matrix_df,
        title="CSF Sample Composition",
        output_pdf=os.path.join(OUTPUT_DIR, "SupplementaryFigure1E_CSF_sample_composition.pdf"),
        output_png=os.path.join(OUTPUT_DIR, "SupplementaryFigure1E_CSF_sample_composition.png"),
        disease_palette=disease_palette,
    )

    fig_pbmc, ax_pbmc = plot_dataset_sample_composition(
        matrix_df=pbmc_matrix_df,
        title="PBMC Sample Composition",
        output_pdf=os.path.join(OUTPUT_DIR, "SupplementaryFigure1F_PBMC_sample_composition.pdf"),
        output_png=os.path.join(OUTPUT_DIR, "SupplementaryFigure1F_PBMC_sample_composition.png"),
        disease_palette=disease_palette,
    )

    return {
        "csf_count_df": csf_count_df,
        "csf_matrix_df": csf_matrix_df,
        "pbmc_count_df": pbmc_count_df,
        "pbmc_matrix_df": pbmc_matrix_df,
        "combined_count_df": combined_count_df,
        "fig_csf": fig_csf,
        "fig_pbmc": fig_pbmc,
        "ax_csf": ax_csf,
        "ax_pbmc": ax_pbmc,
    }


# %%
# =============================================================================
# 6. Run Supplementary Figure 1E-F generation
# =============================================================================

results = run_sample_composition_analysis()
