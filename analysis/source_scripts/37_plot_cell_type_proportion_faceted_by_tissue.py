#!/usr/bin/env python3
"""Plot disease-level cell-type proportions in CSF/PBMC tissue facets."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


PROJECT = Path("/home/lishaogang/NEU2026_July")
INPUT_H5AD = PROJECT / "final/h5ad/adata_final_Disease_corrected_min400_except_CART.h5ad"
OUTPUT_DIR = PROJECT / "final/Figures/Cell_Type_Proportion_Faceted_by_Tissue"
TABLE_DIR = OUTPUT_DIR / "tables"
OUTPUT_STEM = (
    "Cell_Type_Proportion_StackedBar_Grouped_by_Disease_"
    "Faceted_by_Tissue_NoFrame_OneLegendColumn"
)
ARIAL = Path("/home/lishaogang/.fonts/arial.ttf")

TISSUE_ORDER = ["CSF", "PBMC"]
DISEASE_ORDER = [
    "Younger control", "Older control", "Unknown", "MCI/AD", "PD", "IIH", "MS",
    "Ab-mediated IDD", "VE", "Neuro-COVID", "CAR-T-DIPG", "NT1", "CIS", "RIS",
    "DLB", "PHN", "Uveitis", "NMO", "NT2",
]

# Preserve the palette used by the original stacked-bar figures.
CELL_ORDER = [
    "Naive B", "Memory B", "Plasma cell",
    "CD4 Naive", "CD8 Naive", "CD4 Memory", "CD8 Memory",
    "Treg", "CD4 CTL", "CD8 CTL", "MAIT", "γδT", "NKT",
    "NK bright", "Transit. NK", "NK dim", "ILC2",
    "cMonocytes", "ncMonocytes", "mdMac", "trMac", "infMac",
    "cDC2", "pDC", "ASDC",
]

CELL_COLORS = {
    "Naive B": "#d2c089", "Memory B": "#ddcc9f", "Plasma cell": "#e8d8b4",
    "CD4 Naive": "#a86c60", "CD8 Naive": "#6e8299", "CD4 Memory": "#bb8478",
    "CD8 Memory": "#8499b0", "Treg": "#ddb8ab", "CD4 CTL": "#cc9e90",
    "CD8 CTL": "#9baec4", "MAIT": "#a093b0", "γδT": "#8c7f9c",
    "NKT": "#b5a8c3", "NK bright": "#b98667", "Transit. NK": "#ddb497",
    "NK dim": "#cc9d7e", "ILC2": "#707070", "cMonocytes": "#7a997a",
    "ncMonocytes": "#6b876b", "mdMac": "#a4b9a4", "trMac": "#8fa98f",
    "infMac": "#b9c9b9", "cDC2": "#9383a1", "pDC": "#a97f95",
    "ASDC": "#c99faf",
}

FACET_COLORS = {"CSF": "#596FA8", "PBMC": "#4F8751"}
FACET_BACKGROUNDS = {"CSF": "#F5F6FA", "PBMC": "#F4F8F4"}


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def read_element(element: h5py.Group | h5py.Dataset) -> np.ndarray:
    if isinstance(element, h5py.Group):
        categories = decode(element["categories"][:])
        codes = np.asarray(element["codes"][:], dtype=np.int32)
        values = np.full(len(codes), "Unknown", dtype=object)
        valid = codes >= 0
        values[valid] = categories[codes[valid]]
        return values
    values = np.asarray(element[:])
    return decode(values) if values.dtype.kind in "OSU" else values


def setup_style() -> str:
    family = "Arial"
    if ARIAL.is_file():
        font_manager.fontManager.addfont(str(ARIAL))
        family = font_manager.FontProperties(fname=str(ARIAL)).get_name()
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.sans-serif": [family],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return family


def load_metadata() -> pd.DataFrame:
    with h5py.File(INPUT_H5AD, "r") as handle:
        obs = handle["obs"]
        frame = pd.DataFrame(
            {
                "tissue": read_element(obs["tissue"]),
                "Disease": read_element(obs["Disease"]),
                "cell_subtype_short": read_element(obs["cell_subtype_short"]),
            }
        )
    frame["tissue"] = frame["tissue"].astype(str).str.strip().replace(
        {"PB": "PBMC", "DLB": "PBMC"}
    )
    frame["Disease"] = frame["Disease"].astype(str).str.strip()
    frame["cell_subtype_short"] = (
        frame["cell_subtype_short"].astype(str).str.strip().replace("maMac", "mdMac")
    )
    return frame[
        frame["tissue"].isin(TISSUE_ORDER)
        & frame["Disease"].isin(DISEASE_ORDER)
        & frame["cell_subtype_short"].isin(CELL_ORDER)
    ].copy()


def build_table(metadata: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    present_diseases = {
        tissue: [
            disease for disease in DISEASE_ORDER
            if ((metadata["tissue"] == tissue) & (metadata["Disease"] == disease)).any()
        ]
        for tissue in TISSUE_ORDER
    }
    rows = []
    for tissue in TISSUE_ORDER:
        for disease in present_diseases[tissue]:
            subset = metadata[
                (metadata["tissue"] == tissue) & (metadata["Disease"] == disease)
            ]
            counts = subset["cell_subtype_short"].value_counts()
            total = int(len(subset))
            for cell_type in CELL_ORDER:
                count = int(counts.get(cell_type, 0))
                rows.append(
                    {
                        "tissue": tissue,
                        "Disease": disease,
                        "cell_subtype_short": cell_type,
                        "Count": count,
                        "Total_cells": total,
                        "Proportion": count / total if total else 0,
                        "Proportion_percent": count * 100 / total if total else 0,
                    }
                )
    return pd.DataFrame(rows), present_diseases


def plot_facets(table: pd.DataFrame, present_diseases: dict[str, list[str]]) -> None:
    width_ratios = [len(present_diseases[tissue]) for tissue in TISSUE_ORDER] + [5.2]
    fig = plt.figure(figsize=(22, 8.5))
    grid = fig.add_gridspec(
        1, 3, width_ratios=width_ratios, wspace=0.055,
        left=0.055, right=0.985, bottom=0.27, top=0.88,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(2)]
    legend_ax = fig.add_subplot(grid[0, 2])
    legend_ax.axis("off")

    for index, (ax, tissue) in enumerate(zip(axes, TISSUE_ORDER)):
        diseases = present_diseases[tissue]
        subset = table[table["tissue"] == tissue]
        matrix = subset.pivot(
            index="Disease", columns="cell_subtype_short", values="Proportion"
        ).reindex(index=diseases, columns=CELL_ORDER).fillna(0)
        totals = subset.groupby("Disease", observed=True)["Total_cells"].first()
        positions = np.arange(len(diseases))
        bottom = np.zeros(len(diseases))
        for cell_type in CELL_ORDER:
            values = matrix[cell_type].to_numpy()
            ax.bar(
                positions, values, bottom=bottom, width=0.76,
                color=CELL_COLORS[cell_type], edgecolor="none",
            )
            bottom += values

        ax.set_facecolor(FACET_BACKGROUNDS[tissue])
        ax.plot(
            [0, 1], [1.025, 1.025], transform=ax.transAxes,
            color=FACET_COLORS[tissue], linewidth=2.2, clip_on=False,
        )
        ax.text(
            0.5, 1.055, tissue, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=15, fontweight="bold",
            color=FACET_COLORS[tissue],
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"{disease}\n(n={int(totals[disease]):,})" for disease in diseases],
            rotation=52, ha="right", va="top", fontsize=7.3,
        )
        ax.set_xlim(-0.62, len(diseases) - 0.38)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if index == 0:
            ax.set_ylabel("Cell Type Proportion (%)", fontsize=12, fontweight="bold")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.spines["left"].set_visible(False)

    fig.supxlabel("Disease Group (number of cells)", y=0.035, fontsize=12, fontweight="bold")
    fig.suptitle(
        "Cell Type Proportion by Disease and Tissue",
        y=0.97, fontsize=17, fontweight="bold", color="#555555",
    )
    handles = [Patch(facecolor=CELL_COLORS[cell_type], edgecolor="none") for cell_type in CELL_ORDER]
    labels = ["gdT" if cell_type == "γδT" else cell_type for cell_type in CELL_ORDER]
    legend = legend_ax.legend(
        handles, labels, title="Cell Type", loc="center left", frameon=False,
        ncol=1, fontsize=8.5, title_fontsize=11,
        handlelength=1.0, handleheight=0.8, labelspacing=0.28,
    )
    legend.get_title().set_fontweight("bold")

    stem = OUTPUT_DIR / OUTPUT_STEM
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved faceted figure: {stem}.pdf/.png", flush=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    metadata = load_metadata()
    table, present_diseases = build_table(metadata)
    table.to_csv(TABLE_DIR / "Cell_Type_Counts_and_Proportions_by_Disease_and_Tissue.csv", index=False)
    pd.DataFrame(
        {"cell_subtype_short": CELL_ORDER, "color_hex": [CELL_COLORS[x] for x in CELL_ORDER]}
    ).to_csv(TABLE_DIR / "Cell_Type_previous_stackedbar_color_key.csv", index=False)
    plot_facets(table, present_diseases)
    print(
        "Facet groups: "
        + "; ".join(f"{tissue}={len(present_diseases[tissue])}" for tissue in TISSUE_ORDER)
    )


if __name__ == "__main__":
    main()
