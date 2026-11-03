#!/usr/bin/env python3
"""Update disease stacked bars and metadata UMAPs without recomputing UMAP."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter


PROJECT = Path("/home/lishaogang/NEU2026_July")
MASTER = PROJECT / "final/h5ad/adata_final_Disease_corrected_min400_except_CART.h5ad"
FIGURE2 = PROJECT / "final/Figures/Figure2"
FIGURE3 = PROJECT / "final/Figures/Figure3_and_Supplement Figure3"
DISEASE_COUNTS = FIGURE3 / "Cell_Type_Counts_by_Disease.csv"
SAMPLE_COMPOSITION = (
    PROJECT
    / "final/Figures/Supplement_Figure1/Supplement_Figure1E_F/tables/"
    "SupplementaryFigure1E_F_combined_sample_composition_counts.csv"
)
ARIAL = Path("/home/lishaogang/.fonts/arial.ttf")

CELL_ORDER = [
    "Naive B", "Memory B", "Plasma cell",
    "CD4 Naive", "CD8 Naive", "CD4 Memory", "CD8 Memory",
    "Treg", "CD4 CTL", "CD8 CTL", "MAIT", "γδT", "NKT",
    "NK bright", "Transit. NK", "NK dim", "ILC2",
    "cMonocytes", "ncMonocytes", "mdMac", "trMac", "infMac",
    "cDC2", "pDC", "ASDC",
]

CELL_COLORS = {
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

STACKED_DISEASE_ORDER = [
    "Younger control", "Older control", "Unknown", "MCI/AD", "PD", "IIH", "MS"
]

SAMPLE_COMPOSITION_DISEASE_ORDER = [
    "Younger control", "Older control", "Unknown", "MCI/AD", "PD", "IIH", "MS",
    "Ab-mediated IDD", "VE", "Neuro-COVID", "CAR-T-DIPG", "NT1", "CIS", "RIS",
    "DLB", "PHN", "Uveitis",
]

GENDER_ORDER = ["Female", "Male", "Unknown"]
GENDER_COLORS = {"Female": "#D46A92", "Male": "#4C78A8", "Unknown": "#8A8A8A"}


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


def read_master_umap() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(MASTER, "r") as handle:
        coordinates = np.asarray(handle["obsm/X_umap"][:], dtype=np.float32)
        disease = read_element(handle["obs/Disease"]).astype(str)
        gender = read_element(handle["obs/gender"]).astype(str)
    if not (len(coordinates) == len(disease) == len(gender)):
        raise ValueError("Master UMAP and metadata lengths differ")
    return coordinates, disease, gender


def sample_composition_disease_palette() -> tuple[list[str], dict[str, str]]:
    composition = pd.read_csv(SAMPLE_COMPOSITION)
    present = composition["Disease"].astype(str).drop_duplicates().tolist()
    order = [label for label in SAMPLE_COMPOSITION_DISEASE_ORDER if label in present]
    order += sorted(set(present) - set(order))
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(order), 1)))
    return order, {
        label: mpl.colors.to_hex(color) for label, color in zip(order, colors)
    }


def plot_metadata_umap(
    coordinates: np.ndarray,
    labels: np.ndarray,
    order: list[str],
    colors: dict[str, str],
    field: str,
) -> None:
    present = [label for label in order if np.any(labels == label)]
    extras = sorted(set(labels) - set(present))
    if extras:
        raise ValueError(f"Unexpected {field} values: {extras}")

    fig, ax = plt.subplots(figsize=(10, 8))
    for label in present:
        mask = labels == label
        ax.scatter(
            coordinates[mask, 0], coordinates[mask, 1],
            s=0.8, c=colors[label], alpha=0.72, linewidths=0,
            rasterized=True, label=label,
        )
    ax.set_title(f"UMAP by {field}", fontsize=16)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        bbox_to_anchor=(1.01, 0.5), loc="center left", frameon=False,
        markerscale=5, fontsize=8,
    )
    fig.tight_layout()
    stem = FIGURE2 / f"UMAP_by_{field}_Arial_editable"
    fig.savefig(stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", transparent=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)
    pd.DataFrame(
        {field: present, "color_hex": [colors[label] for label in present]}
    ).to_csv(FIGURE2 / f"UMAP_by_{field}_color_key.csv", index=False)
    print(f"Saved {field} UMAP: {stem}.pdf/.png", flush=True)


def plot_disease_stacked_bar() -> None:
    counts = pd.read_csv(DISEASE_COUNTS)
    counts["cell_subtype_short"] = counts["cell_subtype_short"].replace("maMac", "mdMac")
    counts = counts[
        counts["Disease"].isin(STACKED_DISEASE_ORDER)
        & counts["cell_subtype_short"].isin(CELL_ORDER)
    ].copy()
    grouped = counts.groupby(["Disease", "cell_subtype_short"], observed=True)["Count"].sum()
    full_index = pd.MultiIndex.from_product(
        [STACKED_DISEASE_ORDER, CELL_ORDER], names=["Disease", "cell_subtype_short"]
    )
    plot_data = grouped.reindex(full_index, fill_value=0).rename("Count").reset_index()
    totals = plot_data.groupby("Disease", observed=True)["Count"].sum().reindex(
        STACKED_DISEASE_ORDER
    )
    plot_data["Proportion"] = plot_data["Count"] / plot_data["Disease"].map(totals)
    matrix = plot_data.pivot(
        index="Disease", columns="cell_subtype_short", values="Proportion"
    ).reindex(index=STACKED_DISEASE_ORDER, columns=CELL_ORDER).fillna(0)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    positions = np.arange(len(STACKED_DISEASE_ORDER))
    bottom = np.zeros(len(positions))
    for cell_type in CELL_ORDER:
        values = matrix[cell_type].to_numpy()
        ax.bar(
            positions, values, bottom=bottom, width=0.7,
            color=CELL_COLORS[cell_type], edgecolor="none",
            label="gdT" if cell_type == "γδT" else cell_type,
        )
        bottom += values

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            f"{disease}\n(n={int(totals[disease]):,} cells)"
            for disease in STACKED_DISEASE_ORDER
        ],
        rotation=45, ha="right", va="top", fontweight="bold", color="black",
    )
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Disease Group", fontweight="bold", color="black")
    ax.set_ylabel("Cell Type Proportion (%)", fontweight="bold", color="black")
    ax.grid(False)
    legend = ax.legend(
        title="Cell Type", loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=False, ncol=1, fontsize=8.2, title_fontsize=11,
        handlelength=1.0, handleheight=0.8, borderaxespad=0, labelspacing=0.28,
    )
    legend.get_title().set_fontweight("bold")
    fig.subplots_adjust(left=0.12, right=0.74, bottom=0.22, top=0.95)
    stem = FIGURE3 / "Cell_Type_Proportion_StackedBar_Grouped_NoFrame_OneLegendColumn"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)
    pd.DataFrame(
        {"cell_subtype_short": CELL_ORDER, "color_hex": [CELL_COLORS[x] for x in CELL_ORDER]}
    ).to_csv(FIGURE3 / "Cell_Type_Proportion_color_key.csv", index=False)
    print(f"Saved disease stacked bar: {stem}.pdf/.png", flush=True)


def main() -> None:
    FIGURE2.mkdir(parents=True, exist_ok=True)
    FIGURE3.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_disease_stacked_bar()
    coordinates, disease, gender = read_master_umap()
    disease_order, disease_colors = sample_composition_disease_palette()
    plot_metadata_umap(coordinates, disease, disease_order, disease_colors, "Disease")
    plot_metadata_umap(coordinates, gender, GENDER_ORDER, GENDER_COLORS, "Gender")
    print("Existing X_umap was used; no dimensional reduction was recomputed.")


if __name__ == "__main__":
    main()
