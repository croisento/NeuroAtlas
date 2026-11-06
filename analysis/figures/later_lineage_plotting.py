from __future__ import annotations
"""Later lineage plotting functions from Code/32, with CLI-provided paths."""


import argparse

from pathlib import Path

import h5py

import matplotlib as mpl

import matplotlib.patheffects as pe

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from adjustText import adjust_text

from matplotlib import font_manager

SOFT_COLOR_MAP = {
    "Naive B": "#C96F6F",
    "Transitional B": "#D89A84",
    "Memory B": "#B85C68",
    "Plasma cell": "#B77AB5",
    "CD4 Naive": "#7DA7CF",
    "CD4 Memory": "#AEC9E4",
    "CD4 CTL": "#3E7C9D",
    "CD8 Naive": "#4F9A78",
    "MAIT": "#75B38A",
    "CD8 Memory": "#98C970",
    "CD8 CTL": "#C3D894",
    "\u03b3\u03b4T": "#A6B85A",
    "NKT": "#BFB552",
    "Treg": "#3F6BA7",
    "Transit. NK": "#A995C1",
    "NK dim": "#C99235",
    "NK bright": "#D7BE66",
    "ILC2": "#DDD36A",
    "cMonocytes": "#C97891",
    "ncMonocytes": "#9E5F73",
    "trMac": "#D6A196",
    "infMac": "#C17DB4",
    "mdMac": "#9568B1",
    "cDC2": "#B46AB0",
    "pDC": "#7F8CC1",
    "ASDC": "#B79BD0",
}

GLOBAL_ORDER = [
    "Naive B",
    "Memory B",
    "Plasma cell",
    "CD4 Naive",
    "CD8 Naive",
    "CD4 Memory",
    "CD8 Memory",
    "Treg",
    "CD4 CTL",
    "CD8 CTL",
    "MAIT",
    "\u03b3\u03b4T",
    "NKT",
    "NK bright",
    "Transit. NK",
    "NK dim",
    "ILC2",
    "cMonocytes",
    "ncMonocytes",
    "mdMac",
    "trMac",
    "infMac",
    "cDC2",
    "pDC",
    "ASDC",
]

def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )

def read_categorical(group: h5py.Group) -> np.ndarray:
    categories = decode(group["categories"][:])
    codes = np.asarray(group["codes"][:], dtype=np.int32)
    values = np.full(len(codes), "Unknown", dtype=object)
    valid = codes >= 0
    values[valid] = categories[codes[valid]]
    return values

def setup_style() -> str:
    family = "Arial"
    if ARIAL_PATH.is_file():
        font_manager.fontManager.addfont(str(ARIAL_PATH))
        family = font_manager.FontProperties(fname=str(ARIAL_PATH)).get_name()
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.sans-serif": [family],
            "font.weight": "regular",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 16,
            "axes.labelsize": 10,
        }
    )
    return family

def load_existing_umap(path: Path, preferred_basis: str) -> tuple[np.ndarray, np.ndarray, str]:
    with h5py.File(path, "r") as handle:
        labels = read_categorical(handle["obs/cell_subtype_short"])
        available = set(handle["obsm"].keys())
        basis = preferred_basis if preferred_basis in available else "X_umap"
        if basis not in available:
            raise KeyError(f"No existing UMAP coordinates found in {path}")
        coordinates = np.asarray(handle[f"obsm/{basis}"][:], dtype=np.float32)
    return coordinates, labels, basis

def get_label_positions(coordinates: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    positions = {}
    for label in np.unique(labels):
        points = coordinates[labels == label]
        median = np.median(points, axis=0)
        positions[label] = points[np.argmin(np.sum((points - median) ** 2, axis=1))]
    return positions

def add_axis_arrows(ax: plt.Axes, family: str) -> None:
    ax.annotate(
        "", xy=(0.20, 0.06), xytext=(0.06, 0.06), xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "black"},
        annotation_clip=False,
        zorder=20,
    )
    ax.annotate(
        "", xy=(0.06, 0.20), xytext=(0.06, 0.06), xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "black"},
        annotation_clip=False,
        zorder=20,
    )
    ax.text(0.13, 0.025, "UMAP1", transform=ax.transAxes, ha="center", va="center",
            fontsize=9, fontfamily=family)
    ax.text(0.018, 0.13, "UMAP2", transform=ax.transAxes, ha="center", va="center",
            rotation=90, fontsize=9, fontfamily=family)

def plot_lineage(name: str, config: dict, family: str, dpi: int) -> list[dict]:
    coordinates, labels, basis = load_existing_umap(config["input"], config["preferred_basis"])
    order = [cell_type for cell_type in GLOBAL_ORDER if cell_type in config["types"]]
    keep = np.isin(labels, order)
    coordinates = coordinates[keep]
    labels = labels[keep]
    present_order = [cell_type for cell_type in order if np.any(labels == cell_type)]
    label_positions = get_label_positions(coordinates, labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    for cell_type in present_order:
        mask = labels == cell_type
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=3,
            alpha=0.75,
            c=SOFT_COLOR_MAP[cell_type],
            linewidths=0,
            rasterized=True,
        )

    # Leave room for adjusted labels near the outermost clusters and axis arrows.
    ax.margins(x=0.08, y=0.10)

    texts = []
    for cell_type in present_order:
        x_pos, y_pos = label_positions[cell_type]
        texts.append(
            ax.text(
                x_pos,
                y_pos,
                cell_type,
                fontsize=9,
                fontfamily=family,
                fontweight="normal",
                ha="center",
                va="center",
                color="black",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            )
        )
    adjust_text(
        texts,
        ax=ax,
        expand_text=(1.10, 1.18),
        force_text=(0.22, 0.40),
        arrowprops={"arrowstyle": "-", "lw": 0.65, "color": "#444444", "alpha": 0.40},
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_axis_arrows(ax, family)

    output = OUTPUT_DIR / config["output"]
    fig.savefig(
        output.with_suffix(".pdf"),
        dpi=dpi,
        bbox_inches="tight",
        transparent=True,
    )
    fig.savefig(
        output.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)
    print(f"{name}: {len(labels):,} cells; basis={basis}; order={present_order}")

    return [
        {
            "lineage": name,
            "plot_order": index + 1,
            "cell_subtype_short": cell_type,
            "color_hex": SOFT_COLOR_MAP[cell_type],
            "cell_count": int(np.sum(labels == cell_type)),
            "existing_umap_basis": basis,
        }
        for index, cell_type in enumerate(present_order)
    ]
OUTPUT_DIR = Path(".")
ARIAL_PATH = Path("/home/lishaogang/.fonts/arial.ttf")
