#!/usr/bin/env python3
"""Synchronize dataset colors and draw matched CSF/PBMC QC violins."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager


PROJECT = Path("/home/lishaogang/NEU2026_July")
MASTER = PROJECT / "final/h5ad/adata_final_Disease_corrected_min400_except_CART.h5ad"
CSF = PROJECT / "final/h5ad/figure_inputs/adata_final_csf_Disease_corrected_min400_except_CART.h5ad"
PBMC = PROJECT / "final/h5ad/figure_inputs/adata_final_pbmc_Disease_corrected_min400_except_CART.h5ad"
FIGURE2 = PROJECT / "final/Figures/Figure2"
QC_DIR = PROJECT / "final/Figures/Supplement_Figure1/Supplement_Figure1A_B"
QC_TABLE_DIR = QC_DIR / "tables"
ARIAL = Path("/home/lishaogang/.fonts/arial.ttf")
QC_KEYS = ["n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_ribo"]


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
    return np.asarray(element[:])


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
    sns.set_style("whitegrid")
    return family


def read_dataset_palette() -> tuple[list[str], dict[str, str], dict[str, str]]:
    with h5py.File(MASTER, "r") as handle:
        order = decode(handle["obs/dataset/categories"][:]).tolist()
        base_colors = decode(handle["uns/dataset_colors"][:]).tolist()
    base = dict(zip(order, base_colors))
    # Seaborn violinplot uses saturation=0.75 by default. These are the colors
    # actually visible in the existing CSF QC violin plot.
    visible = {
        dataset: mpl.colors.to_hex(sns.desaturate(base[dataset], 0.75))
        for dataset in order
    }
    return order, base, visible


def plot_gsm_umap(dataset_order: list[str], visible_palette: dict[str, str]) -> None:
    with h5py.File(MASTER, "r") as handle:
        gsm = read_element(handle["obs/orig.ident"])
        dataset = read_element(handle["obs/dataset"])
        coordinates = np.asarray(handle["obsm/X_umap"][:], dtype=np.float32)

    sample_map = pd.DataFrame({"GSM": gsm, "dataset": dataset}).drop_duplicates()
    conflicts = sample_map.groupby("GSM", observed=True)["dataset"].nunique()
    if (conflicts > 1).any():
        raise ValueError("At least one GSM maps to more than one dataset.")
    sample_map["color_hex"] = sample_map["dataset"].map(visible_palette)
    sample_map = sample_map.sort_values(["dataset", "GSM"])
    sample_map.to_csv(FIGURE2 / "UMAP_by_GSM_color_key.csv", index=False)

    present = [name for name in dataset_order if np.any(dataset == name)]
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    for name in present:
        mask = dataset == name
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            c=visible_palette[name],
            s=0.35,
            alpha=0.72,
            linewidths=0,
            rasterized=True,
            label=name,
        )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("UMAP by GSM")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        title="Dataset color assigned to GSM",
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        frameon=False,
        markerscale=5,
        fontsize=8,
        title_fontsize=9,
    )
    fig.tight_layout()
    stem = FIGURE2 / "UMAP_by_GSM_Arial_editable"
    fig.savefig(stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", transparent=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)


def calculate_raw_count_metrics(
    path: Path, n_cells: int, cache_prefix: str, chunk_size: int = 5000
) -> tuple[np.ndarray, np.ndarray]:
    ribo_cache = QC_TABLE_DIR / f"{cache_prefix}_pct_counts_ribo.npy"
    total_cache = QC_TABLE_DIR / f"{cache_prefix}_raw_total_counts.npy"
    if ribo_cache.is_file() and total_cache.is_file():
        cached_ribo = np.load(ribo_cache)
        cached_total = np.load(total_cache)
        if len(cached_ribo) == n_cells and len(cached_total) == n_cells:
            print(f"Using cached {cache_prefix} raw-count QC metrics: {QC_TABLE_DIR}")
            return cached_total, cached_ribo

    with h5py.File(path, "r") as handle:
        var = handle["var"]
        index_key = var.attrs.get("_index", "_index")
        if isinstance(index_key, bytes):
            index_key = index_key.decode("utf-8")
        genes = np.char.upper(decode(var[index_key][:]).astype(str))
        is_ribo = np.char.startswith(genes, "RPS") | np.char.startswith(genes, "RPL")
        counts = handle["layers/counts"]
        indptr = np.asarray(counts["indptr"][:], dtype=np.int64)
        raw_total = np.zeros(n_cells, dtype=np.float64)
        ribo_counts = np.zeros(n_cells, dtype=np.float64)

        for start in range(0, n_cells, chunk_size):
            stop = min(start + chunk_size, n_cells)
            pointer = indptr[start : stop + 1]
            left, right = int(pointer[0]), int(pointer[-1])
            indices = np.asarray(counts["indices"][left:right], dtype=np.int32)
            data = np.asarray(counts["data"][left:right], dtype=np.float64)
            row_ids = np.repeat(np.arange(stop - start), np.diff(pointer))
            mask = is_ribo[indices]
            raw_total[start:stop] = np.bincount(
                row_ids, weights=data, minlength=stop - start
            )
            ribo_counts[start:stop] = np.bincount(
                row_ids[mask], weights=data[mask], minlength=stop - start
            )
            if start == 0 or stop == n_cells or stop % 50000 == 0:
                print(f"Calculated raw-count QC: {stop:,}/{n_cells:,}", flush=True)

    pct_ribo = np.divide(
        ribo_counts * 100.0,
        raw_total,
        out=np.zeros_like(ribo_counts),
        where=raw_total > 0,
    )
    np.save(total_cache, raw_total)
    np.save(ribo_cache, pct_ribo)
    return raw_total, pct_ribo


def load_qc(path: Path, cache_prefix: str) -> pd.DataFrame:
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        frame = pd.DataFrame(
            {
                "dataset": read_element(obs["dataset"]),
                "n_genes_by_counts": read_element(obs["n_genes_by_counts"]),
                "total_counts": read_element(obs["total_counts"]),
                "pct_counts_mt": read_element(obs["pct_counts_mt"]),
            }
        )
    raw_total, pct_ribo = calculate_raw_count_metrics(
        path, len(frame), cache_prefix=cache_prefix
    )
    frame["total_counts"] = raw_total
    frame["pct_counts_ribo"] = pct_ribo
    return frame


def plot_qc(
    frame: pd.DataFrame,
    dataset_order: list[str],
    visible_palette: dict[str, str],
    figure_stem: str,
    table_prefix: str,
) -> None:
    present = [name for name in dataset_order if frame["dataset"].eq(name).any()]
    fig, axes = plt.subplots(1, 4, figsize=(32, 4.4))
    for ax, metric in zip(axes, QC_KEYS):
        sns.violinplot(
            data=frame,
            x="dataset",
            y=metric,
            hue="dataset",
            order=present,
            hue_order=present,
            palette=visible_palette,
            saturation=1,
            inner="box",
            linewidth=0.8,
            cut=0,
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    fig.tight_layout(w_pad=2.2)
    stem = QC_DIR / figure_stem
    fig.savefig(stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", transparent=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)

    summary = (
        frame.groupby("dataset", observed=True)
        .agg(
            n_cells=("dataset", "size"),
            median_n_genes_by_counts=("n_genes_by_counts", "median"),
            median_total_counts=("total_counts", "median"),
            median_pct_counts_mt=("pct_counts_mt", "median"),
            median_pct_counts_ribo=("pct_counts_ribo", "median"),
        )
        .reindex(present)
        .reset_index()
    )
    summary.to_csv(QC_TABLE_DIR / f"{table_prefix}_dataset_QC_summary.csv", index=False)


def save_palette_tables(
    dataset_order: list[str], base: dict[str, str], visible: dict[str, str]
) -> None:
    pd.DataFrame(
        {
            "dataset": dataset_order,
            "stored_color_hex": [base[name] for name in dataset_order],
            "CSF_violin_visible_color_hex": [visible[name] for name in dataset_order],
        }
    ).to_csv(QC_TABLE_DIR / "CSF_PBMC_dataset_color_key.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tissues",
        nargs="+",
        choices=("CSF", "PBMC"),
        default=("CSF", "PBMC"),
        help="QC violin plots to generate.",
    )
    parser.add_argument(
        "--skip-gsm",
        action="store_true",
        help="Do not regenerate the GSM-colored UMAP.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FIGURE2.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)
    QC_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    dataset_order, base_palette, visible_palette = read_dataset_palette()
    save_palette_tables(dataset_order, base_palette, visible_palette)
    if not args.skip_gsm:
        plot_gsm_umap(dataset_order, visible_palette)
        print(f"GSM UMAP: {FIGURE2}")
    if "CSF" in args.tissues:
        csf_qc = load_qc(CSF, "CSF")
        plot_qc(
            csf_qc,
            dataset_order,
            visible_palette,
            "SupplementaryFigure1A_CSF_QC_violin_by_dataset",
            "CSF",
        )
    if "PBMC" in args.tissues:
        pbmc_qc = load_qc(PBMC, "PBMC")
        plot_qc(
            pbmc_qc,
            dataset_order,
            visible_palette,
            "PBMC_QC_violin_by_dataset",
            "PBMC",
        )
    print(f"Matched CSF/PBMC QC violins: {QC_DIR}")


if __name__ == "__main__":
    main()
