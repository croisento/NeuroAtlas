#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare raw-count AnnData objects from multiple GEO single-cell datasets.

This script merges the dataset-specific preprocessing scripts into one
publication-ready workflow. Each dataset is processed by an independent function,
which makes the script easier to review, rerun, or partially execute.

Primary outputs:
    - GSE133028_raw_count.h5ad
    - GSE134577_raw_count.h5ad
    - GSE138266_raw_count.h5ad
    - GSE138266_raw_count_rmlowquality.h5ad
    - GSE141578_raw_count.h5ad
    - GSE163005_raw_count.h5ad
    - GSE186802_raw_count_subset.h5ad
    - GSE194078_raw_count.h5ad
    - GSE200164_raw_count.h5ad
    - GSE243905_raw_count.h5ad
    - GSE244499_raw_count.h5ad
    - GSE247134_raw_count.h5ad
"""

from __future__ import annotations

import gc
import gzip
import os
import shutil
import tarfile
import tempfile
import warnings
from pathlib import Path
from typing import Callable, Optional

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

try:
    import GEOparse
except ImportError:
    GEOparse = None
os.chdir("/home/lishaogang/NEU2026_July")


# =============================================================================
# 1. Global configuration
# =============================================================================

warnings.filterwarnings("ignore")

BASE_DIR = Path("/home/lishaogang/NEU2026_July/GEO_data")
os.chdir(BASE_DIR)

# Set this list to control which datasets are processed.
RUN_DATASETS = [
    "GSE133028",
    "GSE134577",
    "GSE138266",
    "GSE141578",
    "GSE163005",
    "GSE186802",
    "GSE194078",
    "GSE200164",
    "GSE243905",
    "GSE244499",
    "GSE247134",
]

# Expected files used by the downstream dataset-merging script.
TARGET_H5AD_FILES = [
    "GSE133028_raw_count.h5ad",
    "GSE134577_raw_count.h5ad",
    "GSE138266_raw_count_rmlowquality.h5ad",
    "GSE141578_raw_count.h5ad",
    "GSE163005_raw_count.h5ad",
    "GSE186802_raw_count_subset.h5ad",
    "GSE194078_raw_count.h5ad",
    "GSE200164_raw_count.h5ad",
    "GSE243905_raw_count.h5ad",
    "GSE244499_raw_count.h5ad",
    "GSE247134_raw_count.h5ad",
]

# The original GSE138266 script included additional QC figures.
# Set this to True if these figures are needed.
SAVE_GSE138266_QC_PLOTS = False


# =============================================================================
# 2. Shared helper functions
# =============================================================================

def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def extract_tar(tar_path: Path, extract_dir: Path, mode: str = "r") -> None:
    """Extract a tar or tar.gz archive into the target directory."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, mode) as tar:
        tar.extractall(path=extract_dir)


def extract_gzip(gz_path: Path, output_path: Path) -> None:
    """Extract a gzipped file to the specified output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(gz_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def safe_copy_and_compress(src: Path, dst_name: str, dst_dir: Path) -> None:
    """Copy a file and ensure that the destination is gzipped."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / dst_name

    if str(src).endswith(".gz"):
        shutil.copy(src, dst_path)
    else:
        with open(src, "rb") as f_in:
            with gzip.open(dst_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)


def print_adata_summary(adata_obj: ad.AnnData, dataset_name: str) -> None:
    """Print the basic shape of an AnnData object."""
    print(f"\n{dataset_name} completed.")
    print(f"Number of cells: {adata_obj.n_obs}")
    print(f"Number of genes : {adata_obj.n_vars}")


#def save_adata(adata_obj: ad.AnnData, output_path: Path) -> None:
#    """Save an AnnData object as an h5ad file."""
#    output_path.parent.mkdir(parents=True, exist_ok=True)
#    adata_obj.write_h5ad(output_path)
#    print(f"Saved: {output_path}")


def calculate_mito_qc(adata_obj: ad.AnnData) -> ad.AnnData:
    """Calculate mitochondrial gene percentage and standard QC metrics."""
    adata_obj.var["mt"] = adata_obj.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata_obj,
        qc_vars=["mt"],
        inplace=True
    )
    return adata_obj


def clear_directory(directory: Path) -> None:
    """Remove all files and subdirectories from a directory."""
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        return

    for item in directory.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def standardize_10x_archives(
    raw_dir: Path,
    std_dir: Path,
    temp_dir: Path,
    prefer_filtered: bool = True
) -> list[str]:
    """
    Standardize mixed GEO 10x archives into one folder per GSM sample.

    The standardized structure is:
        standardized_10x/GSMxxxx/matrix.mtx.gz
        standardized_10x/GSMxxxx/features.tsv.gz or genes.tsv.gz
        standardized_10x/GSMxxxx/barcodes.tsv.gz
    """
    print("Initializing standardized 10x extraction folders...")

    for directory in [std_dir, temp_dir]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    all_files = os.listdir(raw_dir)
    gsm_ids = sorted({f.split("_")[0] for f in all_files if f.startswith("GSM")})
    print(f"Detected {len(gsm_ids)} GSM samples for standardized extraction.")

    for gsm_id in gsm_ids:
        sample_dir = std_dir / gsm_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        clear_directory(temp_dir)

        # Strategy A: use loose matrix/features/barcodes files if they are present.
        loose_matrix = sorted(raw_dir.glob(f"{gsm_id}_*matrix.mtx.gz"))
        loose_features = sorted(raw_dir.glob(f"{gsm_id}_*features.tsv.gz"))
        loose_genes = sorted(raw_dir.glob(f"{gsm_id}_*genes.tsv.gz"))
        loose_barcodes = sorted(raw_dir.glob(f"{gsm_id}_*barcodes.tsv.gz"))

        loose_feature_file = loose_features[0] if loose_features else loose_genes[0] if loose_genes else None

        if loose_matrix and loose_feature_file and loose_barcodes:
            shutil.copy(loose_matrix[0], sample_dir / "matrix.mtx.gz")
            feature_dst = "features.tsv.gz" if "features" in loose_feature_file.name else "genes.tsv.gz"
            shutil.copy(loose_feature_file, sample_dir / feature_dst)
            shutil.copy(loose_barcodes[0], sample_dir / "barcodes.tsv.gz")
            continue

        # Strategy B: extract sample-level tar.gz archives.
        target_tars: list[Path] = []
        if prefer_filtered:
            target_tars.extend(sorted(raw_dir.glob(f"{gsm_id}_*filtered*.tar.gz")))

        generic_tars = [
            path for path in sorted(raw_dir.glob(f"{gsm_id}_*.tar.gz"))
            if "raw" not in path.name.lower()
        ]
        target_tars.extend([path for path in generic_tars if path not in target_tars])

        if not target_tars:
            print(f"No usable 10x archive found for {gsm_id}; skipped.")
            continue

        extracted = False

        for target_tar in target_tars:
            try:
                clear_directory(temp_dir)
                with tarfile.open(target_tar, "r:gz") as tar:
                    tar.extractall(path=temp_dir)

                matrix_file = None
                feature_file = None
                barcode_file = None

                for root, _, files in os.walk(temp_dir):
                    for file_name in files:
                        path = Path(root) / file_name
                        if "matrix.mtx" in file_name:
                            matrix_file = path
                        elif "features.tsv" in file_name or "genes.tsv" in file_name:
                            feature_file = path
                        elif "barcodes.tsv" in file_name:
                            barcode_file = path

                if matrix_file and feature_file and barcode_file:
                    safe_copy_and_compress(matrix_file, "matrix.mtx.gz", sample_dir)
                    feature_dst = "features.tsv.gz" if "features" in feature_file.name else "genes.tsv.gz"
                    safe_copy_and_compress(feature_file, feature_dst, sample_dir)
                    safe_copy_and_compress(barcode_file, "barcodes.tsv.gz", sample_dir)
                    extracted = True
                    break

            except Exception as error:
                print(f"Failed to extract {target_tar.name} for {gsm_id}: {error}")

        if not extracted:
            print(f"Failed to locate complete 10x files for {gsm_id}.")

    print("Standardized 10x extraction completed.")
    return gsm_ids


def read_standardized_10x_samples(
    raw_dir: Path,
    std_dir: Path,
    dataset_name: str,
    metadata_callback: Optional[Callable[[str, str], dict[str, str]]] = None,
) -> ad.AnnData:
    """Read standardized 10x sample folders and concatenate them into one AnnData object."""
    all_files = os.listdir(raw_dir)
    gsm_ids = sorted([d.name for d in std_dir.iterdir() if d.is_dir() and d.name.startswith("GSM")])
    adata_dict = {}

    print(f"Reading {len(gsm_ids)} standardized samples for {dataset_name}...")

    for gsm_id in gsm_ids:
        sample_dir = std_dir / gsm_id
        mtx_path = sample_dir / "matrix.mtx.gz"
        barcode_path = sample_dir / "barcodes.tsv.gz"

        if (sample_dir / "features.tsv.gz").exists():
            feature_path = sample_dir / "features.tsv.gz"
        elif (sample_dir / "genes.tsv.gz").exists():
            feature_path = sample_dir / "genes.tsv.gz"
        else:
            feature_path = None

        if not (mtx_path.exists() and barcode_path.exists() and feature_path is not None):
            print(f"Incomplete standardized files for {gsm_id}; skipped.")
            continue

        try:
            adata_obj = sc.read_mtx(mtx_path)
            barcodes = pd.read_csv(barcode_path, header=None, sep="\t")
            genes = pd.read_csv(feature_path, header=None, sep="\t")

            # Validate matrix orientation and transpose when needed.
            if adata_obj.shape[0] == len(barcodes) and adata_obj.shape[1] == len(genes):
                pass
            elif adata_obj.shape[1] == len(barcodes) and adata_obj.shape[0] == len(genes):
                adata_obj = adata_obj.T
            else:
                raise ValueError(
                    f"Matrix shape {adata_obj.shape} does not match "
                    f"barcodes ({len(barcodes)}) and genes ({len(genes)})."
                )

            # Assign cell and gene names.
            adata_obj.obs_names = barcodes[0].astype(str).values
            gene_col = 1 if genes.shape[1] >= 2 else 0
            adata_obj.var_names = genes[gene_col].astype(str).values
            adata_obj.var_names_make_unique()

            original_name_candidates = [f for f in all_files if f.startswith(gsm_id)]
            original_name = original_name_candidates[0] if original_name_candidates else gsm_id

            if metadata_callback is None:
                tissue = "CSF" if "CSF" in original_name else "PBMC" if "PBMC" in original_name else "Unknown"
                metadata = {
                    "GSM_ID": gsm_id,
                    "orig.ident": gsm_id,
                    "tissue": tissue,
                }
            else:
                metadata = metadata_callback(gsm_id, original_name)

            for key, value in metadata.items():
                adata_obj.obs[key] = value

            adata_obj.obs_names = f"{gsm_id}_" + adata_obj.obs_names.astype(str)
            adata_obj.obs["cell_id"] = adata_obj.obs_names

            adata_dict[gsm_id] = adata_obj
            print(f"Loaded: {gsm_id} ({adata_obj.n_obs} cells)")

        except Exception as error:
            print(f"Failed to read standardized sample {gsm_id}: {error}")

    if not adata_dict:
        raise RuntimeError(f"No samples were successfully loaded for {dataset_name}.")

    adata_combined = ad.concat(adata_dict, label="batch")
    adata_combined.obs_names_make_unique()
    return adata_combined


def validate_target_files(base_dir: Path = BASE_DIR) -> None:
    """Report whether all expected output h5ad files are present."""
    print_section("Validating expected output files")
    missing_files = []

    for file_name in TARGET_H5AD_FILES:
        file_path = base_dir / file_name
        if file_path.exists():
            print(f"Found:   {file_name}")
        else:
            print(f"Missing: {file_name}")
            missing_files.append(file_name)

    if missing_files:
        print("\nThe following expected output files are missing:")
        for file_name in missing_files:
            print(f"  - {file_name}")
    else:
        print("\nAll expected output files are present.")


# =============================================================================
# 3. GSE133028
# =============================================================================

def process_gse133028(base_dir: Path = BASE_DIR) -> ad.AnnData:
    """Process GSE133028 from rawCounts.csv.gz expression matrices."""
    print_section("Processing GSE133028")

    data_dir = base_dir / "GSE133028_RAW"
    output_path = base_dir / "GSE133028_raw_count.h5ad"
    legacy_output_path = base_dir / "GSE133028_merged_ALL.h5ad"

    count_files = sorted([f for f in os.listdir(data_dir) if "rawCounts.csv.gz" in f])
    print(f"Detected {len(count_files)} raw count matrix files.")

    adata_dict = {}

    for file_name in count_files:
        file_path = data_dir / file_name
        gsm_id = file_name.split("_")[0]
        tissue = "CSF" if "_CSF" in file_name else "PB" if "_PB" in file_name else "Unknown"

        try:
            sep = "," if file_name.endswith(".csv.gz") else "\t"
            df = pd.read_csv(file_path, sep=sep, index_col=0)

            # Convert genes x cells into the Scanpy convention: cells x genes.
            adata_obj = ad.AnnData(df.T)

            adata_obj.obs["GSM_ID"] = gsm_id
            adata_obj.obs["orig.ident"] = gsm_id
            adata_obj.obs["tissue"] = tissue

            # Prefix cell barcodes with GSM accession to avoid cross-sample duplicates.
            adata_obj.obs_names = f"{gsm_id}_" + adata_obj.obs_names.astype(str)
            adata_obj.obs["cell_id"] = adata_obj.obs_names

            adata_dict[gsm_id] = adata_obj
            print(f"Loaded: {file_name} ({tissue}, {adata_obj.n_obs} cells, {adata_obj.n_vars} genes)")

        except Exception as error:
            print(f"Failed to load {file_name}: {error}")

    if not adata_dict:
        raise RuntimeError("No GSE133028 samples were loaded.")

    print("\nConcatenating all GSE133028 samples...")
    adata_combined = ad.concat(adata_dict, label="batch")
    adata_combined.obs_names_make_unique()

    save_adata(adata_combined, output_path)

    # Save the original legacy filename as well for backward compatibility.
    save_adata(adata_combined, legacy_output_path)

    print_adata_summary(adata_combined, "GSE133028")
    return adata_combined


# =============================================================================
# 4. GSE134577
# =============================================================================

def process_gse134577(base_dir: Path = BASE_DIR) -> ad.AnnData:
    """Process GSE134577 from mixed nested 10x MTX archives."""
    print_section("Processing GSE134577")

    raw_dir = base_dir / "GSE134577_RAW"
    std_dir = raw_dir / "standardized_10x"
    temp_dir = raw_dir / "temp_extract"
    output_path = base_dir / "GSE134577_raw_count.h5ad"

    # Extract the top-level archive when the raw folder has not been prepared yet.
    top_level_tar = base_dir / "GSE134577_RAW.tar"
    if top_level_tar.exists() and (not raw_dir.exists() or not any(raw_dir.iterdir())):
        raw_dir.mkdir(parents=True, exist_ok=True)
        extract_tar(top_level_tar, raw_dir, mode="r")

    # Optionally extract the GEO series matrix file if it is present.
    series_gz_path = base_dir / "GSE134577_series_matrix.txt.gz"
    if series_gz_path.exists():
        extract_gzip(series_gz_path, raw_dir / "GSE134577_series_matrix.txt")

    def metadata_callback(gsm_id: str, original_name: str) -> dict[str, str]:
        tissue = "CSF" if "CSF" in original_name else "PBMC" if "PBMC" in original_name else "Unknown"
        return {
            "GSM_ID": gsm_id,
            "GSM": gsm_id,
            "orig.ident": gsm_id,
            "origin.ident": gsm_id,
            "tissue": tissue,
        }

    standardize_10x_archives(raw_dir, std_dir, temp_dir, prefer_filtered=False)
    adata_combined = read_standardized_10x_samples(
        raw_dir=raw_dir,
        std_dir=std_dir,
        dataset_name="GSE134577",
        metadata_callback=metadata_callback,
    )

    save_adata(adata_combined, output_path)
    print_adata_summary(adata_combined, "GSE134577")

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    gc.collect()
    return adata_combined


# =============================================================================
# 5. GSE138266
# =============================================================================

def process_gse138266(
    base_dir: Path = BASE_DIR,
    save_qc_plots: bool = SAVE_GSE138266_QC_PLOTS,
) -> ad.AnnData:
    """Process GSE138266 from gzipped 10x MTX files and remove low-quality cells."""
    print_section("Processing GSE138266")

    tar_path = base_dir / "GSE138266_RAW.tar"
    extract_dir = base_dir / "GSE138266_RAW"

    raw_output_path = base_dir / "GSE138266_raw_count.h5ad"
    filtered_output_path = base_dir / "GSE138266_raw_count_rmlowquality.h5ad"

    if tar_path.exists():
        extract_tar(tar_path, extract_dir, mode="r")

    samples = {}

    for file_name in os.listdir(extract_dir):
        if not file_name.endswith(".gz"):
            continue

        parts = file_name.split("_")
        if len(parts) < 4:
            continue

        gsm_id = parts[0]
        sample_id = parts[1]
        source = parts[3]
        key = f"{gsm_id}_{sample_id}_{source}"

        if key not in samples:
            samples[key] = {
                "barcodes": None,
                "genes": None,
                "matrix": None,
                "sample_id": sample_id,
                "source": source,
                "GSM": gsm_id,
            }

        file_path = extract_dir / file_name

        if "barcodes" in file_name:
            samples[key]["barcodes"] = file_path
        elif "genes" in file_name or "features" in file_name:
            samples[key]["genes"] = file_path
        elif "matrix" in file_name:
            samples[key]["matrix"] = file_path

    print("Number of detected samples:", len(samples))

    adata_list = []

    for uid, info in samples.items():
        print(f"\nReading sample: {uid}")

        if not all([info["barcodes"], info["genes"], info["matrix"]]):
            print("Skipped because one or more required 10x files are missing.")
            continue

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            extract_gzip(info["barcodes"], tmp_dir / "barcodes.tsv")
            extract_gzip(info["genes"], tmp_dir / "genes.tsv")
            extract_gzip(info["matrix"], tmp_dir / "matrix.mtx")

            adata_obj = sc.read_10x_mtx(
                tmp_dir,
                var_names="gene_symbols",
                cache=False,
            )

            adata_obj.obs["sample_id"] = info["sample_id"]
            adata_obj.obs["source"] = info["source"]
            adata_obj.obs["orig.ident"] = info["GSM"]

            adata_list.append(adata_obj)
            print(f"Loaded successfully: {adata_obj.n_obs} cells")

    if not adata_list:
        raise RuntimeError("No GSE138266 samples were loaded.")

    print("\nConcatenating GSE138266 samples...")
    adata_gse138266 = sc.concat(adata_list, index_unique="-")
    adata_gse138266.obs["cell_id"] = adata_gse138266.obs.index

    print_adata_summary(adata_gse138266, "GSE138266 raw")
    print("\nSource distribution:")
    print(adata_gse138266.obs["source"].value_counts())

    save_adata(adata_gse138266, raw_output_path)

    calculate_mito_qc(adata_gse138266)
    print("\nAvailable obs columns after QC calculation:")
    print(adata_gse138266.obs.columns.tolist())

    if save_qc_plots:
        save_gse138266_qc_plots(adata_gse138266, base_dir)

    adata_filtered = adata_gse138266[
        (adata_gse138266.obs["n_genes_by_counts"] > 400) &
        (adata_gse138266.obs["pct_counts_mt"] < 20)
    ].copy()

    print("\nQC filtering summary:")
    print("Cells before filtering:", adata_gse138266.n_obs)
    print("Cells after filtering :", adata_filtered.n_obs)

    save_adata(adata_filtered, filtered_output_path)
    return adata_gse138266


def save_gse138266_qc_plots(adata_obj: ad.AnnData, base_dir: Path = BASE_DIR) -> None:
    """Save the QC plots that were included in the original GSE138266 workflow."""
    print("\nSaving GSE138266 QC plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    sc.pl.violin(adata_obj, "n_genes_by_counts", ax=axes[0, 0], show=False)
    axes[0, 0].set_title("n_genes_by_counts")
    axes[0, 0].set_ylim(0, 4000)

    sc.pl.violin(adata_obj, "total_counts", ax=axes[0, 1], show=False)
    axes[0, 1].set_title("total_counts")
    axes[0, 1].set_ylim(0, 20000)

    sc.pl.violin(adata_obj, "pct_counts_mt", ax=axes[1, 0], show=False)
    axes[1, 0].set_title("pct_counts_mt")

    sc.pl.scatter(
        adata_obj,
        x="n_genes_by_counts",
        y="pct_counts_mt",
        ax=axes[1, 1],
        show=False,
    )
    axes[1, 1].set_title("n_genes_by_counts vs pct_counts_mt")

    plt.tight_layout()
    plt.savefig(base_dir / "GSE138266_qc_panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    sc.pl.violin(adata_obj, keys="n_genes_by_counts", show=False)
    plt.title("n_genes_by_counts")
    plt.savefig(base_dir / "qc_violin_ngenes.png", dpi=300, bbox_inches="tight")
    plt.close()

    sc.pl.violin(adata_obj, keys="total_counts", show=False)
    plt.title("total_counts")
    plt.savefig(base_dir / "qc_violin_umi.png", dpi=300, bbox_inches="tight")
    plt.close()

    sc.pl.violin(adata_obj, keys="pct_counts_mt", show=False)
    plt.title("pct_counts_mt")
    plt.savefig(base_dir / "qc_violin_mt.png", dpi=300, bbox_inches="tight")
    plt.close()

    sc.pl.scatter(
        adata_obj,
        x="n_genes_by_counts",
        y="pct_counts_mt",
        show=False,
    )
    plt.title("n_genes_by_counts vs pct_counts_mt")
    plt.savefig(base_dir / "qc_scatter_mt_ngene.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("GSE138266 QC plots saved.")


# =============================================================================
# 6. GSE141578
# =============================================================================

def process_gse141578(base_dir: Path = BASE_DIR) -> ad.AnnData:
    """Process GSE141578 from mixed GEO 10x archives."""
    print_section("Processing GSE141578")

    raw_dir = base_dir / "GSE141578_RAW"
    std_dir = raw_dir / "standardized_10x"
    temp_dir = raw_dir / "temp_extract"
    output_path = base_dir / "GSE141578_raw_count.h5ad"

    top_level_tar = base_dir / "GSE141578_RAW.tar"
    if top_level_tar.exists() and (not raw_dir.exists() or not any(raw_dir.iterdir())):
        raw_dir.mkdir(parents=True, exist_ok=True)
        extract_tar(top_level_tar, raw_dir, mode="r")

    def metadata_callback(gsm_id: str, original_name: str) -> dict[str, str]:
        tissue = "CSF" if "CSF" in original_name else "PBMC" if "PBMC" in original_name else "Unknown"
        return {
            "GSM_ID": gsm_id,
            "orig.ident": gsm_id,
            "origin.ident": gsm_id,
            "tissue": tissue,
        }

    standardize_10x_archives(raw_dir, std_dir, temp_dir, prefer_filtered=True)
    adata_combined = read_standardized_10x_samples(
        raw_dir=raw_dir,
        std_dir=std_dir,
        dataset_name="GSE141578",
        metadata_callback=metadata_callback,
    )

    save_adata(adata_combined, output_path)
    print_adata_summary(adata_combined, "GSE141578")

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    return adata_combined


# =============================================================================
# 7. GSE163005
# =============================================================================

def process_gse163005(base_dir: Path = BASE_DIR) -> ad.AnnData:
    """Add patient and GEO metadata to the existing GSE163005 raw-count AnnData object."""
    print_section("Processing GSE163005")

    data_dir = base_dir / "GSE163005_RAW"
    input_candidates = [
        data_dir / "GSE163005_raw_count.h5ad",
        base_dir / "GSE163005_raw_count.h5ad",
    ]
    input_path = next((path for path in input_candidates if path.exists()), input_candidates[0])
    output_path = base_dir / "GSE163005_raw_count.h5ad"

    gz_path = data_dir / "GSE163005_annotation_patients.csv.gz"
    csv_path = data_dir / "GSE163005_annotation_patients.csv"
    patient_csv_path = gz_path if gz_path.exists() else csv_path

    print("Reading base AnnData object...")
    adata_obj = sc.read_h5ad(input_path)

    print("Mapping single-cell barcodes to patient IDs...")
    df_patients = pd.read_csv(patient_csv_path)
    df_patients.columns = ["barcode", "orig.ident"]
    df_patients.set_index("barcode", inplace=True)
    adata_obj.obs["orig.ident"] = df_patients["orig.ident"]

    if GEOparse is None:
        raise ImportError(
            "GEOparse is required for processing GSE163005. "
            "Please install it with: pip install GEOparse"
        )

    print("Parsing GEO metadata for GSE163005...")
    gse = GEOparse.get_GEO(geo="GSE163005", destdir=str(data_dir))
    df_meta = gse.phenotype_data

    df_meta["Patient_ID"] = (
        df_meta["characteristics_ch1.3.subject id"]
        .astype(str)
        .str.replace("subject id: ", "", regex=False)
    )

    patient_to_gsm = dict(zip(df_meta["Patient_ID"], df_meta.index))
    patient_to_dx = dict(zip(df_meta["Patient_ID"], df_meta["characteristics_ch1.1.diagnosis"]))
    patient_to_tissue = dict(zip(df_meta["Patient_ID"], df_meta["characteristics_ch1.0.tissue"]))

    print("Adding GSM accession, diagnosis, and tissue metadata...")
    adata_obj.obs["GSM_ID"] = adata_obj.obs["orig.ident"].map(patient_to_gsm)
    adata_obj.obs["diagnosis"] = (
        adata_obj.obs["orig.ident"]
        .map(patient_to_dx)
        .astype(str)
        .str.replace("diagnosis: ", "", regex=False)
    )
    adata_obj.obs["tissue"] = (
        adata_obj.obs["orig.ident"]
        .map(patient_to_tissue)
        .astype(str)
        .str.replace("tissue: ", "", regex=False)
    )

    print("\nMetadata preview:")
    print(adata_obj.obs.head())

    save_adata(adata_obj, output_path)
    print_adata_summary(adata_obj, "GSE163005")

    return adata_obj


# =============================================================================
# 8. GSE186802
# =============================================================================

def process_gse186802(base_dir: Path = BASE_DIR) -> ad.AnnData:
    """Process GSE186802 and retain the selected CAR-T-DIPG samples."""
    print_section("Processing GSE186802")

    data_dir = base_dir / "GSE186802_RAW"
    output_path = base_dir / "GSE186802_raw_count_subset.h5ad"

    mtx_path = data_dir / "GSM5661368_matrix.mtx.gz"
    barcodes_path = data_dir / "GSM5661368_barcodes.tsv.gz"
    feature_path = data_dir / "GSM5661368_features.tsv.gz"

    if not feature_path.exists():
        feature_path = data_dir / "GSM5661368_genes.tsv.gz"

    print("Reading the raw matrix and assigning barcodes/genes...")
    adata_obj = sc.read_mtx(mtx_path).T

    barcodes = pd.read_csv(barcodes_path, header=None, sep="\t")
    adata_obj.obs_names = barcodes[0].values

    genes = pd.read_csv(feature_path, header=None, sep="\t")
    gene_col = 1 if genes.shape[1] >= 2 else 0
    adata_obj.var_names = genes[gene_col].values
    adata_obj.var_names_make_unique()

    adata_obj.obs["cell_id"] = adata_obj.obs_names
    adata_obj.obs["GSM_ID"] = "GSM5661368"
    adata_obj.obs["orig.ident"] = "GSM5661368"

    # Remove the final numeric suffix from each cell ID to recover sample labels.
    adata_obj.obs["GSM_ID_1"] = (
        adata_obj.obs["cell_id"]
        .astype(str)
        .str.replace(r"-\d+$", "", regex=True)
    )

    target_samples = ["DIPG1-Day7", "DIPG2-Day8", "DIPG3-Day7"]
    adata_subset = adata_obj[adata_obj.obs["GSM_ID_1"].isin(target_samples)].copy()

    print(f"Cells before filtering: {adata_obj.n_obs}")
    print(f"Cells after filtering : {adata_subset.n_obs}")
    print("Retained sample labels:")
    print(adata_subset.obs["GSM_ID_1"].unique())

    save_adata(adata_subset, output_path)
    print_adata_summary(adata_subset, "GSE186802 subset")

    return adata_subset


# =============================================================================
# 9. GSE194078
# =============================================================================

def process_gse194078(base_dir: Path = BASE_DIR) -> ad.AnnData:
    """Process GSE194078 from 10x MTX files using sample-specific prefixes."""
    print_section("Processing GSE194078")

    data_dir = base_dir / "GSE194078_RAW"
    output_path = base_dir / "GSE194078_raw_count.h5ad"

    matrix_files = [f for f in os.listdir(data_dir) if f.endswith("matrix.mtx.gz")]
    prefixes = [f.replace("matrix.mtx.gz", "") for f in matrix_files]

    print(f"Detected {len(prefixes)} standard 10x samples.")

    adata_dict = {}

    for prefix in prefixes:
        parts = prefix.rstrip("_").split("_")
        gsm_id = parts[0]
        patient_id = parts[1]
        tissue = parts[2]

        try:
            adata_obj = sc.read_10x_mtx(
                data_dir,
                var_names="gene_symbols",
                cache=False,
                prefix=prefix,
            )

            adata_obj.obs["GSM_ID"] = gsm_id
            adata_obj.obs["patient_id"] = patient_id
            adata_obj.obs["tissue"] = tissue
            adata_obj.obs["orig.ident"] = f"{patient_id}_{tissue}"

            adata_obj.obs_names = f"{gsm_id}_" + adata_obj.obs_names.astype(str)
            adata_obj.obs["cell_id"] = adata_obj.obs_names

            adata_dict[gsm_id] = adata_obj
            print(f"Loaded: {gsm_id} (Patient: {patient_id}, Tissue: {tissue}, {adata_obj.n_obs} cells)")

        except Exception as error:
            print(f"Failed to load {prefix}: {error}")

    if not adata_dict:
        raise RuntimeError("No GSE194078 samples were loaded.")

    print("\nConcatenating all GSE194078 samples...")
    adata_combined = ad.concat(adata_dict, label="batch")
    adata_combined.obs_names_make_unique()

    save_adata(adata_combined, output_path)
    print_adata_summary(adata_combined, "GSE194078")

    return adata_combined


# =============================================================================
# 10. GSE200164
# =============================================================================

def process_gse200164(base_dir: Path = BASE_DIR) -> ad.AnnData:
    """Process GSE200164 from a dense count CSV file and GEO sample metadata."""
    print_section("Processing GSE200164")

    csv_path = base_dir / "GSE200164_RAW" / "GSE200164_counts.csv"
    output_path = base_dir / "GSE200164_raw_count.h5ad"

    total_cols = pd.read_csv(csv_path, nrows=0, low_memory=False).shape[1]

    df = pd.read_csv(
        csv_path,
        index_col=0,
        usecols=[0] + list(range(6, total_cols)),
        low_memory=False,
    )
    df = df.astype("float32").round().astype("int32")

    print("Count matrix loaded successfully.")
    print("Matrix shape:", df.shape)
    print(df.iloc[:3, :3])

    adata_obj = ad.AnnData(df.T)
    adata_obj.obs["orig.ident"] = [cell.split("_")[0] for cell in adata_obj.obs_names]

    if GEOparse is None:
        raise ImportError(
            "GEOparse is required for processing GSE200164. "
            "Please install it with: pip install GEOparse"
        )

    print("Downloading and parsing GEO metadata for GSE200164...")
    gse = GEOparse.get_GEO(geo="GSE200164", destdir=str(base_dir))
    meta_df = gse.phenotype_data

    print("\nMetadata preview:")
    print(meta_df[["title", "characteristics_ch1.0", "characteristics_ch1.1"]].head())

    df_sample = meta_df[["title", "geo_accession"]].copy()
    df_split = df_sample["title"].str.split("_", expand=True)
    df_split.columns = ["tissue", "Disease", "ID"]

    result = pd.concat([df_sample, df_split], axis=1)
    result.rename(columns={"geo_accession": "GSM"}, inplace=True)

    id_to_gsm = dict(zip(result["ID"], result["GSM"]))
    adata_obj.obs["orig.ident"] = adata_obj.obs["orig.ident"].map(id_to_gsm)

    save_adata(adata_obj, output_path)
    print_adata_summary(adata_obj, "GSE200164")

    return adata_obj


# =============================================================================
# 11. GSE243905
# =============================================================================

def process_gse243905(base_dir: Path = BASE_DIR) -> ad.AnnData:
    """Process GSE243905 from 10x h5 expression matrices."""
    print_section("Processing GSE243905")

    data_dir = base_dir / "GSE243905_RAW"
    output_path = base_dir / "GSE243905_raw_count.h5ad"

    h5_files = [f for f in os.listdir(data_dir) if f.endswith(".h5")]
    print(f"Detected {len(h5_files)} 10x h5 expression matrix files.")

    adata_dict = {}

    for file_name in h5_files:
        file_path = data_dir / file_name
        parts = file_name.split("_")
        gsm_id = parts[0]
        tissue = "BLD" if "BLD" in file_name else "CSF" if "CSF" in file_name else "Unknown"
        is_enriched = "enriched" in file_name
        clean_ident = file_name.replace("_filtered_feature_bc_matrix.h5", "").replace("_enriched", "")

        try:
            adata_obj = sc.read_10x_h5(file_path)
            adata_obj.var_names_make_unique()

            adata_obj.obs["GSM_ID"] = gsm_id
            adata_obj.obs["orig.ident"] = clean_ident
            adata_obj.obs["tissue"] = tissue
            adata_obj.obs["is_enriched"] = str(is_enriched)

            adata_obj.obs_names = f"{gsm_id}_" + adata_obj.obs_names.astype(str)
            adata_obj.obs["cell_id"] = adata_obj.obs_names

            dict_key = f"{clean_ident}_enriched" if is_enriched else clean_ident
            adata_dict[dict_key] = adata_obj

            print(f"Loaded: {dict_key} ({tissue}, {adata_obj.n_obs} cells)")

        except Exception as error:
            print(f"Failed to load {file_name}: {error}")

    if not adata_dict:
        raise RuntimeError("No GSE243905 samples were loaded.")

    print("\nConcatenating all GSE243905 samples...")
    adata_combined = ad.concat(adata_dict, label="batch")
    adata_combined.obs_names_make_unique()

    save_adata(adata_combined, output_path)
    print_adata_summary(adata_combined, "GSE243905")

    return adata_combined


# =============================================================================
# 12. GSE244499
# =============================================================================

def process_gse244499(base_dir: Path = BASE_DIR) -> ad.AnnData:
    """Process GSE244499 from predefined 10x MTX prefixes."""
    print_section("Processing GSE244499")

    data_dir = base_dir / "GSE244499_RAW"
    output_path = base_dir / "GSE244499_raw_count.h5ad"
    prefixes = ["GSE244499_11A_", "GSE244499_5A_"]

    adata_dict = {}

    for prefix in prefixes:
        sample_id = prefix.split("_")[1]

        try:
            adata_obj = sc.read_10x_mtx(
                data_dir,
                var_names="gene_symbols",
                cache=False,
                prefix=prefix,
            )

            adata_obj.obs["GSM_ID"] = sample_id
            adata_obj.obs["orig.ident"] = sample_id
            adata_obj.obs_names = f"{sample_id}_" + adata_obj.obs_names.astype(str)
            adata_obj.obs["cell_id"] = adata_obj.obs_names

            adata_dict[sample_id] = adata_obj
            print(f"Loaded: {sample_id} ({adata_obj.n_obs} cells, {adata_obj.n_vars} genes)")

        except Exception as error:
            print(f"Failed to load sample {sample_id}: {error}")

    if not adata_dict:
        raise RuntimeError("No GSE244499 samples were loaded.")

    print("\nConcatenating GSE244499 samples...")
    adata_combined = ad.concat(adata_dict, label="batch")
    adata_combined.var_names_make_unique()

    # Keep only the 5A sample, as in the original script.
    adata_combined = adata_combined[adata_combined.obs["orig.ident"] == "5A"].copy()

    # Replace the internal sample ID with the corresponding GSM accession.
    adata_combined.obs["orig.ident"] = "GSM7818261"

    save_adata(adata_combined, output_path)
    print_adata_summary(adata_combined, "GSE244499")

    return adata_combined


# =============================================================================
# 13. GSE247134
# =============================================================================

def process_gse247134(base_dir: Path = BASE_DIR) -> ad.AnnData:
    """Process GSE247134 from 10x MTX files using sample-specific prefixes."""
    print_section("Processing GSE247134")

    # The uploaded source script used GSE247134 in the folder and output names.
    # This workflow first tries the corrected GSE247134 folder and falls back to
    # the legacy typo folder if that is what exists on the server.
    data_dir = base_dir / "GSE247134_RAW"
    legacy_data_dir = base_dir / "GSE247134_RAW"

    if not data_dir.exists() and legacy_data_dir.exists():
        data_dir = legacy_data_dir
        print("Using legacy folder name: GSE247134_RAW")

    output_path = base_dir / "GSE247134_raw_count.h5ad"

    matrix_files = [f for f in os.listdir(data_dir) if f.endswith("matrix.mtx.gz")]
    prefixes = [f.replace("matrix.mtx.gz", "") for f in matrix_files]

    print(f"Detected {len(prefixes)} GEX samples.")

    adata_dict = {}

    for prefix in prefixes:
        parts = prefix.rstrip("_").split("_")
        gsm_id = parts[0]
        sample_id = parts[1]
        condition = sample_id.split("-")[0]

        try:
            adata_obj = sc.read_10x_mtx(
                data_dir,
                var_names="gene_symbols",
                cache=False,
                prefix=prefix,
            )

            adata_obj.obs["GSM_ID"] = gsm_id
            adata_obj.obs["orig.ident"] = sample_id
            adata_obj.obs["condition"] = condition

            adata_obj.obs_names = f"{gsm_id}_" + adata_obj.obs_names.astype(str)
            adata_obj.obs["cell_id"] = adata_obj.obs_names

            adata_dict[gsm_id] = adata_obj
            print(f"Loaded: {gsm_id} (Sample: {sample_id}, Group: {condition}, {adata_obj.n_obs} cells)")

        except Exception as error:
            print(f"Failed to load {prefix}: {error}")

    if not adata_dict:
        raise RuntimeError("No GSE247134 samples were loaded.")

    print("\nConcatenating all GSE247134 samples...")
    adata_combined = ad.concat(adata_dict, label="batch")
    adata_combined.obs_names_make_unique()

    # Keep the typo column from the original script for backward compatibility.
    adata_combined.obs["orig.indet"] = adata_combined.obs["GSM_ID"]

    save_adata(adata_combined, output_path)
    print_adata_summary(adata_combined, "GSE247134")

    return adata_combined


# =============================================================================
# 14. Main workflow
# =============================================================================

def main() -> None:
    """Run selected dataset preprocessing workflows."""
    processors = {
        "GSE133028": process_gse133028,
        "GSE134577": process_gse134577,
        "GSE138266": process_gse138266,
        "GSE141578": process_gse141578,
        "GSE163005": process_gse163005,
        "GSE186802": process_gse186802,
        "GSE194078": process_gse194078,
        "GSE200164": process_gse200164,
        "GSE243905": process_gse243905,
        "GSE244499": process_gse244499,
        "GSE247134": process_gse247134,
    }

    for dataset_name in RUN_DATASETS:
        if dataset_name not in processors:
            print(f"Unknown dataset name skipped: {dataset_name}")
            continue

        processors[dataset_name]()

    print("\nAll selected dataset preprocessing workflows have finished.")
    validate_target_files(BASE_DIR)


if __name__ == "__main__":
    main()
