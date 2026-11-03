"""Shared I/O helpers extracted from the dataset preparation script."""
from __future__ import annotations
import gc, gzip, os, shutil, tarfile, tempfile, warnings
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
BASE_DIR = Path(os.environ.get('NEUCSF_RAW_ROOT', 'inputs/raw'))
SAVE_GSE138266_QC_PLOTS = False
TARGET_H5AD_FILES = []

def save_adata(adata_obj, output_path):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f'Output already exists: {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_obj.write_h5ad(output_path, compression='gzip')

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
