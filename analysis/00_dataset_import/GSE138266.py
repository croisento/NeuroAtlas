"""GSE138266 expression-matrix import; original dataset-specific handling."""
from _shared import *

def process_gse138266(
    base_dir: Path = BASE_DIR,
    save_qc_plots: bool = SAVE_GSE138266_QC_PLOTS,
) -> ad.AnnData:
    """Import GSE138266 from gzipped 10x MTX files without dataset-specific cell filtering."""
    print_section("Processing GSE138266")

    tar_path = base_dir / "GSE138266_RAW.tar"
    extract_dir = base_dir / "GSE138266_RAW"

    raw_output_path = base_dir / "GSE138266_raw_count.h5ad"

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

    return adata_gse138266


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, required=True)
    args = parser.parse_args()
    process_gse138266(args.base_dir)

if __name__ == '__main__':
    main()
