"""GSE134577 expression-matrix import; original dataset-specific handling."""
from _shared import *

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, required=True)
    args = parser.parse_args()
    process_gse134577(args.base_dir)

if __name__ == '__main__':
    main()
