"""GSE141578 expression-matrix import; original dataset-specific handling."""
from _shared import *

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, required=True)
    args = parser.parse_args()
    process_gse141578(args.base_dir)

if __name__ == '__main__':
    main()
