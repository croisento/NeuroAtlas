"""GSE133028 expression-matrix import; original dataset-specific handling."""
from analysis.inputs._shared import *

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, required=True)
    args = parser.parse_args()
    process_gse133028(args.base_dir)

if __name__ == '__main__':
    main()
