"""GSE243905 expression-matrix import; original dataset-specific handling."""
from _shared import *

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, required=True)
    args = parser.parse_args()
    process_gse243905(args.base_dir)

if __name__ == '__main__':
    main()
