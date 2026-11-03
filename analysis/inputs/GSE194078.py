"""GSE194078 expression-matrix import; original dataset-specific handling."""
from analysis.inputs._shared import *

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, required=True)
    args = parser.parse_args()
    process_gse194078(args.base_dir)

if __name__ == '__main__':
    main()
