"""GSE247134 expression-matrix import; original dataset-specific handling."""
from analysis.inputs._shared import *

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, required=True)
    args = parser.parse_args()
    process_gse247134(args.base_dir)

if __name__ == '__main__':
    main()
