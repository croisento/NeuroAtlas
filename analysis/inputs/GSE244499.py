"""GSE244499 expression-matrix import; original dataset-specific handling."""
from analysis.inputs._shared import *

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, required=True)
    args = parser.parse_args()
    process_gse244499(args.base_dir)

if __name__ == '__main__':
    main()
