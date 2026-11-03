"""GSE186802 expression-matrix import; original dataset-specific handling."""
from analysis.inputs._shared import *

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, required=True)
    args = parser.parse_args()
    process_gse186802(args.base_dir)

if __name__ == '__main__':
    main()
