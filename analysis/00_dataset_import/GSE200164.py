"""GSE200164 expression-matrix import; original dataset-specific handling."""
from _shared import *

def process_gse200164(base_dir: Path = BASE_DIR) -> ad.AnnData:
    """Process GSE200164 from a dense count CSV file and GEO sample metadata."""
    print_section("Processing GSE200164")

    csv_path = base_dir / "GSE200164_RAW" / "GSE200164_counts.csv"
    output_path = base_dir / "GSE200164_raw_count.h5ad"

    total_cols = pd.read_csv(csv_path, nrows=0, low_memory=False).shape[1]

    df = pd.read_csv(
        csv_path,
        index_col=0,
        usecols=[0] + list(range(6, total_cols)),
        low_memory=False,
    )
    df = df.astype("float32").round().astype("int32")

    print("Count matrix loaded successfully.")
    print("Matrix shape:", df.shape)
    print(df.iloc[:3, :3])

    adata_obj = ad.AnnData(df.T)
    adata_obj.obs["orig.ident"] = [cell.split("_")[0] for cell in adata_obj.obs_names]

    if GEOparse is None:
        raise ImportError(
            "GEOparse is required for processing GSE200164. "
            "Please install it with: pip install GEOparse"
        )

    print("Downloading and parsing GEO metadata for GSE200164...")
    gse = GEOparse.get_GEO(geo="GSE200164", destdir=str(base_dir))
    meta_df = gse.phenotype_data

    print("\nMetadata preview:")
    print(meta_df[["title", "characteristics_ch1.0", "characteristics_ch1.1"]].head())

    df_sample = meta_df[["title", "geo_accession"]].copy()
    df_split = df_sample["title"].str.split("_", expand=True)
    df_split.columns = ["tissue", "Disease", "ID"]

    result = pd.concat([df_sample, df_split], axis=1)
    result.rename(columns={"geo_accession": "GSM"}, inplace=True)

    id_to_gsm = dict(zip(result["ID"], result["GSM"]))
    adata_obj.obs["orig.ident"] = adata_obj.obs["orig.ident"].map(id_to_gsm)

    save_adata(adata_obj, output_path)
    print_adata_summary(adata_obj, "GSE200164")

    return adata_obj


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, required=True)
    args = parser.parse_args()
    process_gse200164(args.base_dir)

if __name__ == '__main__':
    main()
