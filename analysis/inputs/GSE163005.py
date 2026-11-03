"""GSE163005 expression-matrix import; original dataset-specific handling."""
from analysis.inputs._shared import *

def process_gse163005(base_dir: Path = BASE_DIR) -> ad.AnnData:
    """Add patient and GEO metadata to the existing GSE163005 raw-count AnnData object."""
    print_section("Processing GSE163005")

    data_dir = base_dir / "GSE163005_RAW"
    input_candidates = [
        data_dir / "GSE163005_raw_count.h5ad",
        base_dir / "GSE163005_raw_count.h5ad",
    ]
    input_path = next((path for path in input_candidates if path.exists()), input_candidates[0])
    output_path = base_dir / "GSE163005_raw_count.h5ad"

    gz_path = data_dir / "GSE163005_annotation_patients.csv.gz"
    csv_path = data_dir / "GSE163005_annotation_patients.csv"
    patient_csv_path = gz_path if gz_path.exists() else csv_path

    print("Reading base AnnData object...")
    adata_obj = sc.read_h5ad(input_path)

    print("Mapping single-cell barcodes to patient IDs...")
    df_patients = pd.read_csv(patient_csv_path)
    df_patients.columns = ["barcode", "orig.ident"]
    df_patients.set_index("barcode", inplace=True)
    adata_obj.obs["orig.ident"] = df_patients["orig.ident"]

    if GEOparse is None:
        raise ImportError(
            "GEOparse is required for processing GSE163005. "
            "Please install it with: pip install GEOparse"
        )

    print("Parsing GEO metadata for GSE163005...")
    gse = GEOparse.get_GEO(geo="GSE163005", destdir=str(data_dir))
    df_meta = gse.phenotype_data

    df_meta["Patient_ID"] = (
        df_meta["characteristics_ch1.3.subject id"]
        .astype(str)
        .str.replace("subject id: ", "", regex=False)
    )

    patient_to_gsm = dict(zip(df_meta["Patient_ID"], df_meta.index))
    patient_to_dx = dict(zip(df_meta["Patient_ID"], df_meta["characteristics_ch1.1.diagnosis"]))
    patient_to_tissue = dict(zip(df_meta["Patient_ID"], df_meta["characteristics_ch1.0.tissue"]))

    print("Adding GSM accession, diagnosis, and tissue metadata...")
    adata_obj.obs["GSM_ID"] = adata_obj.obs["orig.ident"].map(patient_to_gsm)
    adata_obj.obs["diagnosis"] = (
        adata_obj.obs["orig.ident"]
        .map(patient_to_dx)
        .astype(str)
        .str.replace("diagnosis: ", "", regex=False)
    )
    adata_obj.obs["tissue"] = (
        adata_obj.obs["orig.ident"]
        .map(patient_to_tissue)
        .astype(str)
        .str.replace("tissue: ", "", regex=False)
    )

    print("\nMetadata preview:")
    print(adata_obj.obs.head())

    save_adata(adata_obj, output_path)
    print_adata_summary(adata_obj, "GSE163005")

    return adata_obj


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, required=True)
    args = parser.parse_args()
    process_gse163005(args.base_dir)

if __name__ == '__main__':
    main()
