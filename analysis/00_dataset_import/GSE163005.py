"""Import GEO's combined GSE163005 MTX/TSV counts and patient annotation.

Place GSE163005_matrix.mtx(.gz), GSE163005_features.tsv(.gz),
GSE163005_barcodes.tsv(.gz), and GSE163005_annotation_patients.csv(.gz)
in BASE_DIR/GSE163005_RAW or directly in BASE_DIR. No input h5ad is used.
"""
from _shared import *
from scipy.io import mmread
import numpy as np


def source_file(base_dir, name):
    for directory in (base_dir / 'GSE163005_RAW', base_dir):
        for suffix in ('.gz', ''):
            path = directory / (name + suffix)
            if path.is_file():
                return path
    raise FileNotFoundError(f'Missing GEO input: {name}(.gz) in {base_dir}')


def read_raw_matrix(base_dir):
    """Read the combined gene-by-cell matrix, preserving GEO barcodes."""
    matrix_path = source_file(base_dir, 'GSE163005_matrix.mtx')
    features = pd.read_csv(source_file(base_dir, 'GSE163005_features.tsv'),
                           sep='\t', header=None, dtype=str, keep_default_na=False)
    barcodes = pd.read_csv(source_file(base_dir, 'GSE163005_barcodes.tsv'),
                           sep='\t', header=None, dtype=str, keep_default_na=False)[0]
    opener = gzip.open if matrix_path.suffix == '.gz' else open
    with opener(matrix_path, 'rb') as handle:
        matrix = mmread(handle).T.tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if matrix.shape != (len(barcodes), len(features)):
        raise ValueError('MTX dimensions do not match barcode/feature tables')
    if (not np.isfinite(matrix.data).all() or (matrix.data < 0).any()
            or not np.equal(matrix.data, np.floor(matrix.data)).all()):
        raise ValueError('Expression matrix must contain nonnegative integer counts')
    if barcodes.duplicated().any():
        raise ValueError('Duplicate GEO barcodes; patient mapping would be ambiguous')
    obj = ad.AnnData(matrix, obs=pd.DataFrame(index=pd.Index(barcodes.to_numpy(), name=None)))
    obj.var['gene_ids'] = features[0].to_numpy()
    obj.var_names = pd.Index((features[1] if features.shape[1] > 1 else features[0]).to_numpy(), name=None)
    obj.var_names_make_unique()
    if features.shape[1] > 2:
        obj.var['feature_types'] = features[2].to_numpy()
        obj = obj[:, obj.var['feature_types'].eq('Gene Expression')].copy()
    return obj

def process_gse163005(base_dir: Path = BASE_DIR) -> ad.AnnData:
    """Construct raw-count AnnData from MTX/TSV and add patient/GEO metadata."""
    print_section("Processing GSE163005")

    data_dir = base_dir / "GSE163005_RAW"
    output_path = base_dir / "GSE163005_raw_count.h5ad"
    if output_path.exists():
        raise FileExistsError(f'Output already exists: {output_path}')
    patient_csv_path = source_file(base_dir, 'GSE163005_annotation_patients.csv')
    print("Reading GEO MTX, features, and barcodes...")
    adata_obj = read_raw_matrix(base_dir)

    print("Mapping single-cell barcodes to patient IDs...")
    df_patients = pd.read_csv(patient_csv_path, dtype=str)
    if df_patients.shape[1] != 2:
        raise ValueError('Patient annotation must contain barcode and patient columns')
    df_patients.columns = ["barcode", "orig.ident"]
    df_patients.set_index("barcode", inplace=True)
    if not df_patients.index.is_unique:
        raise ValueError('Duplicate barcodes in patient annotation')
    n_input = adata_obj.n_obs
    annotated = adata_obj.obs_names.isin(df_patients.index)
    if not annotated.any():
        raise ValueError('No matrix barcodes match the patient annotation')
    adata_obj = adata_obj[annotated].copy()
    print(f'Patient annotation filter: {n_input:,} -> {adata_obj.n_obs:,} cells '
          f'({n_input - adata_obj.n_obs:,} removed without an annotation barcode)')
    adata_obj.obs["orig.ident"] = df_patients['orig.ident'].reindex(adata_obj.obs_names)
    if adata_obj.obs['orig.ident'].isna().any():
        raise ValueError('Some matrix barcodes have no patient annotation; refusing partial mapping')

    if GEOparse is None:
        raise ImportError(
            "GEOparse is required for processing GSE163005. "
            "Please install it with: pip install GEOparse"
        )

    print("Parsing GEO metadata for GSE163005...")
    data_dir.mkdir(parents=True, exist_ok=True)
    gse = GEOparse.get_GEO(geo="GSE163005", destdir=str(data_dir))
    df_meta = gse.phenotype_data

    df_meta["Patient_ID"] = (
        df_meta["characteristics_ch1.3.subject id"]
        .astype(str)
        .str.replace("subject id: ", "", regex=False)
    )

    patient_to_gsm = dict(zip(df_meta["Patient_ID"], df_meta.index))
    if df_meta['Patient_ID'].duplicated().any():
        raise ValueError('GEO subject IDs are not unique; sample assignment needs review')
    patient_to_dx = dict(zip(df_meta["Patient_ID"], df_meta["characteristics_ch1.1.diagnosis"]))
    patient_to_tissue = dict(zip(df_meta["Patient_ID"], df_meta["characteristics_ch1.0.tissue"]))

    print("Adding GSM accession, diagnosis, and tissue metadata...")
    adata_obj.obs["GSM_ID"] = adata_obj.obs["orig.ident"].map(patient_to_gsm)
    if adata_obj.obs['GSM_ID'].isna().any():
        raise ValueError('Some patient IDs could not be mapped to GEO samples')
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
