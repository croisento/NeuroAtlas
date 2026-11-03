# Human CSF-PBMC Atlas: Connect Cloud bundle

This directory is a complete Shiny for Python application for Posit Connect
Cloud. 

## Data format

The `data/` directory is the GitHub-safe deployment representation of the
web-slim atlas. Every file is below GitHub's 100 MB single-file limit.

- `atlas_metadata.npz`: UMAP, categorical metadata codes, and four numeric
  cell-level QC arrays
- `metadata_categories.json`: category levels used to decode metadata
- `expression_part_*.npz`: CSC sparse matrices of quantized marker expression
- `manifest.json`: gene-to-file/column lookup and expression scale
- `cell_type_top50_marker_catalog.csv`: Wilcoxon markers for 5 major categories
- `cell_subtype_short_top50_marker_catalog.csv`: Wilcoxon markers for 25 subtypes
- `sample_metadata.csv`: clinical manifestations, pairing and PCA assessment
- `donor_matched_pairs.csv`, `paired_proportions.csv`, `paired_statistics.csv`:
  fixed manuscript cohort and BH-adjusted paired comparisons

Expression values are restored as follows:

```python
matrix = scipy.sparse.load_npz("data/expression_part_00.npz")
logcounts = matrix[:, gene_column].toarray().ravel() / manifest["expression_scale"]
```

`cloud_data_report.json` records file sizes and verifies the GitHub limit.

The application supports discrete and QC-colored UMAPs, interactive legend
visibility controls, sample-level composition, marker expression, a marker
catalog, and violin/box plots for QC metrics or marker genes.

## Manuscript-aligned interface

The header displays 573,569 cells, 161 CSF samples and 40 donor-matched pairs.
The sample table retains every sample record and distinguishes analysis pairs,
the duplicate PBMC record and samples without retained counterparts.

Clinical manifestations are joined by dataset + sample identifier from the
author-supplied Table 2 spreadsheet. They do not overwrite curated clinical
labels, age or gender. Pairing requires the preverified participant mapping.

PCA compositional-outlier flags explicitly distinguish `Flagged` (13 samples),
`Not flagged` (164) and `Not assessed` (28). Predicted doublets (6,157 cells)
and flagged samples are retained by default. Optional sidebar exclusions update
atlas, composition, expression and filtered sample views. The paired page uses
the fixed 40-pair cohort independently of sidebar filters, allows any of the 25
subtypes to be selected, and retains BH correction across all 25 tests.

Pooled composition, faceted by tissue, is the default. An alternative median
mode includes zeros for absent subtypes and does not renormalize medians.
Marker tables use tie-corrected two-sided Wilcoxon rank-sum tests and the
documented detection, effect-size and BH thresholds. Expression covers 668
unique genes in the union of both marker catalogues.

`manuscript_bundle_validation.json` records data-build checks;
`app_validation.json` records application checks. Full download DOI and final
publication citation remain pending; no placeholder repository links are used.

Rebuild with the local analysis environment using
`Code/69_build_manuscript_cloud_bundle.py` from the project root. It is a
maintenance step, not a deployment dependency. Validate using the Shiny runtime
with `python scripts/validate_bundle.py`.

## Disease metadata update

The cloud bundle retains `Disease` and `Disease1`, and excludes the redundant
lowercase `disease` field. The three samples previously labelled `Unknown`
(GSM5827377, GSM5827378, GSM5827388; 17,422 cells) now use
`controls without CNS inflammation (unspecific)` in both disease fields.
Their existing orange color is preserved. Cell order, UMAP coordinates,
expression matrices and numeric QC arrays are unchanged.

`metadata_sync_report.json` records verification against the current master
atlas. The maintenance script `scripts/sync_disease_metadata.py MASTER.h5ad`
requires the local AnnData/h5py analysis environment; it is not needed at runtime.

## Deploy under a new Posit Connect Cloud account

Use this same public repository, branch `main`, and primary application file
`app.py` when creating the application in the new account. Dependencies are
listed in `requirements.txt`. No old-account credentials are stored in this
bundle. Deployment in the new account is a separate step from updating GitHub.
