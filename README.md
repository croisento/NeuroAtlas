# neuroCSF-PBMC

## Database information

neuroCSF-PBMC contains 573,569 cells from 11 public GEO datasets, including
161 CSF samples and 40 donor-matched CSF–PBMC pairs. The atlas covers
inflammatory, neurodegenerative, infectious and treatment-associated conditions.
Healthy or cognitively normal controls are classified as Younger control
(<60 years) or Older control (≥60 years).

Cells are annotated into 5 major cell types and 25 subtypes. T cells, B cells,
NK/ILCs and myeloid/DC populations form four display groups; pDCs retain a
separate major cell-type label. Sample records include GEO identifiers, tissue,
disease, age, sex, clinical manifestations and donor-pairing information.

The website provides UMAP visualization, marker expression, cell-composition
plots, violin and box plots, marker tables and sample metadata downloads.
Expression browsing covers 668 genes from the top-50 marker lists for major
cell types and subtypes. The full atlas contains 12,890 shared genes.

Paired comparisons use 40 donor-matched pairs and two-sided Wilcoxon
signed-rank tests, with Benjamini–Hochberg correction across 25 subtypes.
GSM5264668 is retained in the atlas but excluded from paired comparisons as
a duplicate PBMC record; GSM5264669 is used for that donor.

Predicted doublets (6,157 cells) and PCA compositional outliers (13 samples)
are retained and available for optional filtering. PCA assessment records
distinguish flagged, not flagged and not assessed samples.

## Data format

| File | Contents |
| --- | --- |
| `data/atlas_metadata.npz` | UMAP coordinates, categorical annotation codes, age, QC metrics and doublet scores |
| `data/metadata_categories.json` | Labels for categorical annotation codes |
| `data/expression_part_*.npz` | Sparse CSC matrices of marker-gene expression |
| `data/manifest.json` | Gene locations, expression scale and dataset dimensions |
| `data/cell_type_top50_marker_catalog.csv` | Top-50 markers for each major cell type |
| `data/cell_subtype_short_top50_marker_catalog.csv` | Top-50 markers for each cell subtype |
| `data/sample_metadata.csv` | Sample annotations, clinical manifestations, pairing and PCA flags |
| `data/donor_matched_pairs.csv` | Sample identifiers for the 40 analysis pairs |
| `data/paired_proportions.csv` | Per-sample subtype proportions and within-donor differences |
| `data/paired_statistics.csv` | Paired test results, adjusted P values and effect-size confidence intervals |
| `app.py` | Shiny for Python application |
| `www/style.css` | Website styles |
| `requirements.txt` | Python package dependencies |

Expression matrices store log1p-normalized counts per 10,000, multiplied by
1,000 and rounded to integers. Divide stored values by the expression scale
in `manifest.json` to recover expression values for display.

Marker tables report one-versus-rest, tie-corrected Wilcoxon rank-sum tests.
Markers have an expression fraction ≥25% in either group, estimated log₂ fold
change ≥0.25 and BH-adjusted P <0.05. Up to 50 positive markers per group are
ranked by test statistic.
