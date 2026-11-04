# Processing records

The release contains 573,569 cells and 12,890 shared genes. Predicted doublets
and PCA compositional outliers remain included.

| Stage | Recorded operation | Source |
| --- | --- | --- |
| Early cell QC | At least 400 detected genes; mitochondrial fraction <20%; genes in at least 3 cells | Original `NEU_anno.ipynb`, cell 9 |
| Global reprocessing | 2,000 dataset-aware HVGs; PCA 60; sample-level Harmony; 30-neighbor graph using 30 PCs; Leiden 2.0 | Original `NEU-0126-2.ipynb`, cells 13 and 15 |
| Lineage review | B, T, NK and myeloid analyses; marker-based cluster review and exclusions | Original `NEU_B.ipynb`, `NEU_T.ipynb`, `NEU_NK.ipynb`, `NEU_Myeloid.ipynb` |
| Annotation transfer | Transfer lineage labels by cell identifiers; exclude unresolved and manually reviewed non-target cells | Original `cell_type_integration.ipynb` |
| Raw counts reconstruction | Import 11 datasets; retain genes in at least 1 cell per dataset; inner concatenation gives 12,890 shared genes | Original `adata_integration.ipynb` |
| Cell matching | Within-sample correlation of 48 shared expression features; highest-correlation raw-cell match; retain highest-scoring match per raw ID | Same notebook; 7 duplicate mappings removed |
| Label/logcounts records | Standardized subtype names and log-normalized expression layer | Original `rplot0618.ipynb` |
| Sample filtering | Apply the recorded 8 non-CAR-T exclusions using the uniform-QC audit count, not the final annotated cell count | `config/sample_exclusions.csv`; original sample-filter source |
| Final metadata | Curated Disease/Disease1, age, sex and tissue; retain sample IDs and old/raw cell identifiers | `data/sample_metadata.csv` and source metadata scripts |
| Optional QC | Sample-wise Scrublet and tissue/disease-specific PCA sample flags | Steps 09 and 10 |

The 48-gene procedure restored correspondence with raw counts; it did not
generate new cell-type annotations. The historical matching code has no
minimum correlation rejection threshold and is not a global optimal matching
algorithm. Duplicate mapping removal is not biological doublet removal.

The eight excluded samples are GSM3984210, GSM5827387, GSM6016358, GSM6016365,
GSM6016367, GSM6016370, GSM6016371 and GSM6016374. The three GSE186802 CAR-T
samples were exempted. The threshold used the separate uniform-QC branch's
sample counts; it should not be restated as a new filter on annotated counts.

The final atlas also excludes the historical Basophil and Transitional B
labels. Fine annotations, retained cell identifiers and stored coordinates in
the released objects define the final membership. Some historical manual
selections were interactive, so they are not recoverable as a standalone fixed
cluster-number rule. Cluster IDs from one object must not be applied to another.

The July `july_remerge` profile is a separate analysis: 3,000 HVGs, PCA 50,
dataset-level Harmony (maximum 30 iterations), 30 neighbors using 30 PCs,
cosine distance, Leiden 0.6 and UMAP min_dist 0.35/spread 1. It does not establish
that the final atlas embeddings were recomputed with those parameters.

The numbered scripts isolate operations and expose input/output paths. Original
source names and hashes are recorded in `source_inventory.json`. The final
objects preserve the retained annotation and embedding state.

## Manuscript details to retain

- State the Harmony batch variable for the applicable branch; `orig.ident`
  and `dataset` are different variables.
- Shared raw-count preparation uses `min_cells=1`; the earlier annotation QC
  contains `min_cells=3`. Do not treat them as the same operation.
- Initial GSE138266 import also has a strict `n_genes >400` filter.
- Most early PCA calls use seed 97; the July remerge uses seed 0. Several B/NK
  graph calls omit random_state and use package defaults.
- Doublet scoring was performed on retained final-atlas cells and shared genes;
  it is an optional post-hoc annotation, not a pre-Harmony exclusion step.
- Figure 4 reports paired tests across 40 donors with BH correction across all
  25 subtypes. Figure 5d reports pooled proportions without disease-group tests.
- The latest marker code includes estimated log2 fold change ≥0.25 and a
  positive signed statistic in addition to detection fraction and adjusted P.
