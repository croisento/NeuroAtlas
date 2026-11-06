# NeuroAtlas code


| Order | File                                                     | Content                                                                                              |
| ----- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 00    | `00_dataset_import/GSE*.py`                            | Separate raw-count importer for each of 11 GEO series                                                |
| 01    | `01_load_merge_and_plot_gene_intersections.py`         | Retain genes detected in at least three cells per dataset; merge shared genes and plot intersections |
| 02    | `02_preprocessing_Harmony_integration_clustering.py`   | Global normalization, HVG selection, PCA, Harmony, Leiden and UMAP                                   |
| 03    | `03_QC_and_contaminant_screening.py`                   | Cell/gene/sample QC, preliminary clustering and marker-expression inspection                         |
| 04    | `04_lineage_specific_Harmony_reclustering_and_UMAP.py` | Lineage reclustering                                                                                |
| 05    | `05_optional_quality_control_flags.py`                 | Per-sample Scrublet doublet scores/labels and sample-composition PCA flags; no deletion              |
