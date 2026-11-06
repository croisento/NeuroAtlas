# CSF–PBMC analysis code

## Files

| Order | File | Content |
|---|---|---|
| 00 | `00_dataset_import/GSE*.py` | Separate raw-count importer for each of 11 GEO series |
| 01 | `01_load_merge_and_plot_gene_intersections.py` | Retain genes detected in at least three cells per dataset; merge shared genes and plot intersections |
| 02 | `02_preprocessing_Harmony_integration_clustering.py` | Global normalization, HVG selection, PCA, Harmony, Leiden and UMAP |
| 02.1 | `02.1_QC_and_contaminant_screening.py` | Cell/gene/sample QC, preliminary clustering and marker-expression inspection |
| 03 | `03_lineage_specific_Harmony_reclustering_and_UMAP.py` | Lineage reclustering of an already annotated parent object |
| 05 | `05_optional_quality_control_flags.py` | Per-sample Scrublet doublet scores/labels and sample-composition PCA flags; no deletion |

`00_dataset_import/_shared.py` contains importer helpers. `analysis/` contains
the helpers required by 02.1 and selected-expression access for figures. The combined 00 script and earlier alternative
integration/annotation entry points are not included.

## Inputs and use

Install packages from `requirements.txt`. Run dataset importers separately:

```bash
python 00_dataset_import/GSE163005.py --base-dir /path/to/raw_data
```

GSE163005 reads GEO MTX/TSV and retains only matrix barcodes present in the
patient annotation CSV. GSE138266 exports raw counts without its former
dataset-specific cell filtering. Existing importer output files are protected
against overwriting.

01, 02 and 03 retain the original section-based script style. Set their input,
output and working-directory paths near the top before running. 02 and 03 are
unchanged copies of the original July scripts. 01 differs only in the requested
min_cells=3 threshold and use of unfiltered GSE138266 raw counts.

```bash
python 02.1_QC_and_contaminant_screening.py \
  --input /path/to/raw_counts.h5ad --output /path/to/new_QC_output
```

02.1 retains cells with >=400 detected genes and mitochondrial counts <20%,
then genes detected in >=3 retained cells. It requires >=400 cells per sample,
except GSE186802. Per-dataset clustering and marker dotplots follow. It does
not classify or remove contaminant clusters. The decimal number is a file
label: this script reads raw counts (preferentially `layers['counts']`), not a
Harmony embedding, and can be run on the merged raw-count object from 01.

02 calculates QC metrics but does not filter cells. 03 requires existing
`cell_type`, `cell_subtype_short` and log-normalized expression; it does not
generate manual annotations. These files are not a complete reconstruction of
the historical annotation steps between global clustering and `adata_final`.

## Optional QC flags

```bash
python 05_optional_quality_control_flags.py \
  --input /path/to/annotated_atlas.h5ad --output /path/to/new_optional_QC_output
```

05 is copied unchanged from the combined Scrublet/PCA module. It saves a
doublet-annotated h5ad and `PCA_compositional_outlier_flags.csv`. Cells, samples
and existing UMAP coordinates are retained. The supplied
`analysis/data/sample_subtype_proportions.csv` and `data/sample_metadata.csv`
are the default inputs for the current atlas; use `--proportions` and
`--metadata` for another input cohort. Sample keys must match the input h5ad.

This folder contains the confirmed processing files, optional QC flags and
manuscript plotting entries in `figures/`. See [figures/README.md](figures/README.md)
for panel mappings, original sources, input requirements and commands.

```bash
python run_manuscript_figures.py --input /path/to/final_atlas.h5ad --h5ad-dir /path/to/lineage_objects
```

Figure outputs are written to `Outputs/Figure2/`, `Outputs/Figure3/`,
`Outputs/Figure4/`, `Outputs/Figure5/` and `Outputs/FigureS1/`. Existing
same-name figures are overwritten. Figure 2 includes both the assembled page
and separate a–d panels using the same rendering. Outputs are generated locally
and excluded from Git; the repository contains code and supporting metadata.
