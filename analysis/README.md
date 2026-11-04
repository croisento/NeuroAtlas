# Analysis code

## Data and processing

The numbered modules follow the Methods sections. Run commands from the
repository root using Python 3.12.11 and `analysis/requirements.txt`.
The root `requirements.txt` is for the Shiny application only.

| Order | Code | Input | Output |
| --- | --- | --- | --- |
| 01 | `inputs/GSE*.py` | Dataset-specific public expression matrices | Per-dataset raw-count AnnData objects |
| 02 | `02_merge_shared_genes.py` | Prepared objects listed in a JSON manifest | Gene-intersection raw-count matrix |
| 03 | `03_cell_quality_control.py` | Raw counts | Cell/gene-filtered object and retained QC annotations |
| 04 | `04_preliminary_clustering.py` | QC object and recorded profile | PCA, clusters and cluster marker table for review |
| 05 | `05_apply_contaminant_review.py` | Exact input object and reviewed excluded-cell CSV | Retained-cell object |
| 06 | `06_harmony_integration.py` | Retained cells and recorded profile | Normalized expression, PCA, Harmony, neighbors, Leiden and UMAP |
| 07 | `07_cell_type_annotation.py` | Object and reviewed per-cell annotation CSV | Annotated object |
| 08 | `08_sample_filter_and_metadata.py` | Annotated object, sample-exclusion list and curated metadata | Sample-filtered object with standardized metadata |
| 09 | `09_identify_doublets.py` | Raw counts in the final object | Sample-wise Scrublet scores and labels; no deletion |
| 10 | `10_PCA_compositional_outliers.py` | Sample-level subtype proportions and metadata | PCA coordinates, distance z-scores and sample flags |
| 11 | `11_cell_type_markers.py` | Final log-normalized expression | Wilcoxon marker tables for major types and subtypes |
| 12 | `12_paired_CSF_PBMC.py` | Sample proportions and verified donor-pair table | Paired differences, tests and bootstrap intervals |
| 13 | `13_export_tables.py` | Dataset, sample and marker records | CSV tables |
| 14 | `14_export_website_data.py` | Final atlas and publication tables | Shiny metadata and compact marker-expression files |
| 15 | `15_export_final_sample_QC_table.py` | Final counts and the clinical/QC workbook | Final-cell QC medians with GSM-linked clinical fields |
| 16 | `16_export_participant_IDs.py` | Sample records, GEO titles and final QC table | Original/internal participant IDs and pairing records |

The release was assembled through several analysis stages. Early annotation
and lineage review preceded the reconstruction of the 12,890-gene raw-count
matrix. Existing annotations and embeddings were transferred to matched cells;
later metadata corrections and sample exclusions retained those coordinates.
The numbered modules separate these operations for reuse; they are not a
single command that reconstructs all historical manual selections.

The numbered scripts contain their implementations directly.
`source_inventory.json` records the original source names and hashes and the
dataset-reader origins. Source references identify processing records; the
original notebooks and unorganized scripts are not part of this code directory.

`processing_notes.md` describes the count reconstruction, cell matching,
sample-exclusion basis and differences between analysis branches.
`parameters.csv` lists the recorded settings and their sources.

### Inputs

Each dataset has a separate reader. For example:

```bash
python -m analysis.inputs.GSE133028 --base-dir /path/to/raw_data
python -m analysis.02_merge_shared_genes \
  --manifest analysis/config/prepared_datasets.example.json \
  --output results/shared_counts.h5ad
```

Edit the example manifest to point to the actual prepared files. The expected
raw filenames and directories are defined inside each reader; they are not
claims that those directories are distributed with this repository. GSE163005
requires a pre-existing count object and its barcode–patient table. The upstream
generation of that object is not inferred from its import step.

The readers preserve dataset-specific import behavior. In particular,
GSE138266 also exports its original prefiltered variant (`n_genes >400`,
mitochondrial percentage <20). The shared-count branch uses `min_cells=1`
per dataset before the gene intersection; the early annotation QC branch used
`min_cells=3`. These are different stages.

The early/lineage and July processing profiles in `preprocessing.py` are
distinct. Select a profile explicitly; no profile is presented as a unique
set of parameters for every historical object. Reviewed cell lists and
per-cell annotation tables are required for steps 05 and 07.

### Released objects and statistical tables

Place the total object and four lineage objects under a local `h5ad/`
directory. Set `NEUCSF_DATA_ROOT` to its parent, or pass `--input` explicitly.
The objects are external data inputs and are not included in GitHub.

```bash
export NEUCSF_DATA_ROOT=/path/to/released_data
python -m analysis.10_PCA_compositional_outliers --output results/PCA_flags.csv
python -m analysis.12_paired_CSF_PBMC --output results/paired
python -m analysis.11_cell_type_markers \
  --input "$NEUCSF_DATA_ROOT/h5ad/adata_final_Disease_corrected_min400_except_CART_scrublet_identified.h5ad" \
  --output results/markers
```

The sample proportions, clinical metadata, pair registry and reference paired
results are included as CSV files. Statistical scripts do not delete predicted
doublets or PCA-flagged samples. GSM5264668 is excluded only from paired
comparisons. The paired cohort contains 40 donor-matched pairs; this does not
imply that all pairs have independently verified same-visit sampling.

## Figure and table code

`figure_panel_map.csv` maps every panel to its organized script, original
source and output filename. The current manuscript uses Figures 1–5 and S1;
older script names such as `04_figure2.py` refer to earlier figure numbering.

| Manuscript figure | Code | Contents |
| --- | --- | --- |
| Figure 1 | `figures/figure_01_workflow.mmd` | Workflow diagram source; manuscript schematic assembled manually |
| Figure 2a–d | `figures/figure_02_QC_and_integration.py` | CSF/PBMC QC violins, dataset UMAP and tissue UMAP |
| Figure 3a–f | `figures/figure_03_cell_annotations.py` | Global and four lineage embeddings; canonical marker dot plot |
| Figure 4a–c | `figures/figure_04_paired_proportions.py` | NK bright, NK dim and CD4 memory paired plots |
| Figure 5a–d | `figures/figure_05_clinical_composition.py` | Clinical-group UMAP, sample counts and pooled cell composition |
| Figure S1a–c | `figures/figure_S1_optional_QC_flags.py` | PCA sample flags and predicted-doublet/score UMAPs |
| Dataset/sample/marker tables | `13_export_tables.py` | Source-study, clinical and marker CSV tables |

```bash
python -m analysis.figures.figure_02_QC_and_integration --output results/Figure2
python -m analysis.figures.figure_03_cell_annotations --output results/Figure3
python -m analysis.figures.figure_04_paired_proportions --output results/Figure4
python -m analysis.figures.figure_05_clinical_composition --output results/Figure5
python -m analysis.figures.figure_S1_optional_QC_flags --output results/FigureS1
python -m analysis.13_export_tables --output results/Tables
python -m analysis.14_export_website_data --output results/website_data
```

The plotting scripts redraw panel contents from the released data and export
PDF, SVG and PNG files. They do not fit new embeddings. Typeface, panel spacing,
label placement and the final assembled artwork can differ from the manually
arranged manuscript figures. Figure 2 uses the frozen per-cell QC vectors in
the web bundle; its raw-total and ribosomal vectors were calculated from the
counts layer. Coordinates and cell order are checked against the total object.

Figure 5d is a pooled descriptive composition plot. It does not display
disease-versus-control significance tests. The former lineage-restricted
disease comparison is not included as a current manuscript figure.

Steps 15 and 16 export the final QC and participant tables. They require the
author-supplied clinical workbook and sample-title records in addition to the
released object; pass their paths explicitly as shown by each script's `--help`.
