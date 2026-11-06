"""Manuscript Figure 3a–f, using original 04 functions and saved embeddings."""
from pathlib import Path
import h5py
import anndata as ad
from figures.common import arguments, metadata_object, categorical_umap, style
from figures.annotation_plotting import (
    SOFT_COLOR_MAP, CELL_TYPE_ORDER, MARKER_DICT, DOTPLOT_CELL_TYPE_ORDER,
    DOTPLOT_EXCLUDE_CELLTYPES, plot_lineage_umap, plot_marker_gene_dotplot,
)
from analysis.expression import selected_logcounts


def main():
    from figures.final_publication_figures import setup
    from figures.final_figure3 import render
    p = arguments(__doc__)
    p.add_argument('--h5ad-dir', type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    setup()
    render(a.input, a.h5ad_dir or a.input.parent, a.output)



if __name__ == '__main__':
    main()
