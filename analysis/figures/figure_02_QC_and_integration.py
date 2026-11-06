"""Manuscript Figure 2a–d; QC/UMAP from original 08 and tissue display from 04."""
from figures.common import arguments, qc_object, categorical_umap, style
from figures.qc_plotting import plot_supplementary_figure_1a_qc_violin


def main():
    from figures.final_publication_figures import setup, figure2
    from figures.common import metadata_object
    a = arguments(__doc__).parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    setup()
    figure2(metadata_object(a.input), a.output, a.input)



if __name__ == '__main__':
    main()
