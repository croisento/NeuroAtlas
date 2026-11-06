"""Figure S1a,b: original CSF/PBMC composition PCA plotting functions.

PCA is calculated on sample proportions for this figure, not on cell expression.
No cell or sample is excluded. Doublet panel plotting is a separate entry.
"""
from figures.common import arguments, metadata_object, style
from figures import csf_pca, pbmc_pca


def main():
    from figures.final_publication_figures import setup, figureS1
    a = arguments(__doc__).parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    setup()
    figureS1(metadata_object(a.input), a.output)



if __name__ == '__main__':
    main()
