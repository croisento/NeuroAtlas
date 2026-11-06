"""Figure 5a–d: original 04 UMAP style and 05 pooled-cell counting logic.

Dataset/tissue facets and sample-count labels follow the current manuscript.
No disease-control hypothesis test is performed for these descriptive panels.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from figures.common import arguments, metadata_object, categorical_umap, style
from figures.annotation_plotting import SOFT_COLOR_MAP, CELL_TYPE_ORDER


def main():
    from figures.final_publication_figures import setup, figure5
    a = arguments(__doc__).parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    setup()
    figure5(metadata_object(a.input), a.output)



if __name__ == '__main__':
    main()
