"""Figure 4a–c: 65/61 donor-matched analysis or existing result-table plotting."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figures.common import style


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proportions',type=Path,default=root/'data/paired_proportions.csv')
    p.add_argument('--statistics',type=Path,default=root/'data/paired_statistics.csv')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input',type=Path,help='Recalculate from the annotated atlas using original 65/61 rules')
    p.add_argument('--metadata',type=Path,default=root/'data/sample_metadata.csv')
    p.add_argument('--titles',type=Path,default=root/'data/GEO_sample_titles.csv')
    a = p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); style()
    source = pd.read_csv(a.proportions); stats = pd.read_csv(a.statistics)
    if a.input:
        from figures.paired_comparison import calculate_from_atlas
        source, stats = calculate_from_atlas(a.input,a.metadata,a.titles,
                                            a.output/'Tables',stats.cell_subtype_short.tolist())
    if len(stats)!=25:raise ValueError('Expected BH statistics across all 25 subtypes')
    from figures.final_publication_figures import setup, figure4
    setup()
    figure4(source, stats, a.output)



if __name__ == '__main__':
    main()
