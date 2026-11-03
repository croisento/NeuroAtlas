"""Figure 4a–c: NK bright, NK dim and CD4 memory in 40 donor-matched pairs."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from analysis.common import REPOSITORY,save_figure,setup_style


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--proportions',type=Path,default=REPOSITORY/'data/paired_proportions.csv')
    p.add_argument('--statistics',type=Path,default=REPOSITORY/'data/paired_statistics.csv')
    a=p.parse_args();setup_style()
    source=pd.read_csv(a.proportions);stats=pd.read_csv(a.statistics)
    colors=dict(zip(['GSE133028','GSE138266','GSE194078','GSE243905'],['#377eb8','#4daf4a','#984ea3','#e39d31']))
    fig,axes=plt.subplots(1,3,figsize=(12,5.8))
    for ax,letter,subtype in zip(axes,'abc',['NK bright','NK dim','CD4 Memory']):
        data=source.loc[source.cell_subtype_short.eq(subtype)]
        assert data.pair_id.nunique()==40
        for row in data.itertuples():ax.plot([0,1],[row.CSF_percent,row.PBMC_percent],color=colors[row.dataset],alpha=.6,lw=.8,marker='o',ms=3)
        stat=stats.loc[stats.cell_subtype_short.eq(subtype)].iloc[0]
        ax.set_xticks([0,1],['CSF','PBMC']);ax.set_ylabel('Cell proportion (%)')
        ax.set_title(f'{letter}  {subtype}\nBH-adjusted P = {stat.p_adjusted_BH:.3g}')
    save_figure(fig,a.output/'Figure4abc_paired_proportions')


if __name__=='__main__':main()
