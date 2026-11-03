"""Figure 5a–d: disease UMAP, sample counts and pooled subtype proportions."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from analysis.common import arguments,obs_umap,save_figure,setup_style
from analysis.figures.helpers import discrete_figure,DISEASE_COLORS,COLORS,ORDER


def main():
    p=arguments(__doc__);a=p.parse_args();setup_style();a.output.mkdir(parents=True,exist_ok=True)
    obs,xy=obs_umap(a.input)
    discrete_figure(xy,obs.Disease,DISEASE_COLORS,'Disease / control group',a.output/'Figure5a_disease_UMAP')
    samples=obs[['dataset','orig.ident','tissue','Disease']].drop_duplicates()
    for letter,tissue in [('b','CSF'),('c','PBMC')]:
        s=samples.loc[samples.tissue.astype(str).eq(tissue)]
        table=pd.crosstab(s.dataset,s.Disease)
        table=table.loc[table.sum(axis=1).sort_values(ascending=False).index]
        fig,ax=plt.subplots(figsize=(11,5));bottom=np.zeros(len(table))
        for disease in table:
            heights=table[disease].to_numpy()
            ax.bar(np.arange(len(table)),heights,bottom=bottom,color=DISEASE_COLORS.get(disease,'#888888'),label=disease)
            for i,(h,b) in enumerate(zip(heights,bottom)):
                if h:ax.text(i,b+h/2,f'{h}',ha='center',va='center',fontsize=7)
            bottom+=heights
        for i,total in enumerate(bottom):ax.text(i,total+.3,f'n={int(total)}',ha='center',fontsize=8)
        ax.set_xticks(np.arange(len(table)),table.index,rotation=50,ha='right');ax.set_ylabel('Samples');ax.set_title(tissue)
        ax.legend(frameon=False,fontsize=6,bbox_to_anchor=(1.01,1));save_figure(fig,a.output/f'Figure5{letter}_{tissue}_sample_counts')
        table.to_csv(a.output/f'Figure5{letter}_source_counts.csv')
    fig,axes=plt.subplots(1,2,figsize=(20,6),sharey=True)
    exports=[]
    for ax,tissue in zip(axes,['CSF','PBMC']):
        df=obs.loc[obs.tissue.astype(str).eq(tissue)]
        counts=pd.crosstab(df.Disease,df.cell_subtype_short).reindex(columns=ORDER,fill_value=0)
        totals=counts.sum(axis=1);prop=counts.div(totals,axis=0);bottom=np.zeros(len(prop))
        for subtype in ORDER:
            ax.bar(np.arange(len(prop)),prop[subtype],bottom=bottom,color=COLORS[subtype],label=subtype)
            bottom+=prop[subtype].to_numpy()
        labels=[f'{disease}\n(n={totals.loc[disease]:,})' for disease in prop.index]
        ax.set_xticks(np.arange(len(prop)),labels,rotation=60,ha='right',fontsize=7)
        ax.set_title(tissue);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylim(0,1)
        exports.append(prop.assign(tissue=tissue).reset_index())
    axes[0].set_ylabel('Pooled cell proportion');axes[-1].legend(frameon=False,fontsize=6,bbox_to_anchor=(1.01,1))
    save_figure(fig,a.output/'Figure5d_pooled_composition_by_tissue')
    pd.concat(exports).to_csv(a.output/'Figure5d_source_proportions.csv',index=False)


if __name__=='__main__':main()
