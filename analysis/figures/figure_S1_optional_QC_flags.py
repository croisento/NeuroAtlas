"""Figure S1a–c: tissue-specific PCA flags and retained Scrublet predictions."""
import importlib,pandas as pd,numpy as np
import matplotlib.pyplot as plt
from analysis.common import arguments,obs_umap,REPOSITORY,save_figure,setup_style


def main():
    p=arguments(__doc__);a=p.parse_args();setup_style();a.output.mkdir(parents=True,exist_ok=True)
    module=importlib.import_module('analysis.10_PCA_compositional_outliers')
    result=module.calculate(pd.read_csv(REPOSITORY/'analysis/data/sample_subtype_proportions.csv'),pd.read_csv(REPOSITORY/'data/sample_metadata.csv'))
    result.to_csv(a.output/'FigureS1_PCA_source.csv',index=False)
    for letter,tissue in [('a','CSF'),('b','PBMC')]:
        frame=result.loc[result.tissue.eq(tissue)&result.PCA_compositional_outlier.ne('Not assessed')]
        groups=list(frame.groupby('Disease',observed=True));cols=min(3,len(groups));rows=int(np.ceil(len(groups)/cols))
        fig,axes=plt.subplots(rows,cols,figsize=(5*cols,4*rows),squeeze=False)
        for ax,(disease,group) in zip(axes.flat,groups):
            flagged=group.PCA_compositional_outlier.eq('Flagged')
            ax.scatter(group.PC1,group.PC2,c=np.where(flagged,'#D1495B','#4C78A8'))
            for r in group.loc[flagged].itertuples():ax.annotate(r.GSM,(r.PC1,r.PC2),fontsize=6)
            ax.set_title(disease);ax.set_xlabel('PC1');ax.set_ylabel('PC2')
        for ax in list(axes.flat)[len(groups):]:ax.set_visible(False)
        fig.suptitle(f'{tissue}: PCA compositional outliers');fig.tight_layout()
        save_figure(fig,a.output/f'FigureS1{letter}_{tissue}_PCA')
    obs,xy=obs_umap(a.input);fig,axes=plt.subplots(1,2,figsize=(12,5))
    predicted=obs.predicted_doublet.astype(bool).to_numpy()
    for flag,color,label in [(False,'#B7BDC5','Predicted singlet'),(True,'#D1495B','Predicted doublet')]:
        mask=predicted==flag;axes[0].scatter(xy[mask,0],xy[mask,1],s=.5,c=color,label=label,rasterized=True,linewidths=0)
    axes[0].legend(frameon=False,markerscale=8);axes[0].set_title('Predicted doublet')
    image=axes[1].scatter(xy[:,0],xy[:,1],c=obs.doublet_score,s=.5,cmap='viridis',vmin=0,vmax=1,rasterized=True,linewidths=0)
    fig.colorbar(image,ax=axes[1],label='Doublet score');axes[1].set_title('Doublet score')
    for ax in axes:ax.set_xticks([]);ax.set_yticks([]);ax.set_xlabel('UMAP1');ax.set_ylabel('UMAP2')
    save_figure(fig,a.output/'FigureS1c_doublet_labels_and_scores')


if __name__=='__main__':main()
