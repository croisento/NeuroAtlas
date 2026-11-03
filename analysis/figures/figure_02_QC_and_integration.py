"""Figure 2a–d: frozen cell QC metrics and dataset/tissue UMAPs."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analysis.common import arguments,obs_umap,REPOSITORY,save_figure,setup_style
from analysis.figures.helpers import discrete_figure,STYLE


def main():
    p=arguments(__doc__);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);setup_style()
    obs,xy=obs_umap(a.input)
    # These raw-total/ribosomal vectors were generated from layers/counts.
    frozen=np.load(REPOSITORY/'data/atlas_metadata.npz')
    assert np.array_equal(xy.astype(np.float32),frozen['X_umap'])
    metrics=['n_genes_by_counts','total_counts','pct_counts_mt','pct_counts_rb']
    frame=obs[['dataset','tissue']].copy()
    for key in metrics:frame[key]=frozen[key+'_values']
    for letter,tissue in [('a','CSF'),('b','PBMC')]:
        df=frame.loc[frame.tissue.astype(str).eq(tissue)]
        fig,axes=plt.subplots(1,4,figsize=(20,4))
        for ax,key in zip(axes,metrics):
            sns.violinplot(data=df,x='dataset',y=key,inner='quartile',cut=0,ax=ax,
                           hue='dataset',legend=False,palette=STYLE['dataset_colors'])
            ax.tick_params(axis='x',rotation=65);ax.set_xlabel('Dataset');ax.set_title(key)
        save_figure(fig,a.output/f'Figure2{letter}_{tissue}_QC')
    discrete_figure(xy,obs.dataset,STYLE['dataset_colors'],'Source dataset',a.output/'Figure2c_dataset_UMAP')
    discrete_figure(xy,obs.tissue,{'CSF':'#596FA8','PBMC':'#4F8751'},'Tissue',a.output/'Figure2d_tissue_UMAP')


if __name__=='__main__':main()
