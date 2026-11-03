"""Normalize, log1p, select HVGs, scale, PCA and cluster before review."""
import anndata as ad
import scanpy as sc
from analysis.common import arguments,refuse_overwrite
from analysis.preprocessing import PROFILES,prepare,graph


def main():
    p=arguments(__doc__);p.add_argument('--profile',choices=PROFILES,required=True)
    a=p.parse_args();refuse_overwrite(a.output)
    x=graph(prepare(ad.read_h5ad(a.input),PROFILES[a.profile]),PROFILES[a.profile],'X_pca')
    sc.tl.rank_genes_groups(x,'leiden',method='wilcoxon',tie_correct=True,use_raw=False,layer='logcounts')
    sc.get.rank_genes_groups_df(x,None).to_csv(a.output.with_suffix('.cluster_markers.csv'),index=False)
    x.write_h5ad(a.output,compression='gzip')


if __name__=='__main__':main()
