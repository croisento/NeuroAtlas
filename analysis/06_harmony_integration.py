"""Reprocess retained cells and compute Harmony, neighbors, Leiden and UMAP."""
import anndata as ad
from analysis.common import arguments,refuse_overwrite
from analysis.preprocessing import PROFILES,prepare,harmony


def main():
    p=arguments(__doc__);p.add_argument('--profile',choices=PROFILES,required=True)
    a=p.parse_args();refuse_overwrite(a.output)
    x=harmony(prepare(ad.read_h5ad(a.input),PROFILES[a.profile]),PROFILES[a.profile])
    x.write_h5ad(a.output,compression='gzip')


if __name__=='__main__':main()
