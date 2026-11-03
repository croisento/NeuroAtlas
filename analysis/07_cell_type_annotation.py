"""Apply reviewed per-cell annotations; no automatic disease or cell-type inference."""
import pandas as pd
import anndata as ad
from analysis.common import arguments,refuse_overwrite


def main():
    p=arguments(__doc__);p.add_argument('--annotations',required=True,help='CSV indexed by cell_id, with cell_type and cell_subtype_short')
    a=p.parse_args();refuse_overwrite(a.output)
    labels=pd.read_csv(a.annotations,index_col=0)
    assert labels.index.is_unique and {'cell_type','cell_subtype_short'}.issubset(labels)
    x=ad.read_h5ad(a.input)
    assert x.obs_names.isin(labels.index).all(),'Every retained cell requires a reviewed annotation'
    for column in labels:x.obs[column]=pd.Categorical(labels.loc[x.obs_names,column].to_numpy())
    x.write_h5ad(a.output,compression='gzip')


if __name__=='__main__':main()
