"""Apply the recorded sample-exclusion list and curated sample metadata."""
import pandas as pd
import anndata as ad
from analysis.common import arguments,refuse_overwrite,REPOSITORY


def main():
    p=arguments(__doc__)
    p.add_argument('--metadata',default=REPOSITORY/'data/sample_metadata.csv')
    p.add_argument('--exclusions',default=REPOSITORY/'analysis/config/sample_exclusions.csv')
    a=p.parse_args();refuse_overwrite(a.output)
    metadata=pd.read_csv(a.metadata,keep_default_na=False).set_index(['GSE','GSM'])
    excluded=pd.read_csv(a.exclusions)
    assert not excluded.dataset.eq('GSE186802').any()
    x=ad.read_h5ad(a.input)
    keys=pd.MultiIndex.from_arrays([x.obs.dataset.astype(str),x.obs['orig.ident'].astype(str)])
    banned=pd.MultiIndex.from_frame(excluded[['dataset','GSM']])
    x=x[~keys.isin(banned)].copy()
    keys=pd.MultiIndex.from_arrays([x.obs.dataset.astype(str),x.obs['orig.ident'].astype(str)])
    assert keys.isin(metadata.index).all(),'Input contains samples outside curated metadata'
    for column in ['Disease','Disease1','age','gender','tissue']:
        x.obs[column]=pd.Categorical(metadata.loc[keys,column].astype(str).to_numpy())
    x.obs['age_numeric']=pd.to_numeric(x.obs.age.astype(str),errors='coerce')
    if 'disease' in x.obs:del x.obs['disease']
    x.write_h5ad(a.output,compression='gzip')


if __name__=='__main__':main()
