"""Sample-level PCA outlier flags; no samples or cells are deleted."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from analysis.common import REPOSITORY


def calculate(proportions,metadata):
    matrix=proportions.pivot(index=['dataset','orig.ident','tissue'],columns='cell_subtype_short',values='Proportion_percent').fillna(0)
    info=metadata.set_index(['GSE','GSM','tissue'])
    assert matrix.index.is_unique and matrix.index.isin(info.index).all()
    info=info.loc[matrix.index].copy()
    info['PC1']=np.nan;info['PC2']=np.nan;info['PCA_distance_z']=np.nan
    info['PCA_compositional_outlier']='Not assessed'
    for (tissue,disease),group in info.reset_index().groupby(['tissue','Disease'],observed=True):
        if len(group)<5:continue
        ids=pd.MultiIndex.from_frame(group[['dataset','orig.ident','tissue']]) if 'dataset' in group else pd.MultiIndex.from_frame(group[['GSE','GSM','tissue']])
        coords=PCA(n_components=2).fit_transform(StandardScaler().fit_transform(matrix.loc[ids]))
        distance=np.sqrt((coords**2).sum(axis=1));z=zscore(distance,ddof=0)
        z=np.nan_to_num(z,nan=0.)
        info.loc[ids,'PC1']=coords[:,0];info.loc[ids,'PC2']=coords[:,1]
        info.loc[ids,'PCA_distance_z']=z
        info.loc[ids,'PCA_compositional_outlier']=np.where(z>1.5,'Flagged','Not flagged')
    return info.reset_index()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proportions',type=Path,default=REPOSITORY/'analysis/data/sample_subtype_proportions.csv')
    p.add_argument('--metadata',type=Path,default=REPOSITORY/'data/sample_metadata.csv')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    result=calculate(pd.read_csv(a.proportions),pd.read_csv(a.metadata))
    result.to_csv(a.output,index=False)
    print(result.PCA_compositional_outlier.value_counts())


if __name__=='__main__':main()
