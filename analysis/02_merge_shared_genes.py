"""Concatenate prepared raw-count datasets using the gene-name intersection."""
import argparse,json
from pathlib import Path
import anndata as ad
import scanpy as sc
from analysis.common import refuse_overwrite


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True,help='JSON mapping GSE to prepared h5ad path')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();refuse_overwrite(a.output)
    datasets={}
    for gse,path in json.loads(a.manifest.read_text()).items():
        x=ad.read_h5ad(path);x.var_names_make_unique()
        sample_column={'GSE134577':'GSM','GSE186802':'GSM_ID_1','GSE194078':'GSM_ID',
                       'GSE243905':'GSM_ID','GSE247134':'GSM_ID'}.get(gse)
        if sample_column is not None:
            if sample_column not in x.obs:raise ValueError(f'{gse}: missing {sample_column}')
            x.obs['orig.ident']=x.obs[sample_column].copy()
        if 'orig.ident' not in x.obs:
            if 'GSM_ID' not in x.obs:raise ValueError(f'{gse}: sample identifiers required')
            x.obs['orig.ident']=x.obs.GSM_ID.copy()
        sc.pp.filter_genes(x,min_cells=1)
        datasets[gse]=x
    merged=ad.concat(datasets,join='inner',label='dataset',index_unique='-')
    merged.layers['counts']=merged.X.copy()
    merged.write_h5ad(a.output,compression='gzip')
    print(merged)


if __name__=='__main__':main()
