"""Export compact Shiny data from the released atlas and publication tables."""
import argparse,json,shutil
from pathlib import Path
import h5py,numpy as np,pandas as pd
from scipy import sparse
from anndata.io import read_elem
from analysis.common import MASTER,REPOSITORY
from analysis.expression import selected_logcounts


def raw_total_and_ribosomal_counts(path):
    with h5py.File(path,'r') as f:
        var=read_elem(f['var']);node=f['layers/counts'];n,p=map(int,node.attrs['shape'])
        ribosomal=var.index.str.upper().str.startswith(('RPL','RPS'))
        total=np.zeros(n);ribo=np.zeros(n)
        for first in range(0,n,5000):
            last=min(first+5000,n);ptr=node['indptr'][first:last+1]
            start,end=int(ptr[0]),int(ptr[-1])
            block=sparse.csr_matrix((node['data'][start:end],node['indices'][start:end],ptr-start),shape=(last-first,p))
            block=block.astype(np.float64)
            total[first:last]=np.asarray(block.sum(axis=1)).ravel()
            ribo[first:last]=np.asarray(block[:,ribosomal].sum(axis=1)).ravel()
    percent=np.divide(ribo*100,total,out=np.zeros(n),where=total>0)
    return total,percent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=MASTER)
    parser.add_argument('--tables',type=Path,default=REPOSITORY/'data')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a new export directory')
    args.output.mkdir(parents=True)
    with h5py.File(args.input,'r') as f:
        obs=read_elem(f['obs']);xy=np.asarray(f['obsm/X_umap'][:],dtype=np.float32)
    sample=pd.read_csv(args.tables/'sample_metadata.csv').set_index(['GSE','GSM'])
    keys=pd.MultiIndex.from_arrays([obs.dataset.astype(str),obs['orig.ident'].astype(str)])
    assert keys.isin(sample.index).all() and len(obs)==573569
    categorical=obs[['orig.ident','dataset','tissue','gender','age','Disease','Disease1','cell_type','cell_subtype_short','predicted_doublet']].rename(columns={'orig.ident':'GSM','dataset':'GSE'})
    categorical['PCA_compositional_outlier']=sample.loc[keys,'PCA_compositional_outlier'].to_numpy()
    total,ribo=raw_total_and_ribosomal_counts(args.input)
    arrays={'X_umap':xy};levels={}
    for column in categorical:
        cat=pd.Categorical(categorical[column].astype(str))
        arrays[column+'_codes']=cat.codes.astype(np.int16);levels[column]=cat.categories.tolist()
    numeric={'n_genes_by_counts':obs.n_genes_by_counts.to_numpy(),'total_counts':total,
             'pct_counts_mt':obs.pct_counts_mt.to_numpy(),'pct_counts_rb':ribo,
             'doublet_score':obs.doublet_score.to_numpy(),'age_numeric':obs.age_numeric.to_numpy()}
    for column,values in numeric.items():arrays[column+'_values']=np.asarray(values,dtype=np.float32)
    np.savez_compressed(args.output/'atlas_metadata.npz',**arrays)
    (args.output/'metadata_categories.json').write_text(json.dumps(levels,ensure_ascii=False,indent=2)+'\n')
    tables=[pd.read_csv(args.tables/f'{key}_top50_marker_catalog.csv') for key in ('cell_type','cell_subtype_short')]
    genes=pd.concat(tables).gene.drop_duplicates().tolist()
    expression,_,_=selected_logcounts(args.input,genes)
    assert expression.shape[1]==len(genes),'Every marker gene must exist in the input object'
    quantized=np.rint(expression.data*1000)
    assert (quantized>=0).all() and quantized.max(initial=0)<=np.iinfo(np.uint16).max
    expression=sparse.csr_matrix((quantized.astype(np.uint16),expression.indices,expression.indptr),shape=expression.shape)
    parts=[];lookup={}
    for part,start in enumerate(range(0,len(genes),55)):
        names=genes[start:start+55];filename=f'expression_part_{part:02d}.npz'
        sparse.save_npz(args.output/filename,expression[:,start:start+55].tocsc(),compressed=True)
        parts.append(dict(part=part,file=filename,n_genes=len(names),genes=names))
        for column,gene in enumerate(names):lookup[gene.upper()]=dict(part=part,column=column)
    for filename in ['sample_metadata.csv','donor_matched_pairs.csv','paired_proportions.csv','paired_statistics.csv',
                     'cell_type_top50_marker_catalog.csv','cell_subtype_short_top50_marker_catalog.csv']:
        shutil.copy2(args.tables/filename,args.output/filename)
    manifest=dict(n_cells=len(obs),n_genes=len(genes),obs_columns=list(levels)+list(numeric),
                  numeric_obs_columns=list(numeric),expression_scale=1000,
                  expression_units='log1p normalized counts per 10,000',expression_parts=parts,
                  gene_lookup=lookup,source_h5ad=args.input.name,n_CSF_samples=161,n_donor_matched_pairs=40)
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    assert all(path.stat().st_size<100_000_000 for path in args.output.iterdir() if path.is_file())
    print(f'Exported {len(obs):,} cells and {len(genes)} marker genes to {args.output}')


if __name__=='__main__':main()
