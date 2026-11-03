"""Exercise import, merging, QC, review, annotation and metadata on synthetic data."""
import json,subprocess,sys,tempfile
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from analysis.common import REPOSITORY


def execute(module,*args):
    result=subprocess.run([sys.executable,'-m',module,*map(str,args)],cwd=REPOSITORY,
                          capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stdout+'\n'+result.stderr)


def main():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        raw=root/'GSE133028_RAW';raw.mkdir()
        matrix=pd.DataFrame([[1,2],[3,4],[5,6]],index=['A','B','C'],columns=['c1','c2'])
        matrix.to_csv(raw/'GSM000001_CSF_rawCounts.csv.gz',compression='gzip')
        execute('analysis.inputs.GSE133028','--base-dir',root)
        imported=ad.read_h5ad(root/'GSE133028_raw_count.h5ad')
        assert imported.shape==(2,3) and imported.obs['orig.ident'].eq('GSM000001').all()
        small=ad.AnnData(X=sparse.csr_matrix(np.ones((2,3))),obs=pd.DataFrame({'orig.ident':['S2']*2},index=['x','y']),var=pd.DataFrame(index=['B','C','D']))
        small.write_h5ad(root/'second.h5ad')
        (root/'manifest.json').write_text(json.dumps({'GSE133028':str(root/'GSE133028_raw_count.h5ad'),'GSETEST':str(root/'second.h5ad')}))
        execute('analysis.02_merge_shared_genes','--manifest',root/'manifest.json','--output',root/'merged.h5ad')
        merged=ad.read_h5ad(root/'merged.h5ad');assert merged.shape==(4,2) and set(merged.var_names)=={'B','C'}
        counts=np.ones((20,405),dtype=np.int32)
        counts[0,50:]=0;counts[1,0]=500
        obs=pd.DataFrame({'dataset':['GSETEST']*20,'orig.ident':['GSMTEST']*20},index=[f'cell{i}' for i in range(20)])
        toy=ad.AnnData(X=sparse.csr_matrix(counts),obs=obs,var=pd.DataFrame(index=['MT-GENE']+[f'G{i}' for i in range(404)]))
        toy.layers['counts']=toy.X.copy();toy.obsm['X_umap']=np.arange(40).reshape(20,2)
        toy.write_h5ad(root/'raw.h5ad')
        execute('analysis.03_cell_quality_control','--input',root/'raw.h5ad','--output',root/'qc.h5ad')
        qc=ad.read_h5ad(root/'qc.h5ad');assert qc.shape==(18,405)
        assert 'cell0' not in qc.obs_names and 'cell1' not in qc.obs_names
        pd.DataFrame({'cell_id':['cell2'],'reason':['reviewed non-target population']}).to_csv(root/'exclude.csv',index=False)
        execute('analysis.05_apply_contaminant_review','--input',root/'qc.h5ad','--excluded-cells',root/'exclude.csv','--output',root/'clean.h5ad')
        clean=ad.read_h5ad(root/'clean.h5ad');assert clean.n_obs==17
        annotations=pd.DataFrame({'cell_type':'T cell','cell_subtype_short':'CD4 Memory'},index=clean.obs_names)
        annotations.index.name='cell_id';annotations.to_csv(root/'annotations.csv')
        execute('analysis.07_cell_type_annotation','--input',root/'clean.h5ad','--annotations',root/'annotations.csv','--output',root/'annotated.h5ad')
        pd.DataFrame([dict(GSE='GSETEST',GSM='GSMTEST',Disease='Younger control',Disease1='Younger Control (age < 60 years)',age=20,gender='Female',tissue='CSF')]).to_csv(root/'metadata.csv',index=False)
        pd.DataFrame(columns=['dataset','GSM']).to_csv(root/'samples_to_exclude.csv',index=False)
        execute('analysis.08_sample_filter_and_metadata','--input',root/'annotated.h5ad','--metadata',root/'metadata.csv','--exclusions',root/'samples_to_exclude.csv','--output',root/'metadata.h5ad')
        result=ad.read_h5ad(root/'metadata.h5ad')
        assert result.n_obs==17 and result.obs.Disease.eq('Younger control').all()
        assert (result.X!=clean.X).nnz==0 and np.array_equal(result.obsm['X_umap'],clean.obsm['X_umap'])
    print('PASS: dataset import, shared-gene merge, QC, reviewed exclusion, annotations and metadata')


if __name__=='__main__':main()
