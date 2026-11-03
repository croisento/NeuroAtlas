"""Validate packaged sources, panel mappings and recorded numerical results."""
import ast,hashlib,importlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.common import REPOSITORY


def main():
    root=REPOSITORY/'analysis'
    files=list(root.rglob('*.py'))
    for path in files:ast.parse(path.read_text(),filename=str(path))
    inventory=json.loads((root/'source_inventory.json').read_text())
    for row in inventory['historical_sources']:
        path=REPOSITORY/row['file'];assert path.is_file()
        if 'sha256' in row:assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        else:
            notebook=json.loads(path.read_text());digest=hashlib.sha256()
            for cell in notebook['cells']:
                if cell['cell_type']=='code':
                    assert not cell.get('outputs')
                    s=cell.get('source',[]);digest.update((''.join(s) if isinstance(s,list) else s).encode())
            assert digest.hexdigest()==row['code_sha256']
    for record in inventory['dataset_readers']:
        assert (REPOSITORY/record['reader']).is_file()
        importlib.import_module('analysis.inputs.'+record['dataset'])
    panel=pd.read_csv(root/'figure_panel_map.csv')
    for path in panel.organized_code:assert (REPOSITORY/path).is_file(),path
    assert set(panel.figure.astype(str))=={'1','2','3','4','5','S1'}
    proportions=pd.read_csv(root/'data/sample_subtype_proportions.csv')
    assert proportions.Count.sum()==573569
    info=pd.read_csv(REPOSITORY/'data/sample_metadata.csv')
    pca=importlib.import_module('analysis.10_PCA_compositional_outliers').calculate(proportions,info)
    actual=pca.set_index(['GSE','GSM']).PCA_compositional_outlier.sort_index()
    expected=info.set_index(['GSE','GSM']).PCA_compositional_outlier.sort_index()
    assert actual.equals(expected)
    paired=importlib.import_module('analysis.12_paired_CSF_PBMC')
    reference=pd.read_csv(root/'data/paired_statistics_reference.csv')
    _,result=paired.run(proportions,pd.read_csv(REPOSITORY/'data/donor_matched_pairs.csv'),reference.cell_subtype_short.tolist())
    columns=['p_value','p_adjusted_BH','median_paired_difference_pp','bootstrap95CI_low_pp','bootstrap95CI_high_pp']
    for column in columns:assert np.allclose(result[column],reference[column],rtol=1e-10,atol=1e-12),column
    # Check sparse expression extraction against a small independent dense matrix.
    import tempfile,h5py,anndata as ad
    from scipy import sparse
    from analysis.expression import selected_logcounts
    with tempfile.TemporaryDirectory() as directory:
        x=np.array([[0.,1.,2.],[3.,0.,4.],[0.,0.,0.]],dtype=np.float32)
        toy=ad.AnnData(X=sparse.csr_matrix(x),obs=pd.DataFrame(index=['a','b','c']),var=pd.DataFrame(index=['A','B','C']))
        toy.layers['logcounts']=toy.X.copy();path=Path(directory)/'toy.h5ad';toy.write_h5ad(path)
        selected,_,genes=selected_logcounts(path,['C','A'])
        assert np.array_equal(selected.toarray(),x[:,[2,0]]) and genes.index.tolist()==['C','A']
    report={'python_sources_parse':len(files),'dataset_readers_import':11,
            'historical_source_hashes':'passed','figure_panel_map_rows':len(panel),
            'PCA_flags_match_all_205_samples':True,'paired_statistics_and_bootstrap_intervals_match':True,
            'sparse_expression_extraction':'passed',
            'full_upstream_analysis_rerun':False,'manual_annotation_recreation':False}
    (root/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
