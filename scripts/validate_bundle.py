"""Check deployed data contracts and Shiny HTTP startup without external services."""
from pathlib import Path
import importlib.util
import ast
from types import SimpleNamespace
import json
import os
import subprocess
import sys
import time
import urllib.request
import numpy as np
import pandas as pd
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]


def main():
    spec=importlib.util.spec_from_file_location('atlas_validation',ROOT/'app.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    m=module
    assert len(m.obs)==573569 and len(m.SAMPLE_METADATA)==205
    assert m.obs.predicted_doublet.astype(str).eq('True').sum()==6157
    assert m.SAMPLE_METADATA.PCA_compositional_outlier.value_counts().to_dict()=={'Not flagged':164,'Not assessed':28,'Flagged':13}
    assert m.SAMPLE_METADATA.included_in_paired_analysis.sum()==80
    assert len(m.PAIRS)==40 and m.PAIRS.pair_id.is_unique
    for pair in m.PAIRS.itertuples():
        for gsm,partner in [(pair.CSF_GSM,pair.PBMC_GSM),(pair.PBMC_GSM,pair.CSF_GSM)]:
            r=m.SAMPLE_METADATA.loc[m.SAMPLE_METADATA.GSE.eq(pair.dataset)&m.SAMPLE_METADATA.GSM.eq(gsm)].iloc[0]
            assert r.paired_sample==partner
    duplicate=m.SAMPLE_METADATA.loc[m.SAMPLE_METADATA.GSM.eq('GSM5264668')].iloc[0]
    assert not duplicate.included_in_paired_analysis
    clinical=pd.read_csv(ROOT/'data/sample_metadata.csv',keep_default_na=False)
    assert clinical.clinical_manifestation.str.len().gt(0).all()
    assert 'disease' not in m.obs
    ids=np.arange(len(m.obs))
    for key in ('GSE','Disease','Disease1','tissue'):
        pooled=m.composition_summary(ids,key)
        grouping=list(dict.fromkeys(['tissue',key]))
        assert np.allclose(pooled.groupby(grouping,observed=True).Proportion.sum(),1)
        median=m.composition_summary(ids,key,'median')
        assert median.Proportion.between(0,1).all()
    # Independently calculate a zero-filled median from a sample count matrix.
    counts=pd.crosstab([m.obs.GSE,m.obs.GSM],m.obs.cell_subtype_short)
    fractions=counts.div(counts.sum(axis=1),axis=0)
    expected=fractions.median()
    sample_medians=m.composition_summary(ids,'GSE','median')
    for gse in m.choices('GSE'):
        exp=fractions.loc[gse].median()
        got=sample_medians.loc[sample_medians.GSE.astype(str).eq(gse)].set_index('cell_subtype_short').Proportion
        assert np.allclose(exp.reindex(got.index),got)
    for key,n in [('cell_type',250),('cell_subtype_short',1250)]:
        tab=m.MARKER_TABLES[key]
        assert len(tab)==n and tab.groupby('group').size().eq(50).all()
        assert (tab.pvals_adj<.05).all() and (tab.logfoldchanges>=.25).all()
    assert m.PAIRED_PROPORTIONS.groupby('cell_subtype_short').pair_id.nunique().eq(40).all()
    assert len(m.PAIRED_STATISTICS)==25
    # Exercise plot builders with representative selections without a browser.
    tree=ast.parse((ROOT/'app.py').read_text())
    server_node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='server')
    functions={node.name:node for node in server_node.body if isinstance(node,ast.FunctionDef)}
    for name in ('paired_plot','composition_plot','umap_plot'):
        node=functions[name]
        node.decorator_list=[]
        namespace=dict(vars(m))
        namespace['filtered_indices']=lambda:ids
        namespace['display_indices']=lambda:ids[:1000]
        if name=='paired_plot':
            namespace['input']=SimpleNamespace(paired_subtypes=lambda:m.SUBTYPE_ORDER)
        elif name=='composition_plot':
            namespace['input']=SimpleNamespace(composition_group=lambda:'GSE',composition_mode=lambda:'pooled')
        else:
            namespace['input']=SimpleNamespace(color_by=lambda:'PCA_compositional_outlier')
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/'app.py'),'exec'),namespace)
        figure=namespace[name]()
        assert len(figure.data)>0
        figure.to_json()
    for part in m.MANIFEST['expression_parts']:
        x=sparse.load_npz(ROOT/'data'/part['file'])
        assert x.shape==(573569,part['n_genes'])
    for gene in ('FOXP3','CD3E','SIGLEC6'):
        v=m.expression_vector(gene)
        assert len(v)==573569 and np.isfinite(v).all() and (v>=0).all()
    report={'data_contracts':'passed','composition_and_zero_filled_medians':'passed',
            'paired_all25_and_composition_and_PCA_plot_builders':'passed',
            'marker_and_expression_contracts':'passed','pairing_and_PCA_flags':'passed',
            'browser_interaction_test':'not run; Playwright is not installed'}
    log=ROOT/'validation_server.log'
    with log.open('w') as output:
        server=subprocess.Popen([sys.executable,'-m','shiny','run','--host','127.0.0.1','--port','18057',str(ROOT/'app.py')],stdout=output,stderr=subprocess.STDOUT,cwd=ROOT)
        try:
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            last_error=None
            for attempt in range(45):
                if server.poll() is not None:raise RuntimeError(log.read_text())
                try:
                    with opener.open('http://127.0.0.1:18057/',timeout=3) as response:
                        html=response.read().decode()
                    break
                except OSError as error:
                    last_error=error
                    time.sleep(1)
            else:raise RuntimeError(f'HTTP startup timeout: {last_error}')
            for text in ('neuroCSF-PBMC','Paired comparison','PCA compositional outlier','161 CSF samples','40 donor-matched pairs'):
                assert text in html,text
            assert '44 PBMC' not in html
            report['Shiny_HTTP_startup']='passed'
        finally:
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill();server.wait()
    (ROOT/'app_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
