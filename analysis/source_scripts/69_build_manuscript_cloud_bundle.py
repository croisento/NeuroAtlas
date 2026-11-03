#!/usr/bin/env python3
"""Refresh the cloud bundle from final atlas and verified publication records."""
from pathlib import Path
import importlib.util
import json
import shutil
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from anndata.io import read_elem

ROOT = Path('/home/lishaogang/NEU2026_July')
APP = ROOT/'final/ShinyAppsCloud'
DATA = APP/'data'
MASTER = ROOT/'final/h5ad/adata_final_Disease_corrected_min400_except_CART_scrublet_identified.h5ad'
spec = importlib.util.spec_from_file_location('slim_helpers',ROOT/'Code/30_build_web_slim_h5ad.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def main():
    old_arrays = np.load(DATA/'atlas_metadata.npz')
    old_categories = json.loads((DATA/'metadata_categories.json').read_text())
    with h5py.File(MASTER,'r') as f:
        obs = read_elem(f['obs'])
        xy = np.asarray(f['obsm/X_umap'][:],dtype=np.float32)
        genes = helper.read_h5ad_index(f['var'])
    assert len(obs)==573569 and 'disease' not in obs
    assert np.array_equal(xy,old_arrays['X_umap'])
    for cloud,source in [('GSM','orig.ident'),('GSE','dataset')]:
        decoded=np.asarray(old_categories[cloud])[old_arrays[cloud+'_codes']]
        assert np.array_equal(decoded,obs[source].astype(str).to_numpy())
    web_obs=obs[['orig.ident','dataset','tissue','gender','age','Disease','Disease1','cell_type','cell_subtype_short','doublet_score','predicted_doublet']].rename(columns={'orig.ident':'GSM','dataset':'GSE'})
    samples=web_obs.groupby(['GSE','GSM'],observed=True).agg(
        tissue=('tissue','first'),Disease=('Disease','first'),Disease1=('Disease1','first'),
        age=('age','first'),gender=('gender','first'),Atlas_cells=('GSM','size'),
        predicted_doublets=('predicted_doublet','sum')).reset_index()
    clinical=pd.read_excel(ROOT/'final/Table2 Final_all_GSM_QC_audit_unio_NEW3(1).xlsx')
    assert not clinical.duplicated(['dataset','GSM']).any()
    clinical=clinical.rename(columns={'dataset':'GSE'})[['GSE','GSM','Clinical Manifestation']]
    samples=samples.merge(clinical,on=['GSE','GSM'],how='left',validate='one_to_one',indicator=True)
    assert samples['_merge'].eq('both').all()
    samples=samples.drop(columns='_merge').rename(columns={'Clinical Manifestation':'clinical_manifestation'})
    samples['clinical_manifestation']=samples.clinical_manifestation.fillna('Not reported').astype(str)
    pairs_dir=ROOT/'final/Tables/Paired_CSF_PBMC_40pairs_deduplicated'
    pairs=pd.read_csv(pairs_dir/'included_donor_pairs.csv')
    pairing=pd.read_csv(pairs_dir/'sample_pairing_audit.csv')
    samples=samples.merge(pairing[['GSM','dataset','donor_id','pair_id']].rename(columns={'dataset':'GSE'}),on=['GSE','GSM'],how='left',validate='one_to_one')
    partners={}
    for r in pairs.itertuples():
        partners[(r.dataset,r.CSF_GSM)]=r.PBMC_GSM
        partners[(r.dataset,r.PBMC_GSM)]=r.CSF_GSM
    samples['paired_sample']= [partners.get((r.GSE,r.GSM),'') for r in samples.itertuples()]
    samples['included_in_paired_analysis']=samples.paired_sample.ne('')
    samples['pairing_status']=np.where(samples.included_in_paired_analysis,'Included in 40 donor-matched pairs','No retained counterpart in paired analysis')
    samples.loc[samples.GSM.eq('GSM5264668'),'pairing_status']='Duplicate PBMC record; excluded from paired analysis'
    samples.loc[samples.GSM.eq('GSM5264668'),'paired_sample']='GSM5264667 (donor match; duplicate PBMC excluded)'
    assert samples.included_in_paired_analysis.sum()==80
    csf=pd.read_csv(ROOT/'final/Figures/Supplement_Figure1/Supplement_Figure1C/SupplementaryFigure1C_CSF_PCA_outlier_summary.csv')
    pbmc=pd.read_csv(ROOT/'final/Figures/Supplement_Figure1/Supplement_Figure1D/PBMC_PCA_outlier_samples_by_disease.csv')
    samples['PCA_compositional_outlier']='Not assessed'
    samples['PCA_distance_z']=np.nan
    for r in csf.itertuples():
        hit=samples.tissue.eq('CSF')&samples.GSM.eq(str(r.sample))
        assert hit.sum()==1
        samples.loc[hit,'PCA_compositional_outlier']='Flagged' if r.is_outlier else 'Not flagged'
        samples.loc[hit,'PCA_distance_z']=r.pca_distance_z
    for r in pbmc.itertuples():
        hit=samples.tissue.eq('PBMC')&samples.Disease.astype(str).eq(r.disease)
        assert hit.sum()==r.n_samples
        samples.loc[hit,'PCA_compositional_outlier']='Not flagged'
        flagged=[] if pd.isna(r.outlier_samples) else str(r.outlier_samples).split(';')
        samples.loc[hit&samples.GSM.isin(flagged),'PCA_compositional_outlier']='Flagged'
    assert samples.PCA_compositional_outlier.eq('Flagged').sum()==13
    assert samples.PCA_compositional_outlier.ne('Not assessed').sum()==177
    samples['predicted_doublet_fraction']=samples.predicted_doublets/samples.Atlas_cells
    samples.to_csv(DATA/'sample_metadata.csv',index=False)
    lookup=samples.set_index(['GSE','GSM']).PCA_compositional_outlier
    web_obs['PCA_compositional_outlier']=[lookup.loc[(str(gse),str(gsm))] for gse,gsm in zip(web_obs.GSE,web_obs.GSM)]
    numeric=['n_genes_by_counts','total_counts','pct_counts_mt','pct_counts_rb','doublet_score','age_numeric']
    arrays={'X_umap':xy}
    for key in numeric[:4]:
        arrays[key+'_values']=old_arrays[key+'_values']
    arrays['doublet_score_values']=obs.doublet_score.to_numpy(dtype=np.float32)
    arrays['age_numeric_values']=obs.age_numeric.to_numpy(dtype=np.float32)
    categories={}
    for key in web_obs:
        if key=='doublet_score':continue
        series=web_obs[key].astype(str)
        cat=pd.Categorical(series)
        categories[key]=cat.categories.tolist()
        arrays[key+'_codes']=cat.codes.astype(np.int16)
    np.savez_compressed(DATA/'atlas_metadata.npz',**arrays)
    (DATA/'metadata_categories.json').write_text(json.dumps(categories,ensure_ascii=False,indent=2)+'\n')
    markers={}
    for key in ['cell_type','cell_subtype_short']:
        table=pd.read_csv(ROOT/f'final/Tables/DEG_top50_Wilcoxon_filtered/{key}_Top50_markers.csv').rename(columns={'names':'gene'})
        markers[key]=table
        table.to_csv(DATA/f'{key}_top50_marker_catalog.csv',index=False)
    combined=pd.concat(list(markers.values()),ignore_index=True)
    selected=combined.gene.drop_duplicates().tolist()
    gene_lookup={g:i for i,g in enumerate(genes)}
    with h5py.File(MASTER,'r') as f:
        expression=helper.extract_csr_columns(f['layers/logcounts'],np.asarray([gene_lookup[g] for g in selected]),5000)
    parts=[]; lookup={}
    for part,start in enumerate(range(0,len(selected),55)):
        chunk_genes=selected[start:start+55]
        filename=f'expression_part_{part:02d}.npz'
        sparse.save_npz(DATA/filename,expression[:,start:start+55].tocsc(),compressed=True)
        parts.append(dict(part=part,file=filename,n_genes=len(chunk_genes),genes=chunk_genes))
        for col,gene in enumerate(chunk_genes):lookup[gene.upper()]={'part':part,'column':col}
    for r in pairs.itertuples():
        pairs.loc[pairs.pair_id.eq(r.pair_id),'Disease']=samples.loc[samples.GSE.eq(r.dataset)&samples.GSM.eq(r.CSF_GSM),'Disease'].astype(str).iloc[0]
    pairs.to_csv(DATA/'donor_matched_pairs.csv',index=False)
    paired_source=pd.read_csv(pairs_dir/'paired_proportion_plot_source.csv')
    paired_source['Disease']=paired_source.pair_id.map(pairs.set_index('pair_id').Disease)
    paired_source.to_csv(DATA/'paired_proportions.csv',index=False)
    shutil.copy2(pairs_dir/'paired_Wilcoxon_BH_all25.csv',DATA/'paired_statistics.csv')
    manifest={'n_cells':len(obs),'n_genes':len(selected),'obs_columns':list(categories)+numeric,
              'numeric_obs_columns':numeric,'expression_scale':1000,'expression_units':'log1p normalized counts per 10,000',
              'expression_parts':parts,'gene_lookup':lookup,'source_h5ad':MASTER.name,
              'marker_method':'Two-sided tie-corrected Wilcoxon rank-sum; BH over 12890 genes per comparison; detected fraction >=0.25 in either group; estimated log2 fold change >=0.25; adjusted P<0.05; positive statistic; top 50 by statistic',
              'marker_expression_scope':'Union of top-50 markers for cell_type and cell_subtype_short',
              'n_CSF_samples':161,'n_donor_matched_pairs':40}
    (DATA/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    report={'n_cells':len(obs),'n_samples':len(samples),'n_pairs':len(pairs),'n_unique_marker_genes':len(selected),
            'n_predicted_doublets':int(obs.predicted_doublet.sum()),'n_PCA_flagged_samples':13,
            'n_PCA_assessed_samples':177,'n_clinical_rows_matched':len(samples),
            'n_clinical_manifestation_not_reported':int(samples.clinical_manifestation.eq('Not reported').sum()),
            'cell_order_and_umap_verified':True,'numeric_QC_arrays_preserved':True,
            'marker_tables':{k:len(v) for k,v in markers.items()}}
    (APP/'manuscript_bundle_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    files=[{'file':p.name,'bytes':p.stat().st_size} for p in sorted(DATA.iterdir()) if p.is_file()]
    assert all(p['bytes']<100_000_000 for p in files)
    (APP/'cloud_data_report.json').write_text(json.dumps({'files':files,'total_bytes':sum(p['bytes'] for p in files),'all_files_below_github_limit':True},indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
