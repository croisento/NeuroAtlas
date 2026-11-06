"""Pair matching and Wilcoxon/bootstrap implementation from 61, with 65's exclusion.

GSM5264668 is excluded from the paired analysis only; atlas cells are retained.
"""
import re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, false_discovery_control
DATASETS = ['GSE133028','GSE138266','GSE194078','GSE243905']


def calculate_from_atlas(path, metadata_path, titles_path, output, order):
    """Rebuild sample proportions and the original 40-pair comparison."""
    from figures.common import metadata_object
    output.mkdir(parents=True, exist_ok=True)
    obs = metadata_object(path).obs
    metadata = pd.read_csv(metadata_path, keep_default_na=False).rename(columns={'GSE':'dataset'})
    titles = pd.read_csv(titles_path, keep_default_na=False)
    metadata = metadata.merge(titles, on=['dataset','GSM'], how='left', validate='one_to_one')
    if metadata.GEO_sample_title.isna().any():
        raise ValueError('Missing GEO titles for donor matching')
    current = obs[['dataset','orig.ident','tissue','Disease','age','gender']].drop_duplicates()
    if current.duplicated(['dataset','orig.ident']).any():
        raise ValueError('Inconsistent sample metadata in input atlas')
    # Current h5ad labels take precedence over historical metadata fields.
    current = current.rename(columns={'orig.ident':'GSM'})
    metadata = current.merge(metadata[['dataset','GSM','GEO_sample_title']],
                             on=['dataset','GSM'], how='left', validate='one_to_one')
    pairs, audit = build_pairs(metadata)
    counts = obs.groupby(['dataset','orig.ident','tissue','cell_subtype_short'], observed=True).size().unstack(fill_value=0)
    if set(counts.columns) != set(order) or len(order)!=25:
        raise ValueError('Expected the confirmed 25 subtypes')
    proportions = counts.reindex(columns=order).div(counts.sum(axis=1), axis=0)*100
    left = np.array([proportions.loc[(r.dataset,r.CSF_GSM,'CSF'),order].to_numpy() for r in pairs.itertuples()])
    right = np.array([proportions.loc[(r.dataset,r.PBMC_GSM,'PBMC'),order].to_numpy() for r in pairs.itertuples()])
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Non-finite paired proportions')
    assert np.allclose(left.sum(axis=1),100) and np.allclose(right.sum(axis=1),100)
    deltas = pd.DataFrame(left-right,index=pairs.pair_id,columns=order)
    result = tests(deltas,'All donor-matched pairs')
    parts = []
    for i,subtype in enumerate(order):
        part = pairs.copy();part['cell_subtype_short']=subtype
        part['CSF_percent']=left[:,i];part['PBMC_percent']=right[:,i]
        part['paired_difference_pp']=left[:,i]-right[:,i];parts.append(part)
    source = pd.concat(parts,ignore_index=True)
    audit.to_csv(output/'sample_pairing_audit.csv',index=False)
    pairs.to_csv(output/'included_donor_pairs.csv',index=False)
    deltas.to_csv(output/'paired_differences_CSF_minus_PBMC_percentage_points.csv')
    result.to_csv(output/'paired_Wilcoxon_BH_all25.csv',index=False)
    source.to_csv(output/'paired_proportion_plot_source.csv',index=False)
    return source, result

def donor_id(row):
    title=row.GEO_sample_title
    if row.dataset=='GSE133028':
        match=re.fullmatch(r'(\d+)_(CSF|PB)_uns_[35]pGEX(?:_\d+)?',title)
    elif row.dataset in ['GSE138266','GSE194078']:
        match=re.fullmatch(r'([A-Za-z0-9]+)_(CSF|PBMC)',title)
    elif row.dataset=='GSE243905':
        match=re.fullmatch(r'Control (C\d+) (CSF|Blood) scRNAseq(?: \(biol rep \d+\))?',title)
    else:
        return None
    if not match:
        raise ValueError(f'Unrecognized title: {row.dataset}: {title}')
    source_tissue=match.group(2)
    assert ('CSF' if source_tissue=='CSF' else 'PBMC')==row.tissue
    return match.group(1)

def tests(deltas,scope):
    rng=np.random.default_rng(0)
    rows=[]
    for subtype in deltas.columns:
        d=deltas[subtype].dropna().to_numpy()
        # Round percentage-point differences solely for numerical tie handling.
        rounded=np.round(d,10)
        if len(d)<2:
            p,stat=np.nan,np.nan
        elif np.all(rounded==0):
            p,stat=1.,0.
        else:
            result=wilcoxon(rounded,alternative='two-sided',zero_method='wilcox',correction=True,method='auto')
            p,stat=float(result.pvalue),float(result.statistic)
        boot=np.median(rng.choice(d,size=(5000,len(d)),replace=True),axis=1) if len(d) else np.array([np.nan])
        lo,hi=np.nanpercentile(boot,[2.5,97.5])
        rows.append({'scope':scope,'cell_subtype_short':subtype,'n_pairs':len(d),
                     'median_paired_difference_pp':float(np.median(d)) if len(d) else np.nan,
                     'bootstrap95CI_low_pp':float(lo),'bootstrap95CI_high_pp':float(hi),
                     'n_CSF_higher':int((rounded>0).sum()),'n_PBMC_higher':int((rounded<0).sum()),
                     'n_equal':int((rounded==0).sum()),'signed_rank_statistic':stat,'p_value':p})
    frame=pd.DataFrame(rows)
    frame['p_adjusted_BH']=np.nan
    valid=frame.p_value.notna()
    frame.loc[valid,'p_adjusted_BH']=false_discovery_control(frame.loc[valid,'p_value'],method='bh')
    return frame

def build_pairs(metadata, excluded_samples=("GSM5264668",), expected_pairs=40):
    candidates=metadata[metadata.dataset.isin(DATASETS)].copy()
    candidates['donor_id']=candidates.apply(donor_id,axis=1)
    candidates['pair_id']=candidates.dataset+':'+candidates.donor_id
    candidates['pairing_evidence']='shared explicit participant identifier in cached GEO sample titles'
    candidates['sampling_time_verified']=False
    candidates['decision']=''
    excluded = candidates.GSM.isin(excluded_samples)
    candidates.loc[excluded, 'decision'] = 'exclude: duplicate sample reported by Cantoni et al. 2025'
    pairs=[]
    for pair_id,group in candidates.loc[~excluded].groupby('pair_id',sort=True):
        sizes=group.tissue.value_counts()
        if sizes.get('CSF',0)!=1 or sizes.get('PBMC',0)!=1:
            reason='exclude: multiple samples within compartment' if sizes.max()>1 else 'exclude: one compartment absent'
            candidates.loc[group.index,'decision']=reason
            continue
        for column in ['Disease','gender','age']:
            values=set(group[column].dropna().astype(str))-{'Unknown'}
            if len(values)>1:
                raise ValueError(f'Discordant {column} for {pair_id}: {values}')
        candidates.loc[group.index,'decision']='include: unique donor-matched CSF/PBMC pair'
        csf=group[group.tissue=='CSF'].iloc[0];pbmc=group[group.tissue=='PBMC'].iloc[0]
        pairs.append({'pair_id':pair_id,'dataset':csf.dataset,'donor_id':csf.donor_id,'Disease':csf.Disease,
                      'CSF_GSM':csf.GSM,'PBMC_GSM':pbmc.GSM,'CSF_title':csf.GEO_sample_title,
                      'PBMC_title':pbmc.GEO_sample_title,'age':csf.age,'gender':csf.gender,
                      'sampling_time_verified':False})

    pairs=pd.DataFrame(pairs)
    assert len(pairs)==expected_pairs and not pairs.pair_id.duplicated().any()
    assert not set(pairs.PBMC_GSM).intersection(excluded_samples)


    return pairs, candidates
