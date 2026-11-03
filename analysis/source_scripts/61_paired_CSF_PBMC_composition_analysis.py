#!/usr/bin/env python3
"""Donor-matched composition comparisons using explicit IDs in GEO sample titles.

No cell filtering or expression analysis. Ambiguous multiple samples per
donor/compartment are excluded, rather than selected or pooled silently.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, false_discovery_control
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

PROJECT=Path('/home/lishaogang/NEU2026_July')
TABLES=PROJECT/'final/Tables/Paired_CSF_PBMC'
OUT=PROJECT/'final/Figures/Paired_CSF_PBMC'
DATASETS=['GSE133028','GSE138266','GSE194078','GSE243905']
FOCUS=['NK bright','NK dim','CD4 Memory']
sys.path.insert(0,'/home/lishaogang/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment


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


def export(fig,name):
    require_matplotlib_panel_alignment(fig,json_out=str(OUT/f'{name}.alignment.json'),strict=True)
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=400)
    plt.close(fig)


def main(excluded_samples=(), expected_pairs=39):
    TABLES.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    metadata=pd.read_csv(PROJECT/'final/Tables/final_sample_audit.csv')
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
    candidates.to_csv(TABLES/'sample_pairing_audit.csv',index=False)
    pairs=pd.DataFrame(pairs)
    assert len(pairs)==expected_pairs and not pairs.pair_id.duplicated().any()
    assert not set(pairs.PBMC_GSM).intersection(excluded_samples)
    pairs.to_csv(TABLES/'included_donor_pairs.csv',index=False)
    long=pd.read_csv(PROJECT/'final/Tables/Doublets_retained/Sample_cell_subtype_counts_and_proportions_doublets_retained.csv')
    assert long.Count.sum()==573569
    palette=pd.read_csv(PROJECT/'final/Tables/Doublets_retained/UMAP_matched_subtype_color_key.csv')
    order=palette.cell_subtype_short.tolist()
    index=long.pivot(index=['dataset','orig.ident','tissue'],columns='cell_subtype_short',values='Proportion_percent')
    left=np.array([index.loc[(r.dataset,r.CSF_GSM,'CSF'),order].to_numpy() for r in pairs.itertuples()])
    right=np.array([index.loc[(r.dataset,r.PBMC_GSM,'PBMC'),order].to_numpy() for r in pairs.itertuples()])
    assert np.isfinite(left).all() and np.isfinite(right).all()
    assert np.allclose(left.sum(axis=1),100) and np.allclose(right.sum(axis=1),100)
    deltas=pd.DataFrame(left-right,index=pairs.pair_id,columns=order)
    deltas.to_csv(TABLES/'paired_differences_CSF_minus_PBMC_percentage_points.csv')
    paired_source=pairs.copy()
    parts=[]
    for i,subtype in enumerate(order):
        part=paired_source.copy();part['cell_subtype_short']=subtype
        part['CSF_percent']=left[:,i];part['PBMC_percent']=right[:,i];part['paired_difference_pp']=left[:,i]-right[:,i]
        parts.append(part)
    pd.concat(parts).to_csv(TABLES/'paired_proportion_plot_source.csv',index=False)
    result=tests(deltas,'All donor-matched pairs')
    result.to_csv(TABLES/'paired_Wilcoxon_BH_all25.csv',index=False)
    sensitivity=[]
    for dataset in DATASETS:
        keep=pairs.dataset.eq(dataset).to_numpy()
        sensitivity.append(tests(deltas.iloc[keep],f'Within {dataset}'))
        sensitivity.append(tests(deltas.iloc[~keep],f'Exclude {dataset}'))
    control=pairs.Disease.isin(['Younger control','Older control']).to_numpy()
    sensitivity.append(tests(deltas.iloc[control],'Controls only'))
    sensitivity.append(tests(deltas.iloc[pairs.Disease.eq('MS').to_numpy()],'MS only'))
    pd.concat(sensitivity).to_csv(TABLES/'dataset_and_disease_sensitivity.csv',index=False)
    font=Path('/home/lishaogang/.fonts/arial.ttf')
    if font.exists():font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    colors=dict(zip(DATASETS,['#377eb8','#4daf4a','#984ea3','#e39d31']))
    fig,axes=plt.subplots(1,3,figsize=(12,5.8))
    for ax,subtype in zip(axes,FOCUS):
        j=order.index(subtype)
        for i,row in enumerate(pairs.itertuples()):
            ax.plot([0,1],[left[i,j],right[i,j]],color=colors[row.dataset],alpha=.6,lw=.8,marker='o',ms=3)
        stat=result[result.cell_subtype_short.eq(subtype)].iloc[0]
        ax.set(xticks=[0,1],xticklabels=['CSF','PBMC'],xlim=(-.2,1.2),ylabel='Within-sample cell proportion (%)',
               title=f'{subtype}\nBH-adjusted P={stat.p_adjusted_BH:.3g}',ylim=(0,max(left[:,j].max(),right[:,j].max())*1.12+1))
    handles=[Line2D([],[],color=colors[d],marker='o',lw=1,label=f'{d} (n={(pairs.dataset==d).sum()} pairs)') for d in DATASETS]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.10),ncol=2,frameon=False)
    fig.subplots_adjust(left=.07,right=.98,top=.86,bottom=.30,wspace=.32)
    fig.text(.07,.035,f'{len(pairs)} donor-matched pairs; paired Wilcoxon tests; BH across 25 subtypes. Same-visit sampling not verified.',fontsize=8)
    export(fig,'Paired_CSF_PBMC_NK_bright_NK_dim_CD4_memory')
    fig,ax=plt.subplots(figsize=(8,9))
    for i,row in enumerate(result.itertuples()):
        color='#b9364c' if row.p_adjusted_BH<.05 else '#777777'
        ax.plot([row.bootstrap95CI_low_pp,row.bootstrap95CI_high_pp],[i,i],color=color,lw=1)
        ax.scatter(row.median_paired_difference_pp,i,c=color,s=22,zorder=3)
    ax.axvline(0,color='#999999',ls='--',lw=.8)
    ax.set(yticks=np.arange(25),yticklabels=order,xlabel='Median paired difference: CSF - PBMC (percentage points)',
           title=f'Donor-matched cell-subtype proportions | {len(pairs)} pairs')
    ax.invert_yaxis()
    fig.subplots_adjust(left=.23,right=.96,top=.92,bottom=.16)
    fig.text(.08,.055,'Points: median within-donor differences; intervals: 95% paired-donor bootstrap (5,000 draws).\nRed: BH-adjusted P <0.05 across 25 paired tests. No doublet filtering; absent subtypes = 0%.',fontsize=8)
    export(fig,'Paired_CSF_PBMC_all25_effect_sizes')
    report={'cells_in_source':573569,'n_pairs':len(pairs),'n_samples_used':len(pairs)*2,
            'pairs_by_dataset':pairs.dataset.value_counts().to_dict(),'pairs_by_disease':pairs.Disease.value_counts().to_dict(),
            'pairing_basis':'explicit participant IDs in cached GEO sample titles; dataset-scoped IDs',
            'sampling_time_verified':False,
            'duplicate_sample_exclusions':list(excluded_samples),
            'ambiguous_donor_excluded':None if excluded_samples else 'GSE133028:31 (two PBMC records)',
            'GSE243905_note':'Blood titles say biol rep 2; exclude-GSE243905 sensitivity supplied',
            'doublets_removed':False,'PCA_outliers_removed':False,'BH_scope':'25 subtypes per analysis scope',
            'test':'two-sided scipy.stats.wilcoxon, zero_method=wilcox, correction=True, method=auto; differences rounded to 10 decimals',
            'bootstrap':'5,000 paired-donor draws, seed 0, percentile CI for median paired difference',
            'limitations':'Same-visit matching and cross-study donor uniqueness not independently verified. No causal tissue-effect inference.',
            'focus_results':result[result.cell_subtype_short.isin(FOCUS)].to_dict(orient='records')}
    (TABLES/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
