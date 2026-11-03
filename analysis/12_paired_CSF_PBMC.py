"""Reproduce the 40 donor-matched, 25-subtype Wilcoxon comparisons."""
import argparse,ast
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon,false_discovery_control
from analysis.common import REPOSITORY


def paired_tests(deltas,scope='All donor-matched pairs'):
    rng=np.random.default_rng(0);rows=[]
    for subtype in deltas.columns:
        d=deltas[subtype].dropna().to_numpy();rounded=np.round(d,10)
        if np.all(rounded==0):p,stat=1.,0.
        else:
            test=wilcoxon(rounded,alternative='two-sided',zero_method='wilcox',correction=True,method='auto')
            p,stat=float(test.pvalue),float(test.statistic)
        boot=np.median(rng.choice(d,size=(5000,len(d)),replace=True),axis=1)
        lo,hi=np.percentile(boot,[2.5,97.5])
        rows.append(dict(scope=scope,cell_subtype_short=subtype,n_pairs=len(d),
            median_paired_difference_pp=float(np.median(d)),bootstrap95CI_low_pp=lo,bootstrap95CI_high_pp=hi,
            n_CSF_higher=int((rounded>0).sum()),n_PBMC_higher=int((rounded<0).sum()),n_equal=int((rounded==0).sum()),
            signed_rank_statistic=stat,p_value=p))
    result=pd.DataFrame(rows);result['p_adjusted_BH']=false_discovery_control(result.p_value,method='bh')
    return result


def run(proportions,pairs,order):
    assert len(pairs)==40 and pairs.pair_id.is_unique
    assert not pairs.PBMC_GSM.eq('GSM5264668').any()
    index=proportions.pivot(index=['dataset','orig.ident','tissue'],columns='cell_subtype_short',values='Proportion_percent')
    left=np.array([index.loc[(r.dataset,r.CSF_GSM,'CSF'),order].to_numpy() for r in pairs.itertuples()])
    right=np.array([index.loc[(r.dataset,r.PBMC_GSM,'PBMC'),order].to_numpy() for r in pairs.itertuples()])
    assert np.isfinite(left).all() and np.isfinite(right).all()
    assert np.allclose(left.sum(axis=1),100) and np.allclose(right.sum(axis=1),100)
    deltas=pd.DataFrame(left-right,index=pairs.pair_id,columns=order)
    return deltas,paired_tests(deltas)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proportions',type=Path,default=REPOSITORY/'analysis/data/sample_subtype_proportions.csv')
    p.add_argument('--pairs',type=Path,default=REPOSITORY/'data/donor_matched_pairs.csv')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    # Preserve the original subtype order for identical bootstrap draws.
    order=pd.read_csv(REPOSITORY/'analysis/data/paired_statistics_reference.csv').cell_subtype_short.tolist()
    deltas,result=run(pd.read_csv(a.proportions),pd.read_csv(a.pairs),order)
    deltas.to_csv(a.output/'paired_differences_percentage_points.csv')
    result.to_csv(a.output/'paired_statistics.csv',index=False)


if __name__=='__main__':main()
