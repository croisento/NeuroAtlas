#!/usr/bin/env python3
"""One-versus-rest atlas markers; retain all cells and existing annotations."""
from pathlib import Path
import json
import platform
import h5py
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import numba
from scipy import stats


def sparse_wilcoxon(atlas):
    """Exact rank sums for nonnegative sparse values, including all zero ties.

    Same two-sided normal approximation and tie variance as Scanpy Wilcoxon.
    Rank only nonzeros; zeros share rank (n_zero+1)/2. Reuse ranks across labels.
    """
    x = atlas.X
    x.eliminate_zeros()
    n, p = x.shape
    states = {}
    for key in ('cell_type', 'cell_subtype_short'):
        cat = atlas.obs[key].astype('category').cat
        codes = cat.codes.to_numpy()
        assert (codes >= 0).all()
        sizes = np.bincount(codes)
        states[key] = dict(codes=codes, groups=cat.categories, sizes=sizes,
                           scores=np.zeros((len(sizes), p)),
                           means=np.zeros((len(sizes), p)),
                           detected=np.zeros((len(sizes), p)))
    totals = np.zeros(p)
    nnz = np.diff(x.indptr)
    for j in range(p):
        a, b = x.indptr[j:j+2]
        v, rows = x.data[a:b], x.indices[a:b]
        z = n - len(v)
        ranks = stats.rankdata(v, method='average') + z
        zero_rank = (z + 1) / 2
        _, ties = np.unique(v, return_counts=True)
        ties = ties.astype(float)
        tie_factor = 1 - ((ties**3-ties).sum() + float(z)**3-z) / (float(n)**3-n)
        totals[j] = v.sum(dtype=np.float64)
        for s in states.values():
            codes = s['codes'][rows]
            k = len(s['sizes'])
            rank_sum = s['sizes'] * zero_rank + np.bincount(
                codes, weights=ranks-zero_rank, minlength=k)
            denom = np.sqrt(tie_factor * s['sizes'] * (n-s['sizes']) * (n+1)/12)
            s['scores'][:, j] = np.divide(rank_sum-s['sizes']*(n+1)/2, denom,
                                          out=np.zeros(k), where=denom>0)
            s['means'][:, j] = np.bincount(codes, weights=v, minlength=k)/s['sizes']
            s['detected'][:, j] = np.bincount(codes, minlength=k)
        if j % 1000 == 0 or j == p-1:
            print(f'Ranked {j+1}/{p} genes for both annotations', flush=True)
    results = {}
    for key, s in states.items():
        frames = []
        for i, group in enumerate(s['groups']):
            ng = s['sizes'][i]
            rest_mean = (totals-s['means'][i]*ng)/(n-ng)
            pval = 2*stats.norm.sf(np.abs(s['scores'][i]))
            adj = stats.false_discovery_control(pval, method='bh')
            frames.append(pd.DataFrame(dict(
                group=str(group), names=atlas.var_names,
                scores=s['scores'][i],
                logfoldchanges=np.log2((np.expm1(s['means'][i])+1e-9)/(np.expm1(rest_mean)+1e-9)),
                pvals=pval, pvals_adj=adj,
                pct_nz_group=s['detected'][i]/ng,
                pct_nz_reference=(nnz-s['detected'][i])/(n-ng))))
        results[key] = pd.concat(frames, ignore_index=True)
    return results

from analysis.common import MASTER, OUTPUT_ROOT
SOURCE = MASTER
OUT = OUTPUT_ROOT / 'markers'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('Reading log-normalized expression only', flush=True)
    with h5py.File(SOURCE, 'r') as f:
        x = ad.io.read_elem(f['layers/logcounts'])
        obs = ad.io.read_elem(f['obs'])[['cell_type', 'cell_subtype_short']]
        var = ad.io.read_elem(f['var'])
    print('Converting to column-sparse format for gene-wise ranking', flush=True)
    atlas = ad.AnnData(X=x.tocsc(), obs=obs, var=var)
    del x
    numba.set_num_threads(8)
    assert atlas.shape == (573569, 12890), atlas.shape
    assert np.isfinite(atlas.X.data).all() and atlas.X.data.min() >= 0
    results = sparse_wilcoxon(atlas)
    summary = []
    with pd.ExcelWriter(OUT / 'Top50_markers_Wilcoxon_filtered.xlsx') as writer:
        for groupby in ('cell_type', 'cell_subtype_short'):
            print(f'Running {groupby}: {atlas.obs[groupby].nunique()} groups', flush=True)
            result = results[groupby]
            result['passes_filter'] = (
                (result[['pct_nz_group', 'pct_nz_reference']].max(axis=1) >= .25)
                & (result['logfoldchanges'] >= .25)
                & (result['pvals_adj'] < .05)
                & (result['scores'] > 0)
            )
            result = result.sort_values(
                ['group', 'scores', 'pvals_adj', 'names'], ascending=[True, False, True, True]
            )
            filtered = result.loc[result.passes_filter].copy()
            top = filtered.groupby('group', observed=True, sort=False).head(50).copy()
            top.insert(1, 'rank', top.groupby('group', observed=True).cumcount() + 1)
            result.to_csv(OUT / f'{groupby}_all_genes_Wilcoxon.csv.gz', index=False)
            filtered.to_csv(OUT / f'{groupby}_significant_markers.csv', index=False)
            top.to_csv(OUT / f'{groupby}_Top50_markers.csv', index=False)
            top.to_excel(writer, sheet_name=groupby, index=False)
            for group in atlas.obs[groupby].cat.categories:
                n = int((top.group == group).sum())
                summary.append({'annotation': groupby, 'group': group,
                                'n_cells': int((atlas.obs[groupby] == group).sum()),
                                'n_passing_markers': int((filtered.group == group).sum()),
                                'n_exported': n})
            print(f'{groupby}: exported {len(top)} markers', flush=True)
    pd.DataFrame(summary).to_csv(OUT / 'marker_group_summary.csv', index=False)
    metadata = {
        'source': str(SOURCE), 'n_cells': atlas.n_obs, 'n_genes': atlas.n_vars,
        'expression': 'layers/logcounts; natural-log1p normalized expression',
        'test': 'Sparse implementation of Wilcoxon rank-sum; two-sided normal approximation with tie correction, no continuity correction (Scanpy convention); normal-approximation Wilcoxon statistics',
        'comparison': 'each annotation group versus all remaining cells',
        'correction': 'Benjamini-Hochberg across all 12890 genes per group, before filtering',
        'filter': 'max(pct_nz_group,pct_nz_reference)>=0.25; logfoldchanges>=0.25; pvals_adj<0.05; scores>0',
        'logfoldchanges': 'Scanpy approximate log2 fold change from expm1 of mean log-normalized expression',
        'ranking': 'decreasing signed Wilcoxon z score; up to 50 per group',
        'doublets': 'retained', 'experimental_unit': 'cells; descriptive annotation markers, not donor-level disease inference',
        'versions': {'python': platform.python_version(), 'scanpy': sc.__version__,
                     'anndata': ad.__version__, 'numpy': np.__version__,
                     'scipy': scipy.__version__, 'pandas': pd.__version__},
    }
    (OUT / 'marker_method.json').write_text(json.dumps(metadata, indent=2))
    print('Completed: ' + str(OUT), flush=True)


def cli():
    global SOURCE, OUT
    from analysis.common import arguments
    parser = arguments(__doc__)
    args = parser.parse_args()
    SOURCE = args.input
    OUT = args.output
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("Use a new or empty marker output directory")
    main()


if __name__ == '__main__':
    cli()
