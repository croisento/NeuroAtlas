"""Cell QC, per-dataset preliminary clustering and marker-expression inspection.

No contaminant classification or cluster deletion is performed.
"""
from pathlib import Path
import argparse
import anndata as ad
import pandas as pd
import scanpy as sc
from analysis.common import refuse_overwrite
import numpy as np
from analysis.blood_contamination import (
    SEED, MARKERS, available, dense, sum_axis, molecular_metrics,
)


def cell_qc(x):
    if 'counts' in x.layers:
        x.X = x.layers['counts'].copy()
    x.var['mt'] = x.var_names.str.upper().str.startswith('MT-')
    sc.pp.calculate_qc_metrics(x, qc_vars=['mt'], percent_top=None, inplace=True)
    x = x[(x.obs.n_genes_by_counts >= 400) & (x.obs.pct_counts_mt < 20)].copy()
    sc.pp.filter_genes(x, min_cells=3)
    return x


def screen_dataset(x, dataset, output):
    x = x[:, sum_axis(x.X, axis=0) > 0].copy()
    metrics = molecular_metrics(x)
    for col in metrics:
        x.obs[col] = metrics[col].to_numpy()
    present = available(x.var_names, MARKERS)
    raw = dense(x[:, present].X).astype(np.float32)
    logs = np.log1p(raw / np.maximum(sum_axis(x.X)[:, None], 1) * 10000)
    normalized = x.copy()
    normalized.X = normalized.X.astype(np.float32)
    sc.pp.normalize_total(normalized, target_sum=10000)
    sc.pp.log1p(normalized)
    sc.pp.highly_variable_genes(normalized, n_top_genes=min(2000, x.n_vars), flavor='seurat')
    b = normalized[:, normalized.var.highly_variable.to_numpy() | normalized.var_names.isin(present)].copy()
    sc.pp.scale(b, zero_center=False, max_value=None)
    sc.tl.pca(b, n_comps=30, zero_center=True, svd_solver='covariance_eigh', random_state=SEED)
    sc.pp.neighbors(b, n_neighbors=15, n_pcs=30, metric='euclidean', random_state=SEED, transformer='pynndescent')
    sc.tl.leiden(b, resolution=1.5, random_state=SEED, key_added='leiden_screen', flavor='igraph', n_iterations=2, directed=False)
    x.obs['leiden_screen'] = b.obs['leiden_screen']
    x.obs['screen_cluster'] = dataset + ':' + x.obs.leiden_screen.astype(str)
    x.obsm['X_pca_screen'] = b.obsm['X_pca']
    summary = x.obs.groupby('leiden_screen', observed=True).size().rename('n_cells').reset_index()
    summary['dataset'] = dataset
    rows = []
    for cluster in x.obs.leiden_screen.cat.categories:
        selected = x.obs.leiden_screen.astype(str).to_numpy() == str(cluster)
        for j, gene in enumerate(present):
            rows.append({'dataset': dataset, 'cluster': str(cluster), 'gene': gene,
                         'mean_log1p_cp10k': float(logs[selected, j].mean()),
                         'pct_cells_detected': float((raw[selected, j] > 0).mean() * 100)})
    markers = pd.DataFrame(rows)
    if present:
        normalized.obs['leiden_screen'] = x.obs.leiden_screen
        plot = sc.pl.dotplot(normalized, present, groupby='leiden_screen',
                             use_raw=False, standard_scale='var', return_fig=True)
        plot.savefig(output / f'{dataset}_cluster_marker_dotplot.pdf')
        import matplotlib.pyplot as plt
        plt.close('all')
    return x, summary, markers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New output directory')
    a = p.parse_args()
    refuse_overwrite(a.output)
    a.output.mkdir()
    x = ad.read_h5ad(a.input)
    if not {'dataset', 'orig.ident'}.issubset(x.obs.columns):
        raise ValueError('Input requires dataset and orig.ident metadata')
    x = cell_qc(x)
    audit = x.obs.groupby(['dataset', 'orig.ident'], observed=True).size().rename('cells_after_cell_QC').reset_index()
    audit['sample_keep_min400'] = (audit.cells_after_cell_QC >= 400) | audit.dataset.eq('GSE186802')
    audit.to_csv(a.output / 'sample_min400_audit.csv', index=False)
    summaries = []
    for dataset in x.obs.dataset.astype(str).unique():
        samples = audit.loc[audit.dataset.eq(dataset) & audit.sample_keep_min400, 'orig.ident']
        part = x[x.obs.dataset.astype(str).eq(dataset) & x.obs['orig.ident'].isin(samples)].copy()
        if part.n_obs == 0:
            continue
        screened, summary, markers = screen_dataset(part, dataset, a.output)
        screened.obs.to_csv(a.output / f'{dataset}_cell_screen.csv.gz', index_label='cell_id')
        summary.to_csv(a.output / f'{dataset}_cluster_counts.csv', index=False)
        markers.to_csv(a.output / f'{dataset}_cluster_marker_expression.csv', index=False)
        screened.write_h5ad(a.output / f'{dataset}_clustered_rawcounts.h5ad', compression='gzip')
        summaries.append(summary)
        print(f'{dataset}: {screened.n_obs:,} cells; clusters retained for marker inspection')
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(a.output / 'all_cluster_counts.csv', index=False)


if __name__ == '__main__':
    main()
