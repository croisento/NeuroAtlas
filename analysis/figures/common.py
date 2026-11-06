"""Figure inputs, portable paths and metadata-only UMAP access."""
import argparse
from pathlib import Path
import h5py
import numpy as np
import anndata as ad
from scipy import sparse
from anndata.io import read_elem, sparse_dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def arguments(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    return p


def metadata_object(path, basis='X_umap'):
    with h5py.File(path, 'r') as f:
        obs = read_elem(f['obs'])
        x = ad.AnnData(sparse.csr_matrix((len(obs), 0)), obs=obs)
        x.obsm[basis] = f['obsm'][basis][:]
    return x


def qc_object(path):
    """Original 08: stored gene/mt QC; raw-count total and RPL/RPS percentages."""
    x = metadata_object(path)
    with h5py.File(path, 'r') as f:
        var = read_elem(f['var'])
        names = var.index.astype(str).str.upper()
        ribo = np.asarray(names.str.startswith('RPS') | names.str.startswith('RPL'))
        counts = sparse_dataset(f['layers/counts'])
        total = np.zeros(x.n_obs)
        fraction = np.zeros(x.n_obs)
        for first in range(0, x.n_obs, 5000):
            last = min(first + 5000, x.n_obs)
            block = counts[first:last]
            t = np.asarray(block.sum(axis=1)).ravel()
            r = np.asarray(block[:, ribo].sum(axis=1)).ravel()
            total[first:last] = t
            fraction[first:last] = np.divide(100*r, t, out=np.zeros_like(t, dtype=float), where=t>0)
    x.obs['total_counts'] = total
    x.obs['pct_counts_ribo'] = fraction
    return x


def categorical_umap(x, column, stem, colors=None):
    import scanpy as sc
    sc.pl.umap(x, color=column, palette=colors, size=2, frameon=False,
               use_raw=False, show=False)
    fig = plt.gcf()
    fig.savefig(str(stem)+'.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(str(stem)+'.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,
                         'ps.fonttype':42,'svg.fonttype':'none'})
