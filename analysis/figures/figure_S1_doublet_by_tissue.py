#!/usr/bin/env python3
"""Figure S1c: doublet score faceted by tissue on the existing UMAP."""
import json
import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

# Paths are supplied explicitly by the CLI.
SOURCE = OUT = TABLES = ATLAS = None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    if ATLAS is not None:
        obs, xy = ATLAS.obs, ATLAS.obsm['X_umap']
    else:
        with h5py.File(SOURCE, 'r') as handle:
            obs = read_elem(handle['obs'])
            xy = handle['obsm/X_umap'][:]
    scores = obs.doublet_score.to_numpy(dtype=float)
    positive = obs.doublet_call.astype(str).eq('Predicted_doublet').to_numpy()
    tissue = obs.tissue.astype(str).to_numpy()
    assert len(obs) == 573569 and positive.sum() == 6157
    assert np.isfinite(scores).all() and np.isfinite(xy).all()
    assert set(tissue) == {'CSF', 'PBMC'}
    assert set(obs.doublet_call.astype(str)) == {'Predicted_doublet', 'Predicted_singlet'}
    assert np.array_equal(positive, obs.predicted_doublet.to_numpy(dtype=bool))
    font = Path('/home/lishaogang/.fonts/arial.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    lower, upper = xy.min(axis=0), xy.max(axis=0)
    padding = (upper-lower)*.04
    report = []
    for name in ['CSF', 'PBMC']:
        mask = tissue == name
        n = int(mask.sum()); d = int(positive[mask].sum())
        report.append({'tissue': name, 'n_cells': n, 'n_predicted_doublets': d,
                       'n_predicted_singlets': n-d, 'predicted_doublet_percent': d/n*100})
    for mode in ['doublet_score']:
        fig = plt.figure(figsize=(13, 6.6))
        grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, .035], wspace=.15,
                               left=.05, right=.94, bottom=.18, top=.86)
        axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
        for ax, name, row in zip(axes, ['CSF', 'PBMC'], report):
            mask = tissue == name
            if mode == 'doublet_score':
                positions = np.flatnonzero(mask)
                positions = positions[np.argsort(scores[positions], kind='stable')]
                points = ax.scatter(xy[positions,0], xy[positions,1], c=scores[positions],
                                    cmap='viridis', vmin=0, vmax=1, s=.5, linewidths=0, rasterized=True)
                title = f"{name} (n={row['n_cells']:,})"
            else:
                for is_doublet, color, size, alpha in [(False,'#c9ced2',.4,.7),(True,'#d73027',1.2,.9)]:
                    selected = mask & (positive == is_doublet)
                    ax.scatter(xy[selected,0], xy[selected,1], c=color, s=size, alpha=alpha,
                               linewidths=0, rasterized=True)
                title = f"{name} (n={row['n_cells']:,})\nPredicted doublets: {row['n_predicted_doublets']:,} ({row['predicted_doublet_percent']:.2f}%)"
            ax.set(xlabel='UMAP1', ylabel='UMAP2', xticks=[], yticks=[], title=title,
                   xlim=(lower[0]-padding[0], upper[0]+padding[0]),
                   ylim=(lower[1]-padding[1], upper[1]+padding[1]))
            ax.set_aspect('equal', adjustable='box')
        if mode == 'doublet_score':
            cax = fig.add_subplot(grid[0,2]); cax.set_label('<colorbar>')
            fig.colorbar(points, cax=cax, label='Scrublet doublet score')
            fig.suptitle('Doublet score by tissue', y=.97)
        else:
            handles = [Line2D([],[],marker='o',linestyle='none',color=c,markersize=6,label=l)
                       for c,l in [('#c9ced2','Predicted singlet'),('#d73027','Predicted doublet')]]
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.46,.07),ncol=2,frameon=False)
            fig.suptitle('Predicted doublet labels by tissue', y=.97)
        fig.text(.05,.03,'All cells retained; original global UMAP coordinates and identical axis limits. Scrublet was fitted separately per sample.',fontsize=8)
        stem = f'UMAP_{mode}_CSF_PBMC_side_by_side_doublets_retained'
        for ext in ['pdf']:
            fig.savefig(OUT/f'{stem}.{ext}',dpi=400)
        plt.close(fig)


def cli():
    global SOURCE, OUT, TABLES
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    SOURCE, OUT = args.input, args.output
    TABLES = OUT / 'Tables'
    main()


if __name__ == '__main__':
    cli()
