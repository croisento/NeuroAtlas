#!/usr/bin/env python3
"""Visualize all Scrublet-annotated cells without filtering or recomputing UMAP."""
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

PROJECT = Path('/home/lishaogang/NEU2026_July')
ROOT = PROJECT / 'final/Doublet_identification_Scrublet'
SOURCE = ROOT / 'adata_final_Disease_corrected_min400_except_CART_scrublet_identified.h5ad'
OUT = ROOT / 'Figures/Doublet_overview'
TABLES = ROOT / 'Tables/Doublet_overview'
sys.path.insert(0, '/home/lishaogang/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment


def export(fig, stem):
    require_matplotlib_panel_alignment(fig, json_out=str(OUT / f'{stem}.alignment.json'), strict=True)
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'{stem}.{extension}', dpi=400)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    with h5py.File(SOURCE, 'r') as handle:
        obs = read_elem(handle['obs'])
        xy = np.asarray(handle['obsm/X_umap'][:])
    assert len(obs) == 573569 and xy.shape == (len(obs), 2)
    scores = obs['doublet_score'].to_numpy(dtype=float)
    calls = obs['doublet_call'].astype(str)
    assert np.isfinite(scores).all() and np.isfinite(xy).all()
    assert set(calls) == {'Predicted_singlet', 'Predicted_doublet'}
    positive = calls.eq('Predicted_doublet').to_numpy()
    assert positive.sum() == 6157
    font = Path('/home/lishaogang/.fonts/arial.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(13, 6.3))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.035], wspace=0.15,
                          left=0.05, right=0.94, bottom=0.18, top=0.88)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    cax = fig.add_subplot(grid[0, 2])
    cax.set_label('<colorbar>')
    axes[0].scatter(xy[~positive, 0], xy[~positive, 1], s=0.3, c='#c9ced2',
                    alpha=0.7, linewidths=0, rasterized=True)
    axes[0].scatter(xy[positive, 0], xy[positive, 1], s=1.1, c='#d73027',
                    alpha=0.9, linewidths=0, rasterized=True)
    positions = np.argsort(scores, kind='stable')
    points = axes[1].scatter(xy[positions, 0], xy[positions, 1], c=scores[positions],
                             cmap='viridis', vmin=0, vmax=1, s=0.45, linewidths=0, rasterized=True)
    fig.colorbar(points, cax=cax, label='Scrublet doublet score')
    for ax, title in zip(axes, ['a  Predicted doublet labels', 'b  Doublet score']):
        ax.set(xlabel='UMAP1', ylabel='UMAP2', xticks=[], yticks=[], title=title)
        ax.set_xlim(float(xy[:, 0].min()) - 1, float(xy[:, 0].max()) + 1)
        ax.set_ylim(float(xy[:, 1].min()) - 1, float(xy[:, 1].max()) + 1)
        ax.set_aspect('equal', adjustable='box')
    handles = [Line2D([], [], marker='o', linestyle='none', color=color, markersize=6, label=label)
               for color, label in [('#c9ced2', f'Predicted singlet (n={(~positive).sum():,})'),
                                    ('#d73027', f'Predicted doublet (n={positive.sum():,})')]]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.45, 0.065), ncol=2, frameon=False)
    fig.text(0.05, 0.03, 'All 573,569 cells retained; existing UMAP coordinates. Scores and thresholds were estimated separately per sample.', fontsize=8)
    export(fig, 'UMAP_predicted_labels_and_doublet_score_doublets_retained')

    # Recompute fractions directly from cell-level calls and verify the historical summary.
    frame = obs[['dataset', 'orig.ident']].astype(str).copy()
    frame['predicted_doublet'] = positive.astype(int)
    samples = frame.groupby(['dataset', 'orig.ident'], observed=True).agg(
        n_cells=('predicted_doublet', 'size'), n_predicted_doublet=('predicted_doublet', 'sum')).reset_index()
    historical = pd.read_csv(ROOT / 'sample_doublet_summary.csv')
    samples = samples.merge(historical[['dataset', 'sample', 'threshold', 'predicted_doublet_fraction']],
                            left_on=['dataset', 'orig.ident'], right_on=['dataset', 'sample'], validate='one_to_one')
    fraction = samples['n_predicted_doublet'] / samples['n_cells']
    assert len(samples) == 205 and np.allclose(fraction, samples['predicted_doublet_fraction'])
    samples['predicted_doublet_percent'] = fraction * 100
    palette_source = PROJECT / 'final/h5ad/adata_final_Disease_corrected_min400_except_CART.h5ad'
    with h5py.File(palette_source, 'r') as handle:
        order = list(read_elem(handle['obs/dataset']).categories)
        colors = list(read_elem(handle['uns/dataset_colors']))
    palette = dict(zip(order, colors))
    order = [dataset for dataset in order if dataset in set(samples['dataset'])]
    samples['dataset_order'] = samples['dataset'].map({dataset: i for i, dataset in enumerate(order)})
    samples = samples.sort_values(['dataset_order', 'predicted_doublet_percent', 'orig.ident']).reset_index(drop=True)
    samples['plot_position'] = np.arange(len(samples)) + 1
    samples['dataset_color'] = samples['dataset'].map(palette)
    assert samples['dataset_color'].notna().all()
    samples.to_csv(TABLES / 'sample_predicted_doublet_fractions_plot_data.csv', index=False)
    fig, ax = plt.subplots(figsize=(14, 6.2))
    centers = []
    for dataset in order:
        group = samples[samples['dataset'].eq(dataset)]
        ax.scatter(group['plot_position'], group['predicted_doublet_percent'], c=palette[dataset],
                   s=22, edgecolors='#444444', linewidths=0.3, zorder=3)
        centers.append(group['plot_position'].mean())
        ax.axvline(group['plot_position'].max() + 0.5, color='#dddddd', lw=0.5, zorder=0)
    ax.axhline(5, color='#777777', linestyle='--', linewidth=0.9)
    high = samples.loc[samples['predicted_doublet_percent'].idxmax()]
    ax.annotate(f"{high['orig.ident']}\n{high['predicted_doublet_percent']:.2f}%", xy=(high['plot_position'], high['predicted_doublet_percent']),
                xytext=(high['plot_position'] + 8, high['predicted_doublet_percent'] - 10),
                arrowprops={'arrowstyle': '-', 'lw': 0.7}, fontsize=8)
    ax.set(xlim=(0, len(samples) + 2), ylim=(0, 100), xlabel='Samples grouped by source dataset',
           ylabel='Predicted doublets per sample (%)', title='Sample-level predicted doublet fractions | all cells retained')
    ax.set_xticks(centers, order, rotation=40, ha='right')
    for index, label in enumerate(ax.get_xticklabels()):
        label.set_y(-0.055 if index % 2 else 0)
    ax.xaxis.labelpad = 22
    ax.grid(axis='y', color='#eeeeee', linewidth=0.6)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.90, bottom=0.30)
    fig.text(0.075, 0.065, 'Each point represents one sample (n=205). Dashed line: expected doublet rate supplied to Scrublet (5%), not an exclusion cutoff.', fontsize=8)
    fig.text(0.075, 0.03, 'Samples are ordered by predicted fraction within each dataset. GSM4104133 requires threshold review; predictions are not validated doublets.', fontsize=8)
    export(fig, 'Sample_predicted_doublet_fraction_by_dataset_doublets_retained')
    report = {'cells': len(obs), 'samples': len(samples), 'predicted_doublets': int(positive.sum()),
              'predicted_doublet_percent_overall': float(positive.mean() * 100),
              'cells_removed': 0, 'UMAP_recomputed': False, 'score_range': [float(scores.min()), float(scores.max())],
              'historical_sample_fractions_verified': True, 'source_modified': False,
              'highest_fraction_sample': str(high['orig.ident']),
              'collision_audit': 'Requires PyMuPDF; run separately when available'}
    (TABLES / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
