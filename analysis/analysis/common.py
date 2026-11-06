"""Paths and small helpers shared by the analysis scripts."""
from pathlib import Path
import os
import argparse
import h5py
import numpy as np
from anndata.io import read_elem

REPOSITORY = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('NEUCSF_DATA_ROOT', REPOSITORY/'inputs'))
OUTPUT_ROOT = Path(os.environ.get('NEUCSF_OUTPUT_ROOT', REPOSITORY/'results'))
MASTER_NAME = 'adata_final_Disease_corrected_min400_except_CART_scrublet_identified.h5ad'
MASTER = DATA_ROOT/'h5ad'/MASTER_NAME
LABEL = 'controls without CNS inflammation (unspecific)'


def arguments(description, default_input=MASTER):
    parser=argparse.ArgumentParser(description=description)
    parser.add_argument('--input',type=Path,default=default_input)
    parser.add_argument('--output',type=Path,required=True)
    return parser


def obs_umap(path):
    with h5py.File(path,'r') as f:
        return read_elem(f['obs']),np.asarray(f['obsm/X_umap'][:])


def refuse_overwrite(path):
    path=Path(path)
    if path.exists():raise FileExistsError(f'Output already exists: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)


def save_figure(fig, stem):
    import matplotlib.pyplot as plt
    stem=Path(stem);stem.parent.mkdir(parents=True,exist_ok=True)
    for extension in ('pdf','svg','png'):
        fig.savefig(stem.with_suffix('.'+extension),dpi=400,bbox_inches='tight')
    plt.close(fig)


def setup_style():
    import matplotlib as mpl
    mpl.use('Agg')
    mpl.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
