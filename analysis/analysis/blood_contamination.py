"""Marker panels and expression metrics from original script 41; no decision rules."""
import numpy as np
import pandas as pd
from scipy import sparse

SEED = 97

HB = ["HBB", "HBA1", "HBA2", "HBD", "HBG1", "HBG2"]

ERY = ["ALAS2", "AHSP", "GYPA", "SLC4A1", "EPB42"]

PLT_CORE = ["PF4", "PPBP", "GP9", "ITGA2B", "TUBB1"]

PLT = PLT_CORE + ["RGS18", "PF4V1", "GP1BA", "GP1BB"]

IMMUNE = ["PTPRC", "CD3D", "CD3E", "CD79A", "MS4A1", "LYZ", "LST1", "TYROBP", "NKG7", "CD74", "LILRA4"]

MARKERS = list(dict.fromkeys(HB + ERY + PLT + IMMUNE))

def sum_axis(x, axis=1):
    return np.asarray(x.sum(axis=axis)).ravel()

def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)

def available(names, genes):
    return [g for g in genes if g in names]

def molecular_metrics(a):
    raw = a.X
    total = sum_axis(raw)
    safe = np.maximum(total, 1)
    values = {}
    for prefix, genes in [("hb", HB), ("ery", ERY), ("platelet", PLT), ("immune", IMMUNE)]:
        present = available(a.var_names, genes)
        matrix = dense(raw[:, a.var_names.get_indexer(present)]) if present else np.zeros((a.n_obs, 0))
        values[f"pct_counts_{prefix}"] = matrix.sum(axis=1) / safe * 100
        values[f"n_{prefix}_genes_detected"] = (matrix > 0).sum(axis=1)
    core = available(a.var_names, PLT_CORE)
    m = dense(raw[:, a.var_names.get_indexer(core)]) if core else np.zeros((a.n_obs, 0))
    values["n_platelet_core_genes_detected"] = (m > 0).sum(axis=1)
    # Multiple immune genes reduce the influence of a single ambient transcript.
    values["immune_signature_detected"] = values["n_immune_genes_detected"] >= 2
    return pd.DataFrame(values, index=a.obs_names)

