#!/usr/bin/env python3
# =============================================================================
# Optional QC flags: per-sample Scrublet and sample-composition PCA; no deletion.
# =============================================================================
import os
for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[name] = "2"

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import gc
import json
import shutil
import sys
import traceback
import warnings
from importlib.metadata import version
import h5py
import numpy as np
import pandas as pd
import anndata as ad
from anndata.io import read_elem, write_elem, sparse_dataset
import scanpy as sc
from threadpoolctl import threadpool_limits

from analysis.common import MASTER, OUTPUT_ROOT
SOURCE = MASTER
OUT = OUTPUT_ROOT / "Scrublet"
TARGET = OUT / "atlas_scrublet_identified.h5ad"
COUNTS = None
OBS = None
VAR = None


def analyze_sample(job):
    number, dataset, sample, positions = job
    path = OUT / "per_sample" / f"sample_{number:03d}.h5ad"
    expected = OBS.index[positions]
    if path.exists():
        cached = ad.read_h5ad(path)
        assert cached.obs_names.equals(expected)
        if cached.uns["run_status"] == "complete":
            return number, "cached", len(positions)
    with threadpool_limits(limits=2):
        captured = []
        attempts = []
        result = None
        last_error = ""
        for pcs in [30, 20, 10]:
            a = ad.AnnData(X=COUNTS[positions].copy(), obs=pd.DataFrame(index=expected), var=VAR.copy())
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    sc.pp.scrublet(a, expected_doublet_rate=0.05, stdev_doublet_rate=0.02,
                                   sim_doublet_ratio=2.0, n_prin_comps=pcs,
                                   use_approx_neighbors=False, threshold=None,
                                   random_state=0, verbose=False)
                captured.extend(str(w.message) for w in caught)
                result = a
                attempts.append({"PCs": pcs, "status": "complete"})
                break
            except Exception as exc:
                last_error = traceback.format_exc()
                attempts.append({"PCs": pcs, "status": "failed", "error": str(exc)})
                # A lower-rank retry addresses small-sample/feature rank only.
                if not any(term in str(exc).lower() for term in ["n_components", "n_prin_comps", "min(n_samples", "arpack"]):
                    break
                del a
                gc.collect()
        scored = pd.DataFrame(index=expected)
        threshold = np.nan
        status = "failed"
        if result is not None:
            scored["doublet_score"] = pd.to_numeric(result.obs["doublet_score"], errors="coerce")
            scrub = result.uns["scrublet"]
            threshold = float(scrub.get("threshold", np.nan))
            predictions = result.obs.get("predicted_doublet", pd.Series(pd.NA, index=expected))
            scored["predicted_doublet"] = pd.array(predictions, dtype="boolean")
            finite = np.isfinite(scored["doublet_score"])
            status = "complete" if finite.all() and scored["predicted_doublet"].notna().all() else "unresolved"
            metadata = {"scrublet": scrub}
        else:
            scored["doublet_score"] = np.nan
            scored["predicted_doublet"] = pd.array([pd.NA] * len(expected), dtype="boolean")
            metadata = {}
        scored["doublet_threshold"] = threshold
        scored["scrublet_status"] = status
        checkpoint = ad.AnnData(obs=scored, uns={**metadata, "run_status": status,
            "dataset": str(dataset), "sample": str(sample), "warnings": np.asarray(captured, dtype=str),
            "attempts_json": json.dumps(attempts), "error": last_error if status == "failed" else ""})
        temp = path.with_name("." + path.name + ".tmp")
        checkpoint.write_h5ad(temp)
        os.replace(temp, path)
        return number, status, len(positions)


def main():
    global COUNTS, OBS, VAR
    (OUT / "per_sample").mkdir(parents=True, exist_ok=True)
    stat = SOURCE.stat()
    identity = {"source": str(SOURCE), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    identity_file = OUT / "input_identity.json"
    if identity_file.exists():
        assert json.loads(identity_file.read_text()) == identity, "Input changed; do not reuse old sample checkpoints"
    else:
        identity_file.write_text(json.dumps(identity, indent=2) + "\n")
    with h5py.File(SOURCE) as f:
        OBS = read_elem(f["obs"])
        VAR = read_elem(f["var"])
        print(f"Loading layers/counts: {len(OBS):,} cells, {len(VAR):,} genes", flush=True)
        COUNTS = sparse_dataset(f["layers/counts"]).to_memory()
    assert COUNTS.shape == (len(OBS), len(VAR))
    assert np.isfinite(COUNTS.data).all() and (COUNTS.data >= 0).all()
    assert np.allclose(COUNTS.data, np.rint(COUNTS.data))
    assert COUNTS.data.max() < 2 ** 24
    COUNTS = COUNTS.astype(np.float32)
    jobs = []
    for number, ((dataset, sample), positions) in enumerate(OBS.groupby(["dataset", "orig.ident"], observed=True, sort=True).indices.items()):
        jobs.append((number, str(dataset), str(sample), np.sort(positions)))
    pd.DataFrame([{"number": n, "dataset": d, "sample": s, "n_cells": len(p)} for n, d, s, p in jobs]).to_csv(OUT / "sample_manifest.csv", index=False)
    print(f"Scrublet: {len(jobs)} samples; 4 workers; 2 threads per worker; no cell removal", flush=True)
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("fork")) as pool:
        pending = {pool.submit(analyze_sample, job): job for job in jobs}
        for count, future in enumerate(as_completed(pending), start=1):
            job = pending[future]
            number, status, n_cells = future.result()
            print(f"[{count}/{len(jobs)}] {job[1]} / {job[2]}: {n_cells:,} cells; {status}", flush=True)
    del COUNTS
    gc.collect()
    all_results, sample_rows, batch_metadata = [], [], {}
    for number, dataset, sample, positions in jobs:
        a = ad.read_h5ad(OUT / "per_sample" / f"sample_{number:03d}.h5ad")
        assert a.obs_names.equals(OBS.index[positions])
        table = a.obs.copy()
        table["doublet_batch"] = f"{dataset}|{sample}"
        all_results.append(table)
        pred = table["predicted_doublet"]
        sample_rows.append({"dataset": dataset, "sample": sample, "n_cells": len(table),
            "n_scored": int(table["doublet_score"].notna().sum()), "n_called": int(pred.notna().sum()),
            "n_predicted_doublet": int(pred.sum()), "predicted_doublet_fraction": float(pred.mean()) if pred.notna().any() else np.nan,
            "threshold": float(table["doublet_threshold"].iloc[0]), "status": str(a.uns["run_status"]),
            "attempts": a.uns["attempts_json"], "warnings": "; ".join(a.uns["warnings"].tolist()), "error": a.uns["error"]})
        batch_metadata[f"sample_{number:03d}"] = dict(a.uns)
    scores = pd.concat(all_results).reindex(OBS.index)
    assert scores.index.is_unique and scores.index.equals(OBS.index)
    scores["doublet_call"] = pd.Categorical(np.where(scores["predicted_doublet"].isna(), "Unresolved",
        np.where(scores["predicted_doublet"].fillna(False).to_numpy(dtype=bool), "Predicted_doublet", "Predicted_singlet")))
    enriched = OBS.copy()
    for column in scores:
        enriched[column] = scores[column]
    enriched.to_csv(OUT / "cell_doublet_scores_and_metadata.csv.gz", compression="gzip", index_label="cell_id")
    sample_summary = pd.DataFrame(sample_rows)
    sample_summary.to_csv(OUT / "sample_doublet_summary.csv", index=False)
    for column in ["dataset", "tissue", "cell_type", "cell_subtype_short"]:
        enriched.groupby(column, observed=True).agg(n_cells=("doublet_call", "size"),
            n_scored=("doublet_score", "count"), n_called=("predicted_doublet", "count"),
            n_predicted_doublet=("predicted_doublet", "sum"), predicted_doublet_fraction=("predicted_doublet", "mean")).to_csv(OUT / f"doublet_summary_by_{column}.csv")
    parameters = {"method": "scanpy.pp.scrublet", "counts_layer": "counts", "batch_columns": ["dataset", "orig.ident"],
        "expected_doublet_rate": 0.05, "stdev_doublet_rate": 0.02, "sim_doublet_ratio": 2.0, "random_state": 0,
        "default_PCs": 30, "neighbor_search": "exact", "threshold": "automatic from simulated-score distribution",
        "cell_removal": False, "expression_or_embedding_recomputed": False,
        "limitations": "Post-hoc on retained final-atlas cells and 12,890 retained genes. Sample identifiers may not resolve individual capture libraries. Expected rate is a prior, not a measured rate. Scores do not prove physical doublets."}
    versions = {p: version(p) for p in ["scanpy", "anndata", "numpy", "scipy", "scikit-learn", "scikit-image"]}
    temp = TARGET.with_name("." + TARGET.name + ".writing")
    print("Copying complete original h5ad and adding annotations; original remains unchanged", flush=True)
    shutil.copy2(SOURCE, temp)
    with h5py.File(temp, "r+") as f:
        write_elem(f, "obs", enriched)
        write_elem(f["uns"], "scrublet_identification", {"parameters": parameters, "versions": versions, "batches": batch_metadata})
    with h5py.File(SOURCE) as original, h5py.File(temp) as new:
        assert read_elem(new["obs"]).index.equals(read_elem(original["obs"]).index)
        assert read_elem(new["var"]).equals(read_elem(original["var"]))
        assert set(new["obsm"]) == set(original["obsm"])
        for key in original["obsm"]:
            assert np.array_equal(read_elem(original["obsm"][key]), read_elem(new["obsm"][key]))
        for key in ["X", "layers/counts", "layers/logcounts", "raw/X"]:
            assert new[key].attrs["shape"].tolist() == original[key].attrs["shape"].tolist()
            assert new[key]["data"].shape == original[key]["data"].shape
    assert SOURCE.stat().st_size == identity["size"] and SOURCE.stat().st_mtime_ns == identity["mtime_ns"]
    os.replace(temp, TARGET)
    report = {"input_cells": len(OBS), "output_cells": len(enriched), "genes": len(VAR), "samples": len(jobs),
              "cells_removed": 0, "n_scored": int(scores["doublet_score"].notna().sum()),
              "n_called": int(scores["predicted_doublet"].notna().sum()),
              "n_predicted_doublet": int(scores["predicted_doublet"].sum()),
              "samples_not_complete": int(sample_summary["status"].ne("complete").sum()),
              "original_unchanged": True, "original_embeddings_unchanged": True,
              "output": str(TARGET), "parameters": parameters, "versions": versions}
    (OUT / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)



from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from analysis.common import REPOSITORY, refuse_overwrite

def calculate(proportions,metadata):
    matrix=proportions.pivot(index=['dataset','orig.ident','tissue'],columns='cell_subtype_short',values='Proportion_percent').fillna(0)
    info=metadata.set_index(['GSE','GSM','tissue'])
    assert matrix.index.is_unique and matrix.index.isin(info.index).all()
    info=info.loc[matrix.index].copy()
    info['PC1']=np.nan;info['PC2']=np.nan;info['PCA_distance_z']=np.nan
    info['PCA_compositional_outlier']='Not assessed'
    for (tissue,disease),group in info.reset_index().groupby(['tissue','Disease'],observed=True):
        if len(group)<5:continue
        ids=pd.MultiIndex.from_frame(group[['dataset','orig.ident','tissue']]) if 'dataset' in group else pd.MultiIndex.from_frame(group[['GSE','GSM','tissue']])
        coords=PCA(n_components=2).fit_transform(StandardScaler().fit_transform(matrix.loc[ids]))
        distance=np.sqrt((coords**2).sum(axis=1));z=zscore(distance,ddof=0)
        z=np.nan_to_num(z,nan=0.)
        info.loc[ids,'PC1']=coords[:,0];info.loc[ids,'PC2']=coords[:,1]
        info.loc[ids,'PCA_distance_z']=z
        info.loc[ids,'PCA_compositional_outlier']=np.where(z>1.5,'Flagged','Not flagged')
    return info.reset_index()


def cli():
    global SOURCE, OUT, TARGET
    from analysis.common import arguments
    parser = arguments(__doc__)
    parser.add_argument('--proportions', type=Path,
                        default=REPOSITORY / 'analysis/data/sample_subtype_proportions.csv')
    parser.add_argument('--metadata', type=Path,
                        default=REPOSITORY / 'data/sample_metadata.csv')
    args = parser.parse_args()
    SOURCE = args.input
    OUT = args.output
    TARGET = OUT / "atlas_scrublet_identified.h5ad"
    if TARGET.exists():
        raise FileExistsError(TARGET)
    pca_output = OUT / 'PCA_compositional_outlier_flags.csv'
    refuse_overwrite(pca_output)
    flags = calculate(pd.read_csv(args.proportions), pd.read_csv(args.metadata))
    # Require the supplied proportion table to cover this exact atlas sample set.
    with h5py.File(SOURCE, 'r') as f:
        source_obs = read_elem(f['obs'])
    source_keys = set(zip(source_obs.dataset.astype(str), source_obs['orig.ident'].astype(str), source_obs.tissue.astype(str)))
    flag_keys = set(zip(flags.GSE.astype(str), flags.GSM.astype(str), flags.tissue.astype(str)))
    if source_keys != flag_keys:
        raise ValueError('PCA proportions/metadata must cover the same dataset, sample and tissue keys as the input h5ad')
    main()
    flags.to_csv(pca_output, index=False)
    print(flags.PCA_compositional_outlier.value_counts())


if __name__ == "__main__":
    cli()
