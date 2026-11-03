#!/usr/bin/env python3
"""Remove annotated samples with uniform-QC cells_final <400, except CAR-T."""

from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import pandas as pd


PROJECT_DIR = Path("/home/lishaogang/NEU2026_July")
OUTPUT_DIR = PROJECT_DIR / "min400_sample_filtered"
UNION_CSV = (
    PROJECT_DIR
    / "uniform_QC_merged/audit/adata_final_corrected_all_GSM_QC_audit_union.csv"
)
MASTER_INPUT = PROJECT_DIR / "adata_final_Disease_corrected.h5ad"
CSF_INPUT = PROJECT_DIR / "adata_final_csf_Disease_corrected.h5ad"
MASTER_OUTPUT = OUTPUT_DIR / "adata_final_Disease_corrected_min400_except_CART.h5ad"
CSF_OUTPUT = OUTPUT_DIR / "adata_final_csf_Disease_corrected_min400_except_CART.h5ad"
REMOVAL_AUDIT = OUTPUT_DIR / "sample_filter_min400_except_CART_audit.csv"

MIN_CELLS = 400
CART_DATASET = "GSE186802"
EXPECTED_REMOVED_SAMPLES = {
    "GSM3984210",
    "GSM5827387",
    "GSM6016358",
    "GSM6016365",
    "GSM6016367",
    "GSM6016370",
    "GSM6016371",
    "GSM6016374",
}
CART_SAMPLES = {"DIPG1-Day7", "DIPG2-Day8", "DIPG3-Day7"}


def build_filter_audit() -> tuple[pd.DataFrame, set[str]]:
    audit = pd.read_csv(UNION_CSV)
    audit["filter_rule_below_400"] = audit["cells_final"] < MIN_CELLS
    audit["CAR_T_exempt"] = audit["dataset"].eq(CART_DATASET)
    audit["remove_from_annotated_object"] = (
        audit["present_in_adata_final"]
        & audit["filter_rule_below_400"]
        & ~audit["CAR_T_exempt"]
    )
    audit["filter_decision"] = "Keep"
    audit.loc[audit["CAR_T_exempt"] & audit["filter_rule_below_400"], "filter_decision"] = (
        "Keep: CAR-T exception"
    )
    audit.loc[audit["remove_from_annotated_object"], "filter_decision"] = (
        "Remove: cells_final < 400"
    )
    audit.loc[
        ~audit["present_in_adata_final"] & audit["filter_rule_below_400"],
        "filter_decision",
    ] = "Already absent from adata_final"

    removed = set(audit.loc[audit["remove_from_annotated_object"], "GSM"])
    if removed != EXPECTED_REMOVED_SAMPLES:
        raise ValueError(
            f"Unexpected removal set. Expected {sorted(EXPECTED_REMOVED_SAMPLES)}, "
            f"observed {sorted(removed)}"
        )
    cart_rows = audit[audit["GSM"].isin(CART_SAMPLES)]
    if set(cart_rows["GSM"]) != CART_SAMPLES or not cart_rows["CAR_T_exempt"].all():
        raise ValueError("CAR-T exception validation failed")
    return audit, removed


def subset_backed(input_path: Path, output_path: Path, removed: set[str]) -> None:
    temp_path = output_path.with_name(f".{output_path.name}.writing.tmp.h5ad")
    if temp_path.exists():
        temp_path.unlink()

    source = ad.read_h5ad(input_path, backed="r")
    try:
        sample_values = source.obs["orig.ident"].astype(str)
        keep = ~sample_values.isin(removed)
        removed_cells = int((~keep).sum())
        expected_cells = int(source.n_obs - removed_cells)
        print(
            f"Writing {output_path.name}: {source.n_obs:,} -> {expected_cells:,} cells",
            flush=True,
        )
        source[keep.to_numpy(), :].copy(filename=temp_path)
    finally:
        source.file.close()

    check = ad.read_h5ad(temp_path, backed="r")
    try:
        remaining_samples = set(check.obs["orig.ident"].astype(str))
        if check.n_obs != expected_cells:
            raise ValueError(f"Cell-count validation failed for {output_path.name}")
        if remaining_samples & removed:
            raise ValueError(f"Removed samples remain in {output_path.name}")
        if not CART_SAMPLES <= remaining_samples:
            raise ValueError(f"CAR-T samples missing from {output_path.name}")
    finally:
        check.file.close()

    os.replace(temp_path, output_path)
    print(f"Validated: {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit, removed = build_filter_audit()
    audit.to_csv(REMOVAL_AUDIT, index=False)
    subset_backed(MASTER_INPUT, MASTER_OUTPUT, removed)
    subset_backed(CSF_INPUT, CSF_OUTPUT, removed)
    print(f"Removal audit: {REMOVAL_AUDIT}")


if __name__ == "__main__":
    main()
