#!/usr/bin/env python3
"""Correct two-study metadata in the filtered master and lineage objects.

The supplied table describes the complete raw studies. Rows that are absent from
the current filtered master are recorded in the audit but are never added back.
Expression matrices and all existing dimensional-reduction coordinates are kept.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd


PROJECT_DIR = Path("/home/lishaogang/NEU2026_July")
FILTER_DIR = PROJECT_DIR / "min400_sample_filtered"
LINEAGE_DIR = FILTER_DIR / "lineage_objects"
AUDIT_DIR = FILTER_DIR / "audit"

MASTER = FILTER_DIR / "adata_final_Disease_corrected_min400_except_CART.h5ad"
CSF = FILTER_DIR / "adata_final_csf_Disease_corrected_min400_except_CART.h5ad"
PBMC = FILTER_DIR / "adata_final_pbmc_Disease_corrected_min400_except_CART.h5ad"
LINEAGES = {
    "B": LINEAGE_DIR / "adata_final_B_Disease_corrected_min400_except_CART.h5ad",
    "ILCS": LINEAGE_DIR / "adata_final_ILCS_Disease_corrected_min400_except_CART.h5ad",
    "Myeloid": LINEAGE_DIR
    / "adata_final_Myeloid_Disease_corrected_min400_except_CART.h5ad",
    "T": LINEAGE_DIR / "adata_final_T_Disease_corrected_min400_except_CART.h5ad",
}

CORRECTION_AUDIT = AUDIT_DIR / "GSE134577_GSE141578_metadata_correction_audit.csv"
OBJECT_AUDIT = AUDIT_DIR / "GSE134577_GSE141578_corrected_object_audit.csv"

# study, sample, sex, organ, detailed disease, disease group, age
RAW_METADATA_ROWS = [
    ("GSE134577", "GSM3984206", "F", "CSF", "AD", "ND", 83),
    ("GSE134577", "GSM3984214", "F", "CSF", "MCI", "ND", 83),
    ("GSE134577", "GSM3984212", "M", "CSF", "MCI", "ND", 78),
    ("GSE134577", "GSM3984200", "M", "CSF", "AD", "ND", 77),
    ("GSE134577", "GSM3984203", "M", "CSF", "MCI", "ND", 72),
    ("GSE134577", "GSM3984207", "M", "CSF", "MCI", "ND", 66),
    ("GSE134577", "GSM3984204", "F", "CSF", "AD", "ND", 64),
    ("GSE134577", "GSM3984213", "F", "CSF", "AD", "ND", 58),
    ("GSE134577", "GSM3984210", "M", "CSF", "MCI", "ND", 48),
    ("GSE134577", "GSM3984215", "M", "CSF", "HC", "HC", 66),
    ("GSE134577", "GSM3984211", "F", "CSF", "HC", "HC", 67),
    ("GSE134577", "GSM3984216", "M", "CSF", "HC", "HC", 69),
    ("GSE134577", "GSM3984201", "F", "CSF", "HC", "HC", 71),
    ("GSE134577", "GSM3984208", "F", "CSF", "HC", "HC", 73),
    ("GSE134577", "GSM3984202", "M", "CSF", "HC", "HC", 74),
    ("GSE134577", "GSM3984205", "M", "CSF", "HC", "HC", 74),
    ("GSE134577", "GSM3984209", "F", "CSF", "HC", "HC", 76),
    ("GSE134577", "GSM3984199", "M", "CSF", "HC", "HC", 88),
    ("GSE141578", "GSM4404054", "M", "CSF", "PD", "ND", 77),
    ("GSE141578", "GSM4208771", "M", "CSF", "PD", "ND", 73),
    ("GSE141578", "GSM4404052", "M", "CSF", "DLB", "ND", 71),
    ("GSE141578", "GSM4404053", "F", "CSF", "PD", "ND", 68),
    ("GSE141578", "GSM4208768", "F", "CSF", "PD", "ND", 67),
    ("GSE141578", "GSM4208766", "F", "CSF", "PD", "ND", 66),
    ("GSE141578", "GSM4208769", "M", "CSF", "PD", "ND", 66),
    ("GSE141578", "GSM4208770", "F", "CSF", "PD", "ND", 63),
    ("GSE141578", "GSM4404057", "F", "CSF", "PD", "ND", 63),
    ("GSE141578", "GSM4404058", "M", "CSF", "DLB", "ND", 58),
    ("GSE141578", "GSM4404056", "F", "CSF", "PD", "ND", 57),
    ("GSE141578", "GSM4208772", "M", "CSF", "HC", "HC", 51),
    ("GSE141578", "GSM4208780", "F", "CSF", "HC", "HC", 66),
    ("GSE141578", "GSM5579355", "M", "PBMC", "HC", "HC", 66),
    ("GSE141578", "GSM4208776", "F", "CSF", "HC", "HC", 69),
    ("GSE141578", "GSM4208779", "M", "CSF", "HC", "HC", 69),
    ("GSE141578", "GSM5579354", "M", "PBMC", "HC", "HC", 69),
    ("GSE141578", "GSM4208774", "M", "CSF", "HC", "HC", 70),
    ("GSE141578", "GSM4208775", "M", "CSF", "HC", "HC", 71),
    ("GSE141578", "GSM4208777", "F", "CSF", "HC", "HC", 73),
    ("GSE141578", "GSM4404055", "F", "CSF", "HC", "HC", 74),
    ("GSE141578", "GSM4208773", "F", "CSF", "HC", "HC", 75),
    ("GSE141578", "GSM5579357", "M", "PBMC", "HC", "HC", 75),
    ("GSE141578", "GSM4208778", "M", "CSF", "HC", "HC", 76),
    ("GSE141578", "GSM4404059", "F", "CSF", "HC", "HC", 88),
    ("GSE141578", "GSM5579365", "F", "PBMC", "HC", "HC", 88),
]

RAW_METADATA = pd.DataFrame(
    RAW_METADATA_ROWS,
    columns=["dataset", "GSM", "sex", "organ", "raw_disease", "disease_group", "age"],
).set_index("GSM", verify_integrity=True)

SEX_MAP = {"F": "Female", "M": "Male"}
DISEASE1_MAP = {
    "MCI/AD": "Mild Cognitive Impairment/Alzheimer Disease",
    "PD": "Parkinson Disease",
    "DLB": "Dementia with Lewy Bodies",
    "Older control": "Older Control (age >= 60 years)",
    "Younger control": "Younger Control (age < 60 years)",
}
UPDATED_COLUMNS = [
    "dataset",
    "disease",
    "Disease",
    "Disease1",
    "age",
    "age_numeric",
    "gender",
    "tissue",
]


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in np.asarray(values)
        ],
        dtype=object,
    )


def read_obs_column(obs: h5py.Group, column: str) -> np.ndarray:
    element = obs[column]
    if isinstance(element, h5py.Group):
        categories = decode(element["categories"])
        codes = np.asarray(element["codes"])
        result = np.full(len(codes), None, dtype=object)
        valid = codes >= 0
        result[valid] = categories[codes[valid]]
        return result
    values = np.asarray(element)
    return decode(values) if values.dtype.kind in "SO" else values.copy()


def write_categorical(obs: h5py.Group, column: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=object)
    if any(value is None for value in values):
        raise ValueError(f"Cannot write missing values to categorical column {column}")
    values = values.astype(str)
    categories = np.asarray(sorted(set(values)), dtype=object)
    lookup = {value: index for index, value in enumerate(categories)}
    codes = np.asarray([lookup[value] for value in values], dtype=np.int32)

    if column in obs:
        del obs[column]
    group = obs.create_group(column)
    group.attrs["encoding-type"] = "categorical"
    group.attrs["encoding-version"] = "0.2.0"
    group.attrs["ordered"] = False
    category_data = group.create_dataset(
        "categories", data=categories, dtype=h5py.string_dtype("utf-8")
    )
    category_data.attrs["encoding-type"] = "string-array"
    category_data.attrs["encoding-version"] = "0.2.0"
    code_data = group.create_dataset("codes", data=codes)
    code_data.attrs["encoding-type"] = "array"
    code_data.attrs["encoding-version"] = "0.2.0"


def write_numeric(obs: h5py.Group, column: str, values: np.ndarray) -> None:
    if column in obs:
        del obs[column]
    dataset = obs.create_dataset(column, data=np.asarray(values, dtype=np.float64))
    dataset.attrs["encoding-type"] = "array"
    dataset.attrs["encoding-version"] = "0.2.0"


def broad_disease(raw_disease: str, age: float) -> str:
    if raw_disease in {"AD", "MCI"}:
        return "MCI/AD"
    if raw_disease == "HC":
        return "Older control" if age >= 60 else "Younger control"
    return raw_disease


def dataset_digest(dataset: h5py.Dataset, block_rows: int = 50_000) -> str:
    digest = hashlib.sha256()
    if dataset.ndim == 0:
        digest.update(np.asarray(dataset).tobytes())
        return digest.hexdigest()
    for start in range(0, dataset.shape[0], block_rows):
        digest.update(np.asarray(dataset[start : start + block_rows]).tobytes())
    return digest.hexdigest()


def obsm_digests(path: Path) -> dict[str, str]:
    with h5py.File(path, "r") as handle:
        return {
            key: dataset_digest(handle["obsm"][key])
            for key in handle.get("obsm", {})
            if isinstance(handle["obsm"][key], h5py.Dataset)
        }


def sample_metadata(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        columns = [column for column in UPDATED_COLUMNS if column in obs]
        frame = pd.DataFrame({column: read_obs_column(obs, column) for column in columns})
        frame.insert(0, "GSM", read_obs_column(obs, "orig.ident").astype(str))
    return frame.drop_duplicates("GSM").set_index("GSM")


def corrected_values(record: pd.Series) -> dict[str, object]:
    disease = broad_disease(str(record.raw_disease), float(record.age))
    return {
        "dataset": str(record.dataset),
        "disease": str(record.raw_disease),
        "Disease": disease,
        "Disease1": DISEASE1_MAP[disease],
        "age": f"{float(record.age):g}",
        "age_numeric": float(record.age),
        "gender": SEX_MAP[str(record.sex)],
        "tissue": str(record.organ),
    }


def update_object(path: Path, object_name: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    before_shape = None
    before_obsm = obsm_digests(path)
    before_samples = sample_metadata(path)
    temp = path.with_name(f".{path.name}.metadata_correcting.tmp.h5ad")
    if temp.exists():
        temp.unlink()
    print(f"Copying {object_name} to validated temporary file ...", flush=True)
    shutil.copy2(path, temp)

    try:
        with h5py.File(temp, "r+") as handle:
            before_shape = tuple(int(value) for value in handle["X"].attrs["shape"])
            obs = handle["obs"]
            samples = read_obs_column(obs, "orig.ident").astype(str)
            values = {
                column: read_obs_column(obs, column)
                for column in UPDATED_COLUMNS
                if column in obs
            }
            if set(values) != set(UPDATED_COLUMNS):
                missing = sorted(set(UPDATED_COLUMNS) - set(values))
                raise KeyError(f"{object_name} is missing metadata columns: {missing}")

            for gsm, record in RAW_METADATA.iterrows():
                mask = samples == gsm
                if not mask.any():
                    continue
                expected = corrected_values(record)
                for column, value in expected.items():
                    values[column][mask] = value

            for column in UPDATED_COLUMNS:
                if column == "age_numeric":
                    write_numeric(obs, column, values[column])
                else:
                    write_categorical(obs, column, values[column])
            handle.flush()

        after_obsm = obsm_digests(temp)
        if after_obsm != before_obsm:
            raise ValueError(f"{object_name}: dimensional-reduction coordinates changed")
        after_samples = sample_metadata(temp)
        for gsm, record in RAW_METADATA.iterrows():
            if gsm not in after_samples.index:
                continue
            expected = corrected_values(record)
            actual = after_samples.loc[gsm]
            for column, value in expected.items():
                observed = actual[column]
                if column == "age_numeric":
                    if not np.isclose(float(observed), float(value)):
                        raise ValueError(f"{object_name}: {gsm} {column} validation failed")
                elif str(observed) != str(value):
                    raise ValueError(f"{object_name}: {gsm} {column} validation failed")

        os.replace(temp, path)
        present = RAW_METADATA.index.intersection(after_samples.index)
        changed_samples = 0
        for gsm in present:
            expected = corrected_values(RAW_METADATA.loc[gsm])
            if any(str(before_samples.loc[gsm, col]) != str(value) for col, value in expected.items()):
                changed_samples += 1
        print(
            f"Validated {object_name}: {len(present)} supplied samples present, "
            f"{changed_samples} samples changed; coordinates retained",
            flush=True,
        )
        return {
            "object": object_name,
            "path": str(path),
            "n_obs": before_shape[0],
            "n_vars": before_shape[1],
            "supplied_samples_present": len(present),
            "supplied_samples_absent": len(RAW_METADATA) - len(present),
            "samples_with_any_change": changed_samples,
            "coordinates_retained": True,
            "analysis_recomputed": False,
        }
    except Exception:
        if temp.exists():
            temp.unlink()
        raise


def build_master_correction_audit(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gsm, record in RAW_METADATA.iterrows():
        present = gsm in after.index
        row = record.to_dict()
        row.update({"GSM": gsm, "present_in_filtered_master": present})
        expected = corrected_values(record)
        for column in UPDATED_COLUMNS:
            row[f"before_{column}"] = before.loc[gsm, column] if gsm in before.index else np.nan
            row[f"corrected_{column}"] = expected[column]
            row[f"after_{column}"] = after.loc[gsm, column] if present else np.nan
        row["action"] = "corrected existing sample" if present else "absent; not added"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "GSM"]).reset_index(drop=True)


def create_tissue_subset(tissue: str, output: Path) -> None:
    temp = output.with_name(f".{output.name}.writing.tmp.h5ad")
    if temp.exists():
        temp.unlink()
    source = ad.read_h5ad(MASTER, backed="r")
    try:
        keep = source.obs["tissue"].astype(str).eq(tissue).to_numpy()
        expected = int(keep.sum())
        print(f"Writing corrected {tissue} subset: {expected:,} cells", flush=True)
        source[keep, :].copy(filename=temp)
    finally:
        source.file.close()
    check = ad.read_h5ad(temp, backed="r")
    try:
        if check.n_obs != expected or set(check.obs["tissue"].astype(str)) != {tissue}:
            raise ValueError(f"{tissue} subset validation failed")
    finally:
        check.file.close()
    os.replace(temp, output)


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    before_master = sample_metadata(MASTER)
    object_rows = [update_object(MASTER, "master")]
    after_master = sample_metadata(MASTER)
    build_master_correction_audit(before_master, after_master).to_csv(
        CORRECTION_AUDIT, index=False
    )

    # Recreate both tissue objects from the corrected filtered master so their
    # cell sets and metadata cannot drift from the master.
    create_tissue_subset("CSF", CSF)
    create_tissue_subset("PBMC", PBMC)
    for name, path in LINEAGES.items():
        object_rows.append(update_object(path, name))

    # Every supplied sample present in a lineage must agree with the master.
    master_samples = sample_metadata(MASTER)
    for name, path in LINEAGES.items():
        lineage_samples = sample_metadata(path)
        common = RAW_METADATA.index.intersection(lineage_samples.index)
        for column in UPDATED_COLUMNS:
            left = lineage_samples.loc[common, column].astype(str)
            right = master_samples.loc[common, column].astype(str)
            if not left.equals(right):
                raise ValueError(f"{name}: {column} does not match corrected master")

    pd.DataFrame(object_rows).to_csv(OBJECT_AUDIT, index=False)
    print(f"Correction audit: {CORRECTION_AUDIT}")
    print(f"Object audit: {OBJECT_AUDIT}")


if __name__ == "__main__":
    main()
