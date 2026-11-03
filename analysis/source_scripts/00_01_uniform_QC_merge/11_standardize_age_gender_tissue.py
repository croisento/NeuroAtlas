#!/usr/bin/env python3
"""Standardize age, gender, and tissue metadata in corrected AnnData files."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import h5py
import numpy as np


PROJECT_DIR = Path("/home/lishaogang/NEU2026_July")
H5AD_FILES = [
    PROJECT_DIR / "adata_final_Disease_corrected.h5ad",
    PROJECT_DIR / "adata_final_csf_Disease_corrected.h5ad",
    PROJECT_DIR / "adata_final_pbmc_Disease_corrected.h5ad",
]

SAMPLE_AGE_GENDER = {
    "GSM7884009": (27, "Female"),  # IIH-1_GEX
    "GSM7884010": (52, "Female"),  # IIH-2_GEX
    "GSM7884012": (25, "Female"),  # IIH-4_GEX
    "GSM7884013": (50, "Female"),  # IIH-5_GEX
    "GSM7884014": (24, "Male"),    # RIS-2_GEX
    "GSM7884015": (28, "Female"),  # RIS-3_GEX
    "GSM7884016": (28, "Female"),  # MS-5_GEX
    "GSM7884017": (40, "Female"),  # MS-12_GEX
    "GSM7884018": (31, "Female"),  # MS-13_GEX
    "GSM7884019": (11, "Male"),    # NT1-2_GEX
    "GSM7884021": (17, "Female"),  # NT1-4_GEX
    "GSM7884022": (39, "Female"),  # NT1-5_GEX
    "GSM7884028": (32, "Female"),  # NT1-12_GEX
    "GSM7884032": (14, "Male"),    # NT2-7_GEX
}
SAMPLE_GENDER_ONLY = {"GSM7818261": "Female"}

GENDER_MAP = {
    "f": "Female",
    "female": "Female",
    "m": "Male",
    "male": "Male",
}
TISSUE_MAP = {
    "CSF": "CSF",
    "PB": "PBMC",
    "PBMC": "PBMC",
    "BLD": "PBMC",
    "DLB": "PBMC",  # Defensive alias for the reported BLD/DLB typo.
}


def read_column(obs: h5py.Group, column: str) -> np.ndarray:
    element = obs[column]
    if isinstance(element, h5py.Group):
        categories = np.asarray(element["categories"]).astype(str)
        codes = np.asarray(element["codes"])
        values = np.full(len(codes), None, dtype=object)
        valid = codes >= 0
        values[valid] = categories[codes[valid]]
        return values
    return np.asarray(element)


def write_categorical(obs: h5py.Group, column: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=str)
    categories = np.asarray(sorted(np.unique(values)), dtype=object)
    category_index = {value: index for index, value in enumerate(categories)}
    codes = np.asarray([category_index[value] for value in values], dtype=np.int16)

    if column in obs:
        del obs[column]
    group = obs.create_group(column)
    group.attrs["encoding-type"] = "categorical"
    group.attrs["encoding-version"] = "0.2.0"
    group.attrs["ordered"] = False
    category_dataset = group.create_dataset(
        "categories",
        data=categories,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )
    category_dataset.attrs["encoding-type"] = "string-array"
    category_dataset.attrs["encoding-version"] = "0.2.0"
    code_dataset = group.create_dataset("codes", data=codes)
    code_dataset.attrs["encoding-type"] = "array"
    code_dataset.attrs["encoding-version"] = "0.2.0"


def write_numeric(obs: h5py.Group, column: str, values: np.ndarray) -> None:
    if column in obs:
        del obs[column]
    dataset = obs.create_dataset(column, data=np.asarray(values, dtype=np.float64))
    dataset.attrs["encoding-type"] = "array"
    dataset.attrs["encoding-version"] = "0.2.0"

    column_order = [str(value) for value in obs.attrs["column-order"]]
    if column not in column_order:
        insert_at = column_order.index("age") + 1
        column_order.insert(insert_at, column)
        del obs.attrs["column-order"]
        obs.attrs.create(
            "column-order",
            np.asarray(column_order, dtype=h5py.string_dtype(encoding="utf-8")),
        )


def format_age(value: float) -> str:
    if not np.isfinite(value):
        return "Unknown"
    return f"{value:g}"


def update_file(path: Path) -> None:
    temp_path = path.with_name(f".{path.name}.metadata_standardizing.tmp")
    if temp_path.exists():
        temp_path.unlink()
    print(f"Creating temporary copy: {path.name}", flush=True)
    shutil.copy2(path, temp_path)

    try:
        with h5py.File(temp_path, "r+") as handle:
            obs = handle["obs"]
            samples = read_column(obs, "orig.ident").astype(str)
            if "age_numeric" in obs:
                ages = np.asarray(obs["age_numeric"], dtype=float)
            else:
                ages = np.asarray(read_column(obs, "age"), dtype=float)
            genders = read_column(obs, "gender")
            tissues = read_column(obs, "tissue")

            for gsm, (age, gender) in SAMPLE_AGE_GENDER.items():
                mask = samples == gsm
                ages[mask] = age
                genders[mask] = gender
            for gsm, gender in SAMPLE_GENDER_ONLY.items():
                genders[samples == gsm] = gender

            standardized_gender = np.asarray(
                [GENDER_MAP.get(str(value).strip().lower(), "Unknown") for value in genders],
                dtype=object,
            )
            standardized_tissue = np.asarray(
                [TISSUE_MAP.get(str(value).strip().upper(), "Unknown") for value in tissues],
                dtype=object,
            )
            age_labels = np.asarray([format_age(value) for value in ages], dtype=object)

            write_numeric(obs, "age_numeric", ages)
            write_categorical(obs, "age", age_labels)
            write_categorical(obs, "gender", standardized_gender)
            write_categorical(obs, "tissue", standardized_tissue)
            handle.flush()

        with h5py.File(temp_path, "r") as handle:
            obs = handle["obs"]
            genders = set(read_column(obs, "gender"))
            tissues = set(read_column(obs, "tissue"))
            if not genders <= {"Female", "Male", "Unknown"}:
                raise ValueError(f"Unexpected gender labels in {path.name}: {genders}")
            if not tissues <= {"CSF", "PBMC"}:
                raise ValueError(f"Unexpected tissue labels in {path.name}: {tissues}")
            if np.count_nonzero(read_column(obs, "age") == "Unknown") != np.isnan(
                np.asarray(obs["age_numeric"], dtype=float)
            ).sum():
                raise ValueError(f"Age/age_numeric mismatch in {path.name}")

        os.replace(temp_path, path)
        print(f"Validated and replaced: {path}")
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def main() -> None:
    for path in H5AD_FILES:
        update_file(path)


if __name__ == "__main__":
    main()
