#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Levelling-rod calibration data importer / cleaner
-------------------------------------------------
• Reads 'Tabelle1' from the Excel/ODS workbook.
• Applies a configurable sequence of row-filters and field-transforms.
• Exports:
    - full_list_clean.pkl  (all cleaned fields per row)
    - export_list.pkl      (only fields in `export_fields`)
    - export_tensor.pt     ( Nxlen(export_fields_numeric) tensor )
"""

import pandas as pd
import torch
import pickle
from pathlib import Path

# ------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------

FILEPATH = (
    "../data_stochastic_modelling/data_levelling_rod_calibration/"
    "dataset_levelling_rod_calibration.ods"
)

# --- mapping raw staff types -> reduced 5-class code
REDUCTION = {
    # Leica-2 m
    "leicagpcl2": "l2m",
    "gpcl2": "l2m",
    "wildgpcl2": "l2m",
    "geozmidi": "l2m",
    # Leica-3 m
    "leicagpcl3": "l3m",
    "gpcl3": "l3m",
    'gpcl3(ref.)': 'l3m',
    "wildgpcl3": "l3m",
    "nedogpcl3": "l3m",
    # Trimble/Zeiss 2 m
    "zeissld12": "t2m",
    # Trimble/Zeiss 3 m
    "ld13": "t3m",
    "trimbleld13": "t3m",
    "zeissld13": "t3m",
    # everything else
}

reduction_dict_classes = {
    'gpcl2': 'l2m',
    'gpcl3': 'l3m',
    'gpcl3(ref.)': 'l3m',
    'ld13': 't3m',
    'leicagpcl0.5mini': 'misc',
    'leicagpcl2': 'l2m',
    'leicagpcl3': 'l3m',
    'leicagwcl92': 'misc',
    'leicagwcl182': 'misc',
    'nedogpcl3': 'l3m',
    'trimbleld13': 't3m',
    'wildgpcl2': 'l2m',
    'wildgpcl3': 'l3m',
    'zeissgwld182': 'misc',
    'zeissgwld92': 'misc',
    'zeissld12': 't2m',
    'zeissld13': 't3m',
    'geozmidi': 'l2m',
    'geozmidi(0.9m)': 'misc',
    'geozmidi(1m)': 'misc',
    'geozmini': 'misc',
    'geozmini(0.2m)': 'misc'
}


REQUIRED = {
    "Staff Type",
    "Staff Serial Number",
    "Overall Offset [µm]",
    "Overall Scale [ppm]",
}

# Which columns we ultimately want in the *small* exported list
export_fields = [
    "job_nr",
    "staff_type_reduced",
    "staff_id",
    "overall_offset",
    "overall_scale",
]

# Which of those are numeric and should go to a tensor
export_fields_numeric = ["overall_offset", "overall_scale"]

# ------------------------------------------------------------------
# 2. HELPER FUNCTIONS (filters & transforms)
# ------------------------------------------------------------------

# def filter_nan(row: dict) -> bool:
#     """Return True if row should be kept (i.e. contains no NaN / None)."""
#     return all(not pd.isna(v) for v in row.values())

def filter_nan(row: dict) -> bool:
    """Keep row only if the *required* fields are present and non-NaN."""
    return all(
        (k not in REQUIRED) or (not pd.isna(row[k]))
        for k in row
    )

def filter_mittelwert(row: dict) -> bool:
    """Drop rows whose 'staff_type' is 'mittelwert'."""
    return row.get("staff_type") != "mittelwert"

row_filters = [filter_nan, filter_mittelwert]  # add more whenever you like

# ---- Field transforms ------------------------------------------------

def normalize_staff_type(row: dict):
    raw = str(row["Staff Type"]).replace(" ", "").lower()
    row["staff_type"] = raw
    row["staff_type_reduced"] = REDUCTION.get(raw, "misc")

def add_numeric_aliases(row: dict):
    # rename long column headings once → shorter snake_case keys
    row["overall_offset"] = float(row["Overall Offset [µm]"])
    row["overall_scale"] = float(row["Overall Scale [ppm]"])
    
def add_job_nr(row: dict):
    # Excel column header is “Job Number”
    # cast to str to keep any leading zeros
    row["job_nr"] = str(row["Job Number"])

_field_transforms = [normalize_staff_type, add_numeric_aliases, add_job_nr]  # extend as needed

# ------------------------------------------------------------------
# 3. PIPELINE CLASS
# ------------------------------------------------------------------

class RodDataLoader:
    def __init__(self, path: str):
        # read sheet
        df = pd.read_excel(path, sheet_name="Tabelle1", skiprows=4)
        headers = pd.read_excel(path, sheet_name="Tabelle1", nrows=5).iloc[3]
        df.columns = headers
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
        self.rows = df.to_dict(orient="records")

        # create serial->id map
        serials = sorted({str(r["Staff Serial Number"]) for r in self.rows})
        self.serial_to_id = {s: i  for i, s in enumerate(serials)}

    # -------------------- public API ------------------------

    def clean(self):
        # 1. run transforms + filters first -------------------------------
        kept = []
        for r in self.rows:
            for func in _field_transforms:
                func(r)
            if all(f(r) for f in row_filters):
                kept.append(r)
    
        # 2. build a *dense* serial → id map from the kept rows -----------
        serials = sorted({str(r["Staff Serial Number"]) for r in kept})
        self.serial_to_id = {s: i for i, s in enumerate(serials)}   # 0 … n-1 dense
    
        # 3. write staff_id into each kept row ----------------------------
        for r in kept:
            r["staff_id"] = self.serial_to_id[str(r["Staff Serial Number"])]
    
        self.rows = kept
        return self  # chainable

    # select subset of keys
    def export_list(self, keys):
        return [{k: row[k] for k in keys} for row in self.rows]

    # numeric tensor
    def to_tensor(self, keys_numeric):
        data = [[row[k] for k in keys_numeric] for row in self.rows]
        return torch.tensor(data, dtype=torch.float32)

# ------------------------------------------------------------------
# 4. RUN THE PIPELINE
# ------------------------------------------------------------------

data = RodDataLoader(FILEPATH).clean()

full_list = data.rows
export_list_small = data.export_list(export_fields)
export_tensor = data.to_tensor(export_fields_numeric)

# ------------------------------------------------------------------
# 5. SAVE
# ------------------------------------------------------------------

out_dir = Path("../data_stochastic_modelling/data_levelling_rod_calibration")
out_dir.mkdir(parents=True, exist_ok=True)

with (out_dir / "full_list_clean.pkl").open("wb") as f:
    pickle.dump(full_list, f)

with (out_dir / "minimal_list.pkl").open("wb") as f:
    pickle.dump(export_list_small, f)

torch.save(export_tensor, out_dir / "minimal_tensor.pt")

print(f"✨ Saved {len(full_list)} clean rows → {out_dir}")
