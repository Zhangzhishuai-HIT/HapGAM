"""Parsing of GRIN-Global soybean descriptor downloads.

Each descriptor file holds one row per (accession, evaluation) observation.
The evaluation is identified by ``method_name`` (e.g. ``SOYBEAN.EVALUATION.MS967``),
which we treat as the evaluation batch (trial) of the record.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# descriptor code -> (kind, short name)
QUANTITATIVE = {
    "MATDATE": "maturity_doy",
    "FLOWERDATE": "flowering_doy",
    "HEIGHT": "height_cm",
    "SEEDWEIGHT": "seed_weight_g",
    "YIELD": "yield_mg_ha",
    "OIL": "oil_pct",
    "PROTEIN": "protein_pct",
    "LODGING": "lodging_score",
    "TERMSCORE": "stem_term_score",
    "MATGROUP": "maturity_group",
}
CATEGORICAL = {
    "FLWRCOLOR": "flower_color",
    "PUBCOLOR": "pubescence_color",
    "STEMTERM": "stem_termination",
    "PODCOLOR": "pod_color",
    "HILUMCOLOR": "hilum_color",
    "SCOATCOLOR": "seed_coat_color",
    "LOCUS_E3": "e3_allele",
}

_MG = {"000": -2.0, "00": -1.0, "0": 0.0, "I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0,
       "V": 5.0, "VI": 6.0, "VII": 7.0, "VIII": 8.0, "IX": 9.0, "X": 10.0}
_CUMDAYS = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])


def canonical_id(value: object) -> str:
    """Upper-case accession identifier without spaces, dashes, dots or underscores."""
    return re.sub(r"[\s_\-\.]+", "", str(value).strip().upper())


def mmdd_to_doy(value: str) -> float:
    """Convert GRIN month-day codes (``925`` = Sep 25, ``1011`` = Oct 11) to day of year."""
    try:
        code = int(float(value))
    except (TypeError, ValueError):
        return np.nan
    month, day = divmod(code, 100)
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return np.nan
    return float(_CUMDAYS[month - 1] + day)


def read_descriptor(path: Path) -> pd.DataFrame:
    """Read one descriptor CSV (two banner lines precede the header)."""
    df = pd.read_csv(path, skiprows=2, dtype=str, keep_default_na=False)
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    acc = df["accession_prefix"] + df["accession_number"] + df["accession_suffix"]
    out = pd.DataFrame({
        "accession": acc.map(canonical_id),
        "accession_label": (df["accession_prefix"] + " " + df["accession_number"] + " "
                            + df["accession_suffix"]).str.strip(),
        "descriptor": df["coded_name"].str.strip(),
        "raw_value": df["observation_value"].str.strip(),
        "batch": df["method_name"].str.strip(),
        "taxon": df["taxon"].str.strip(),
        "origin": df["origin"].str.strip(),
        "source_type": df["source_type_code"].str.strip(),
        "latitude": pd.to_numeric(df["latitude"], errors="coerce"),
        "longitude": pd.to_numeric(df["longitude"], errors="coerce"),
        "plant_name": df["plant_name"].str.strip(),
    })
    return out


def numeric_value(descriptor: str, raw: pd.Series) -> pd.Series:
    if descriptor in ("MATDATE", "FLOWERDATE"):
        return raw.map(mmdd_to_doy)
    if descriptor == "MATGROUP":
        return raw.str.upper().map(_MG).astype(float)
    return pd.to_numeric(raw, errors="coerce")


def load_all(directory: Path) -> pd.DataFrame:
    """Long table of all quantitative and categorical observations."""
    frames = []
    for path in sorted(Path(directory).glob("GRIN_crop51_*.csv")):
        df = read_descriptor(path)
        if df.empty:
            continue
        code = df["descriptor"].iloc[0]
        if code in QUANTITATIVE:
            df["trait"] = QUANTITATIVE[code]
            df["value"] = numeric_value(code, df["raw_value"])
            df["kind"] = "quantitative"
        elif code in CATEGORICAL:
            df["trait"] = CATEGORICAL[code]
            df["value"] = np.nan
            df["kind"] = "categorical"
        else:
            continue
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# Region grouping of countries of origin for grouped (leave-region-out) validation.
REGION = {
    "China": "China", "Taiwan": "China", "Hong Kong": "China",
    "Korea, South": "Korea", "Korea, North": "Korea", "Korea": "Korea",
    "Japan": "Japan",
    "Russian Federation": "Russia/C.Asia", "Russia": "Russia/C.Asia", "Soviet Union": "Russia/C.Asia",
    "Georgia": "Russia/C.Asia", "Kazakhstan": "Russia/C.Asia", "Uzbekistan": "Russia/C.Asia",
    "Ukraine": "Europe", "Moldova": "Europe",
    "United States": "Americas", "Canada": "Americas", "Brazil": "Americas", "Argentina": "Americas",
    "Mexico": "Americas", "Colombia": "Americas", "Peru": "Americas", "Paraguay": "Americas",
    "Bolivia": "Americas", "Venezuela": "Americas", "Uruguay": "Americas", "Ecuador": "Americas",
    "Guatemala": "Americas", "Costa Rica": "Americas", "Cuba": "Americas", "Chile": "Americas",
    "Nicaragua": "Americas", "Honduras": "Americas", "El Salvador": "Americas", "Panama": "Americas",
    "India": "S/SE Asia", "Nepal": "S/SE Asia", "Pakistan": "S/SE Asia", "Bhutan": "S/SE Asia",
    "Indonesia": "S/SE Asia", "Vietnam": "S/SE Asia", "Thailand": "S/SE Asia",
    "Philippines": "S/SE Asia", "Malaysia": "S/SE Asia", "Myanmar": "S/SE Asia",
    "Cambodia": "S/SE Asia", "Laos": "S/SE Asia", "Sri Lanka": "S/SE Asia", "Bangladesh": "S/SE Asia",
}
EUROPE = {"Germany", "France", "Sweden", "Netherlands", "Hungary", "Yugoslavia", "Serbia",
          "Romania", "Austria", "Italy", "Poland", "Czechoslovakia", "Czech Republic", "Bulgaria",
          "Switzerland", "Spain", "Portugal", "United Kingdom", "Belgium", "Croatia", "Slovakia",
          "Denmark", "Finland", "Norway", "Greece", "Albania", "Macedonia", "Slovenia", "Belarus",
          "Lithuania", "Latvia", "Estonia", "Ireland", "Bosnia and Herzegovina", "Montenegro"}


def region_of(origin: str) -> str:
    if origin in REGION:
        return REGION[origin]
    if origin in EUROPE:
        return "Europe"
    if not origin or origin == "Unknown":
        return "Unknown"
    return "Other"
