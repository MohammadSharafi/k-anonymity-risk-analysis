from __future__ import annotations

from typing import Iterable

import pandas as pd


EDUCATION_LEVEL_1 = {
    "Preschool": "Primary",
    "1st-4th": "Primary",
    "5th-6th": "Primary",
    "7th-8th": "Middle-School",
    "9th": "High-School",
    "10th": "High-School",
    "11th": "High-School",
    "12th": "High-School",
    "HS-grad": "High-School",
    "Some-college": "Some-College",
    "Assoc-acdm": "Associate",
    "Assoc-voc": "Associate",
    "Bachelors": "Bachelors",
    "Masters": "Graduate",
    "Prof-school": "Graduate",
    "Doctorate": "Graduate",
}

EDUCATION_LEVEL_2 = {
    "Primary": "Pre-College",
    "Middle-School": "Pre-College",
    "High-School": "Pre-College",
    "Some-College": "College-or-Higher",
    "Associate": "College-or-Higher",
    "Bachelors": "College-or-Higher",
    "Graduate": "College-or-Higher",
}

MARITAL_LEVEL_1 = {
    "Married-civ-spouse": "Married",
    "Married-AF-spouse": "Married",
    "Married-spouse-absent": "Separated",
    "Divorced": "Separated",
    "Separated": "Separated",
    "Widowed": "Widowed",
    "Never-married": "Never-married",
}

MARITAL_LEVEL_2 = {
    "Married": "Partnered",
    "Separated": "Not-Partnered",
    "Widowed": "Not-Partnered",
    "Never-married": "Not-Partnered",
}

OCCUPATION_LEVEL_1 = {
    "Exec-managerial": "Managerial-Professional",
    "Prof-specialty": "Managerial-Professional",
    "Tech-support": "Managerial-Professional",
    "Adm-clerical": "Clerical-Sales",
    "Sales": "Clerical-Sales",
    "Craft-repair": "Manual-Labor",
    "Machine-op-inspct": "Manual-Labor",
    "Transport-moving": "Manual-Labor",
    "Handlers-cleaners": "Manual-Labor",
    "Farming-fishing": "Manual-Labor",
    "Other-service": "Service",
    "Priv-house-serv": "Service",
    "Protective-serv": "Service",
    "Armed-Forces": "Other",
    "Unknown": "Other",
}

OCCUPATION_LEVEL_2 = {
    "Managerial-Professional": "White-Collar",
    "Clerical-Sales": "White-Collar",
    "Manual-Labor": "Blue-Collar",
    "Service": "Service",
    "Other": "Other",
}

COUNTRY_TO_REGION = {
    "United-States": "US",
    "Canada": "North-America",
    "Mexico": "Latin-America",
    "Puerto-Rico": "Latin-America",
    "Cuba": "Latin-America",
    "Dominican-Republic": "Latin-America",
    "El-Salvador": "Latin-America",
    "Guatemala": "Latin-America",
    "Haiti": "Latin-America",
    "Honduras": "Latin-America",
    "Jamaica": "Latin-America",
    "Nicaragua": "Latin-America",
    "Outlying-US(Guam-USVI-etc)": "North-America",
    "Trinadad&Tobago": "Latin-America",
    "Ecuador": "Latin-America",
    "Peru": "Latin-America",
    "Columbia": "Latin-America",
    "Cambodia": "Asia",
    "China": "Asia",
    "Hong": "Asia",
    "India": "Asia",
    "Iran": "Asia",
    "Japan": "Asia",
    "Laos": "Asia",
    "Philippines": "Asia",
    "Taiwan": "Asia",
    "Thailand": "Asia",
    "Vietnam": "Asia",
    "England": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Greece": "Europe",
    "Holand-Netherlands": "Europe",
    "Hungary": "Europe",
    "Ireland": "Europe",
    "Italy": "Europe",
    "Poland": "Europe",
    "Portugal": "Europe",
    "Scotland": "Europe",
    "Yugoslavia": "Europe",
    "South": "Other",
    "Unknown": "Unknown",
}

MAX_LEVELS = {
    "age": 4,
    "education": 3,
    "marital_status": 3,
    "occupation": 3,
    "native_country": 3,
    "sex": 1,
}


def max_level(attribute: str) -> int:
    return MAX_LEVELS.get(attribute, 1)


def generalize_frame(
    df: pd.DataFrame,
    qi_columns: Iterable[str],
    levels: dict[str, int],
) -> pd.DataFrame:
    generalized = df.copy()
    for column in qi_columns:
        generalized[column] = generalize_series(generalized[column], column, levels.get(column, 0))
    return generalized


def generalize_series(series: pd.Series, attribute: str, level: int) -> pd.Series:
    if level <= 0:
        return series.astype("string")

    if attribute == "age":
        return _generalize_age(series, level)
    if attribute == "education":
        return _generalize_nested_map(series, level, [EDUCATION_LEVEL_1, EDUCATION_LEVEL_2])
    if attribute == "marital_status":
        return _generalize_nested_map(series, level, [MARITAL_LEVEL_1, MARITAL_LEVEL_2])
    if attribute == "occupation":
        return _generalize_nested_map(series, level, [OCCUPATION_LEVEL_1, OCCUPATION_LEVEL_2])
    if attribute == "native_country":
        return _generalize_country(series, level)
    if attribute == "sex":
        return _generalize_sex(series, level)
    return _full_suppression(series) if level >= 1 else series.astype("string")


def _generalize_age(series: pd.Series, level: int) -> pd.Series:
    ages = pd.to_numeric(series, errors="coerce")

    if level == 1:
        return ages.map(lambda age: _format_band(age, width=5)).astype("string")
    if level == 2:
        return ages.map(lambda age: _format_band(age, width=10)).astype("string")
    if level == 3:
        return ages.map(_broad_age_band).astype("string")
    return _full_suppression(series)


def _format_band(age: float | int | None, width: int) -> str:
    if pd.isna(age):
        return "Unknown"
    age_int = int(age)
    start = (age_int // width) * width
    end = start + width - 1
    return f"{start}-{end}"


def _broad_age_band(age: float | int | None) -> str:
    if pd.isna(age):
        return "Unknown"
    age_int = int(age)
    if age_int < 25:
        return "17-24"
    if age_int < 35:
        return "25-34"
    if age_int < 45:
        return "35-44"
    if age_int < 55:
        return "45-54"
    if age_int < 65:
        return "55-64"
    return "65+"


def _generalize_nested_map(series: pd.Series, level: int, mappings: list[dict[str, str]]) -> pd.Series:
    values = series.astype("string")
    capped_level = min(level, len(mappings) + 1)
    for mapping in mappings[: capped_level - 1]:
        values = values.map(lambda value: mapping.get(value, value))
    if capped_level == len(mappings) + 1:
        return _full_suppression(values)
    return values


def _generalize_country(series: pd.Series, level: int) -> pd.Series:
    values = series.astype("string")
    if level == 1:
        return values.map(lambda value: COUNTRY_TO_REGION.get(value, "Other"))
    if level == 2:
        return values.map(lambda value: "US" if value == "United-States" else "Non-US")
    return _full_suppression(values)


def _generalize_sex(series: pd.Series, level: int) -> pd.Series:
    return _full_suppression(series) if level >= 1 else series.astype("string")


def _full_suppression(series: pd.Series) -> pd.Series:
    return pd.Series(["*"] * len(series), index=series.index, dtype="string")
