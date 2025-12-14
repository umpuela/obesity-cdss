from typing import Any

import pandas as pd


def get_name_translations(data_dict: dict) -> dict[str, str]:
    """Generates a translation dictionary {original: description}."""
    labels = {}
    for key, meta in data_dict.items():
        if "original_name" in meta:
            labels[meta["original_name"]] = meta["description"]

        labels[key] = meta["description"]

    return labels


def get_value_translations_and_order(
    data_dict: dict, col: str
) -> tuple[dict[str, str], list[str]]:
    """
    Gets the original translation map {EN: PT}
    and the ordered list of values in Portuguese.
    """
    if not data_dict or col not in data_dict:
        return {}, []

    if "values" not in data_dict[col]:
        return {}, []

    translation_map = data_dict[col]["values"]
    ordered_pt = list(translation_map.values())

    return translation_map, ordered_pt


def translate_single_value(data_dict: dict, col_name: str, raw_value: Any) -> str:
    """
    Translates a single value (raw_value) based on the data dictionary.
    It automatically handles the conversion of float to integer string.
    """
    trans_map = get_value_translations_and_order(data_dict, col_name)[0]

    if not trans_map:
        return str(raw_value)

    lookup_key = str(raw_value)

    if isinstance(raw_value, float) and raw_value.is_integer():
        lookup_key = str(int(raw_value))

    return trans_map.get(lookup_key, str(raw_value))


def highlight_cell_max(series: pd.Series, bg_color: str, text_color: str) -> list[str]:
    """Colors the background of a table cell with the maximum value in the column."""
    is_max = series == series.max()
    return [
        f"background-color: {bg_color}; color: {text_color}; font-weight: bold"
        if value
        else ""
        for value in is_max
    ]
