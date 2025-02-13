from numpy import argmax
from pandas import merge
from pandas import DataFrame
from pandas import MultiIndex

from typing import Tuple


def add_common_column_name(df: DataFrame, category: str) -> DataFrame:
    """
    Adds common bottom category for all columns in the data frame.
    """
    return DataFrame(data=df.values, columns=MultiIndex.from_product([df.columns, [category]]))


def combine_two_dataframes(df1: DataFrame, df2: DataFrame, table_names: Tuple[str, str], on: str) -> DataFrame:
    """
    Merges two data frames but adds bottom level column categories to keep columns unique.
    Essentially creates table with initial column that have subheadings based on table names.
    """
    result = merge(
        add_common_column_name(df1, table_names[0]),
        add_common_column_name(df2, table_names[1]),
        left_on=[(on, table_names[0])], right_on=[(on, table_names[1])]
    ).drop(columns=[(on, table_names[1])])

    # Let erase table name under on column
    new_column_names = result.columns.tolist()
    new_column_names[argmax(result.columns.get_level_values(0) == on)] = (on, '-')
    result.columns = MultiIndex.from_tuples(new_column_names)

    # Reorder columns so that top level categories are adjacent
    return result.reindex(columns=result.columns.sortlevel(level=0)[0])

