"""
--------------------------------------------- OAS AQUA -------------------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics(doi: 10.1002/pro.4205)\n
------------------------------------------- Analysis ---------------------------------------------\n
Set of programs that help analyze downloaded and processed data. This file parses a file or
directory of OAS AQUA downloads and returns the distribution of categories in a specific column.
"""

from pathlib import Path
from collections import Counter
from typing import cast, Optional
from argparse import ArgumentParser, Namespace

import pandas as pd

from tqdm import tqdm

from postprocessing.post_processing import DTYPE_DICT


class ArgsTypes(Namespace):
    """
    ## Argparse type definitions
    """

    file: Optional[str]
    directory: Optional[str]
    category: str


def read_file(file_path: Path, usecols: str) -> pd.DataFrame:
    """
    ## Returns a Dataframe from a csv file
    """
    return pd.read_csv(filepath_or_buffer=file_path, usecols=usecols, dtype=DTYPE_DICT)


def read_directory(directory_path: Path) -> list[Path]:
    """
    ## Returns a list of Dataframes from a directory of csv files
    """
    files_list: list[Path] = []
    for file in directory_path.iterdir():
        if file.is_file() and file.suffix == ".csv":
            files_list.append(file)

    return files_list


def main(arguments: ArgsTypes) -> None:
    """
    ## Prints summary of results
    """
    # Instantiate counter and totals
    counter: Counter = Counter()
    total: int = 0

    # Get files
    if arguments.file:
        data_list = [arguments.file]
    else:
        data_list = read_directory(arguments.directory)

    # Analyze files
    for data in tqdm(data_list, total=len(data_list)):
        series = read_file(data, arguments.category)
        counter.update(series)
        total += len(series)

    # Build summary table
    summary = pd.Series(counter, name="count").sort_values(ascending=False).to_frame()
    summary["percent"] = 100 * summary["count"] / total if total else 0.0

    # Print table
    print(summary)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="OAS AQUA",
        description="Analyzes the category distribution of a given OAS AQUA file set",
        epilog="For more information on OAS AQUA visit: https://github.com/lschaus0408/OASAQUA",
    )

    parser.add_argument(
        "-f", "--file", help="Path to file to be analyzed", default=None
    )

    parser.add_argument(
        "-d", "--directory", help="Path to directory to be analyzed", default=None
    )

    parser.add_argument(
        "-c", "--category", help="Column name for category to summarize"
    )

    args = cast(ArgsTypes, parser.parse_args())

    if args.file is None and args.directory is None:
        raise ValueError("Please provide either file path (-f) or directory path (-d).")

    if args.file is not None and args.directory is not None:
        raise ValueError(
            "Please provide only one of these two options: file path (-f) or directory path (-d)."
        )

    main(arguments=args)
