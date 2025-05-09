"""
-------------------------- Observed Antibody Space Client Server Tool -----------------------------
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics (doi: 10.1002/pro.4205)
--------------------------------------- Argument Parser -------------------------------------------
This module is the main program of OAS API. It manages the communication between the different
modules to create the dataset. It will use oasdownload to download the files, csvreader to process
the data and filemanager to save the data/add data to existing files. Main is also responsible for
communicating with the post-processing tools.
"""

import re
import ast
import json
import yaml

from pathlib import Path
from typing import Literal
from argparse import ArgumentParser


def argparse_tuple_type(argument: str) -> tuple:
    """
    ## Takes an argument from argparse and transforms it into a tuple
    """
    mapped_argument = []
    trimmed_argument = argument[1:]
    # This pattern matches the input pattern ((X, A, B, ...), (Y, C, D, ...))
    tuple_pattern = r"\(([^)]*(,[^)]*)+)\)"
    for match in re.finditer(pattern=tuple_pattern, string=trimmed_argument):
        start = match.start()
        end = match.end()
        # Add quotes around each entry for literal_eval
        with_literal_quotes = re.sub(
            r"([A-Za-z0-9\+\-\_\/\\]+)", r"'\1'", trimmed_argument[start:end]
        )
        # literal_eval turns string that looks like a tuple into a tuple
        tuple_argument = ast.literal_eval(with_literal_quotes)
        mapped_argument.append(tuple_argument)
    return tuple(mapped_argument)


def argparse_dict_type(argument: str) -> dict:
    """
    ## Takes an argument from argparse and transforms it into a dict
    """
    mapped_argument = []
    # This pattern matches the input pattern {X: [A, B, ...], Y: [C, D, ...]}
    dict_pattern = r"(\w+):\s*(\[[^\]]*\]|\S+)"
    for match in re.finditer(pattern=dict_pattern, string=argument):
        start = match.start()
        end = match.end()
        matched_argument = argument[start:end]
        # Replace some of the dict-specific notation
        matched_argument = (
            matched_argument.replace(":", ",").replace("[", "").replace("]", "")
        )
        # Remove the last comma that remains in case no bracket is used
        if matched_argument[-1] == ",":
            matched_argument = matched_argument[:-1]
        # Turn the string (list of words) into a tuple of strings
        tuple_argument = tuple(word.strip() for word in matched_argument.split(","))
        mapped_argument.append(tuple_argument)
    return tuple(mapped_argument)


def argparse_list_type(argument: str) -> list:
    """
    ## Takes an argument from argparse and transforms it into a list
    """
    argument = argument.replace("(", "").replace(")", "").replace(" ", "")
    return argument.split(",")


def oas_parser() -> ArgumentParser:
    """
    ## Parser for OASCS
    --> TO DO: Finish Docstrings, add postprocessing to parser
    """
    parser = ArgumentParser(
        prog="OAS API",
        description="A downloader, packaging tool, and dataset manager for OAS",
        epilog="For more information on OAS API visit: INSERT URL HERE",
    )
    parser.add_argument(
        "-p",
        "--paired",
        action="store_true",
        help="Download from the paired antibody database instead of the unpaired",
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        help="Output directory for downloaded files",
        default="./oas_api_downloads",
    )
    parser.add_argument("-f", "--filename", help="Prefix for all filenames")
    parser.add_argument(
        "-q",
        "--query",
        type=argparse_dict_type,
        help="Download queries for OAS API files, \
        see link below to find out what queries are allowed. \
        Example: {B-Type: Memory-B-Cells, Chain: [Light, Heavy]} \
        to query light chains and heavy chains from memory B-Cells",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        type=argparse_list_type,
        help="Determines what metadata from the files to keep. \
                        Example: (Author, Species, Chain)",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=argparse_list_type,
        help="Determines what column data to keep. \
                        Example: (aa_sequence, fwr, cdr, v_call)",
    )
    parser.add_argument(
        "-pm",
        "--processing_mode",
        choices=["Individual", "Bulk", "Split"],
        help="Determines how to process downloaded files. \
        Individual is recommended, see link below for details",
    )
    parser.add_argument(
        "-k",
        "--keep_downloads",
        choices=["keep", "delete", "move"],
        help="Determines what to do with raw files from download.",
    )

    parser.add_argument(
        "-f",
        "file_config",
        type=str,
        help="Path to JSON configuration file for OASCS. When used, ignores all other arguments.",
    )
    return parser


def load_config(
    path: Path, file_format: Literal["json", "yaml", "yml"] = "yaml"
) -> dict:
    """
    ## Loads the OASCS Config
    Config file should be in json format.
    Returns a dict
    """
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found at path: {path}")

    if file_format == "json":
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    if file_format == "yaml" or file_format == "yml":
        with open(path, "r", encoding="utf=8") as file:
            return yaml.safe_load(file)

    raise ValueError(
        f"Argument 'format' needs to be 'json' or 'yaml' not: {file_format}"
    )


if __name__ == "__main__":
    print("This is the OAS Parser!")
