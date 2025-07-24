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

from pathlib import Path
from typing import Literal, Any, TypeAlias
from argparse import ArgumentParser
from dataclasses import dataclass

import re
import ast
import json
import yaml

FileFormatType: TypeAlias = Literal["json", "yaml", "yml"]

current_file_path = Path(__file__).resolve()
main_directory = current_file_path.parent
query_file_path = main_directory / "files" / "super_queries.json"
with open(query_file_path, "r", encoding="utf-8") as infile:
    QUERY_ALIASES = json.load(infile)


class ConfigValidationError(Exception):
    """
    ## Exception raised for errors in the config file
    """

    def __init__(self, message: str, config: str = None):
        self.message = message
        self.config = config
        super().__init__(self.message)

    def __str__(self):
        if self.config:
            return f"[{self.config}] {self.message}"
        else:
            return self.message


@dataclass
class OASRun:
    """
    ## Object representing the config of an OAS AQUA Run
    """

    run_number: int
    query: dict[str, Any]
    postprocessors: list[dict[str, dict[str, Any]]]
    # See sheet
    default_queries = {"ProcessingMode": "Default", "KeepDownloads": "Default"}

    def __repr__(self):
        return f"Run number: {self.run_number} \nQuery: {self.query} \
        \nPostprocessors: {self.postprocessors}"

    def fill_defaults(self):
        """
        ## Adds default values to missing queries and postprocessors
        """
        for key, value in self.default_queries.items():
            self.query.setdefault(key, value)

    def validate_run(self):
        """
        ## Raises Exception if Config file is incorrect
        """
        if self.query is not None:
            try:
                self.query["Database"]
            except KeyError as error:
                raise ConfigValidationError(
                    "Database needs to be specified in config", "database"
                ) from error


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


def argparse_dict_type(argument: str) -> tuple:
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
        prog="OAS AQUA",
        description="A downloader, packaging tool, and dataset manager for OAS",
        epilog="For more information on OAS AQUA visit: https://github.com/lschaus0408/OASAQUA",
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
    parser.add_argument("-n", "--filename", help="Prefix for all filenames")
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
        "--file_config",
        type=str,
        help="Path to JSON or YAML configuration file for OASCS. "
        "When used, ignores all other arguments.",
    )

    return parser


def load_config(path: Path, file_format: FileFormatType = "yaml") -> dict:
    """
    ## Loads the OAS AQUA Config
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


def parse_config_file(
    config: dict[str, list[dict[str, Any]]], file_format: FileFormatType = "yaml"
) -> list[OASRun]:
    """
    ## Parses OAS AQUA YAML Configs
    """
    runs: list[OASRun] = []

    # Iterate through runs
    for run in config["Runs"]:
        # Get basic config
        run_number = run["RunNumber"]

        # Process query block
        if run.get("Query"):
            query_config = run["Query"]

            # Adjust Output name
            full_output_directory = Path(query_config["OutputDir"])
            full_output_directory = full_output_directory / f"Run_{run_number}"
            full_output_directory.mkdir()
            query_config["OutputDir"] = full_output_directory

            if file_format == "yaml" or file_format == "yml":
                # Get Attributes as flat dict
                attributes = {}
                for item in query_config.get("Attributes", []):
                    item = get_aliases(item)
                    attributes.update(item)
                query_config["Attributes"] = attributes
        else:
            query_config = None

        # Get Post Processing
        postprocessing_config = run.get("PostProcessing")
        if postprocessing_config:
            postprocessors = parse_postprocessing_info(postprocessing_config)
        else:
            postprocessors = None

        configured_run = OASRun(
            run_number=run_number, query=query_config, postprocessors=postprocessors
        )
        runs.append(configured_run)
    return runs


def get_aliases(query: dict[str, str]) -> dict[str, str]:
    """
    ## Converts query aliases into real queries
    """
    ((key, value),) = query.items()
    try:
        new_value = QUERY_ALIASES[key][value]
    except KeyError:
        new_value = value
    return {key: new_value}


def parse_postprocessing_info(postprocessing_config: dict) -> list[dict]:
    """
    ## Parses postprocessing
    """
    postprocessors = []
    # Get Program Names and args
    for program_name, argument_list in postprocessing_config.items():
        postprocessing_arguments = {}
        if argument_list not in (None, ""):  # Skip empty
            if isinstance(argument_list, list):
                for argument in argument_list:
                    postprocessing_arguments.update(argument)
            elif isinstance(argument_list, dict):
                postprocessing_arguments = argument_list
        postprocessors.append({program_name: postprocessing_arguments})

    return postprocessors


if __name__ == "__main__":
    print("This is the OAS Parser!")
