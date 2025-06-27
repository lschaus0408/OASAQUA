"""
---------------------------------- Observed Antibody Space CS -----------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics(doi: 10.1002/pro.4205)\n
------------------------------------ Encoder Parent Class ---------------------------------------\n
Parent class for encoding post processors. Encoders only differ in their implementation of the
encoding, the rest of the class is contained in here.
"""

import io
import warnings

from pathlib import Path
from typing import Literal, Optional, Callable, TypeAlias
from itertools import cycle
from functools import partial
from abc import abstractmethod

import orjson

import pandas as pd
import numpy as np
import numpy.typing as npt

from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

from postprocessing.post_processing import PostProcessor, DTYPE_DICT
from postprocessing.sequence_tracker import SequenceIdType, ordered_sequence_ids

ArrayDict: TypeAlias = dict[str, npt.NDArray]
SaveFormat: TypeAlias = Literal["json", "numpy", "npy", "npz"]


class Encoder(PostProcessor):
    """
    ## Encoder Parent Class
    Contains all methods and attributes for encoders.
    """

    ALL_AMINO_ACIDS = {
        "A": 0,
        "R": 1,
        "N": 2,
        "D": 3,
        "C": 4,
        "E": 5,
        "Q": 6,
        "G": 7,
        "H": 8,
        "I": 9,
        "L": 10,
        "K": 11,
        "M": 12,
        "F": 13,
        "P": 14,
        "S": 15,
        "T": 16,
        "W": 17,
        "Y": 18,
        "V": 19,
        "X": 20,
    }

    def __init__(
        self,
        directory_or_file_path: Path,
        output_directory: Path,
        pad_size: int,
        save_format: SaveFormat,
        category_column: Optional[str] = None,
        n_jobs: int = 1,
        maximum_file_size_gb: Optional[int] = None,
        flatten: bool = False,
        output_file_prefix: str = "",
    ):
        super().__init__(
            directory_or_file_path=directory_or_file_path,
            output_directory=output_directory,
        )
        assert (
            isinstance(n_jobs, int) and n_jobs > 0
        ), f"n_jobs needs to be int > 0. Provided type of n_jobs: {type(n_jobs)}"

        self.pad_size = pad_size
        self.save_format = save_format
        self.category_column = category_column
        self.n_jobs = n_jobs
        self.maximum_file_size = maximum_file_size_gb
        self.flatten = flatten
        self.output_file_prefix = output_file_prefix

    def load_file(self, file_path: Path, overwrite=False):
        """
        ## Returns file at file_path as a dataframe
        """
        columns_to_use = ["Sequence_aa"]
        if self.category_column is not None:
            columns_to_use.append(self.category_column)
        return pd.read_csv(
            filepath_or_buffer=file_path,
            usecols=columns_to_use,
            dtype=DTYPE_DICT,
        )

    def save_file(self, file_path: Path, data: dict):
        """
        ## Saves files
        Saves file in the format provided by the save_format attribute.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        saving_factory = {
            "json": self._save_as_json,
            "numpy": self._save_as_numpy,
            "npy": self._save_as_numpy,
            "npz": self._save_as_numpy,
        }
        saving_factory[self.save_format](file_path=file_path, data=data)

    @staticmethod
    def _save_as_json(file_path: Path, data: dict):
        """
        ## Saves the file as json
        """
        json_string = orjson.dumps(data)  # pylint: disable=maybe-no-member
        file_path = file_path.with_suffix(".json")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(json_string)

    def _save_as_numpy(self, file_path: Path, data: dict):
        """
        ## Saves the file as npz
        """
        # Order the keys
        ordered_data_keys = ordered_sequence_ids(data.keys())

        # Extract the data into an array
        data_values = [data[key] for key in ordered_data_keys]

        # Save the array
        if self.save_format == "npz":
            file_path = file_path.with_suffix(".npz")
            np.savez(file=file_path, arr=data_values)
        else:
            file_path = file_path.with_suffix(".npy")
            np.save(file=file_path, arr=data_values)

    def process(self):
        """
        ## Processes data and one-hot encodes it
        """
        # Setup output dict
        encoded_sequence_dict = {}
        category_dict = {}
        # Load data
        self.get_files_list(directory_or_file_path=self.directory_or_file_path)
        file_number = 0

        for index, file in enumerate(self.all_files):
            data = self.load_file(file_path=file)
            if data.empty:
                file.unlink(missing_ok=True)
                continue
            # Remove indices where the sequence is greater than the pad_size
            data["sequence_lenght"] = data["Sequence_aa"].str.len()
            filtered_data = data[data["sequence_lenght"] <= self.pad_size]

            # Create SequenceIDType column
            filtered_data.reset_index(drop=True, inplace=True)
            filtered_data["file_id"] = index
            sequence_ids: list[SequenceIdType] = list(
                zip(filtered_data.file_id, filtered_data.index)
            )
            filtered_data["Tuple"] = sequence_ids

            # Create sequence dict to keep track of where the sequences came from
            sequences = filtered_data["Sequence_aa"].values
            sequence_dict = pd.Series(sequences, index=filtered_data.Tuple).to_dict()

            if self.category_column:
                categories = filtered_data[self.category_column].values
                category_dict.update(
                    pd.Series(categories, index=filtered_data.Tuple).to_dict()
                )

            processing_function = self.encoding_factory()
            encoded_sequence_dict.update(processing_function(sequence_dict))
            print("Data size estimation...")
            data_size = estimate_data_size(
                encoded_sequence_dict, data_format=self.save_format
            )
            print("Saving files...")
            if (
                self.maximum_file_size is not None
                and data_size > self.maximum_file_size * 1024**3
            ):
                # Save data file
                output_file_path = Path(
                    self.output_directory, f"{self.output_file_prefix}_{file_number}"
                )
                self.save_file(file_path=output_file_path, data=encoded_sequence_dict)

                # Save category file if it exists
                if self.category_column:
                    category_output_file_path = Path(
                        self.output_directory,
                        f"{self.output_file_prefix}_categories_{file_number}",
                    )
                    self.save_file(
                        file_path=category_output_file_path, data=category_dict
                    )

                file_number += 1

                # Clear data
                encoded_sequence_dict = {}
                category_dict = {}

        if encoded_sequence_dict:
            output_file_path = Path(
                self.output_directory, f"{self.output_file_prefix}_{file_number}"
            )
            self.save_file(file_path=output_file_path, data=encoded_sequence_dict)

        if category_dict:
            category_output_file_path = Path(
                self.output_directory,
                f"{self.output_file_prefix}_categories_{file_number}",
            )
            self.save_file(file_path=category_output_file_path, data=category_dict)

        # Clear data
        encoded_sequence_dict = {}
        category_dict = {}

    def encoding_factory(
        self,
    ) -> Callable[
        [dict[SequenceIdType, npt.NDArray[np.str_]]],
        dict[SequenceIdType, npt.NDArray[np.uint8]],
    ]:
        """
        ## Factory for multi- or single processing
        Determines if sequences should be encoded with one process
        or use multi-processing to encode sequences.
        """
        if self.n_jobs == 1:
            print(f"Encoding {self.output_file_prefix} via single process...")
            return self._encode_single_process
        else:
            # In 3.11 pickling the torch model gets mp stuck in wait()
            if hasattr(self, "model"):
                warnings.warn(
                    "In python 3.11, pickling a torch model gets multiprocessing \
                              stuck in waiter.acquire(). Defaulting to 1 process!",
                    RuntimeWarning,
                )
                self.n_jobs = 1
                return self.encoding_factory()
            print(f"Encoding {self.output_file_prefix} via multiprocess...")
            return self._encode_multi_process

    @abstractmethod
    def _encode_single_process(
        self,
        sequences: dict[SequenceIdType, npt.NDArray[np.str_]],
        show_progress: bool = True,
    ) -> dict[SequenceIdType, npt.NDArray[np.uint8]]:
        """
        ## Encodes sequences using a single process
        """

    def _encode_multi_process(
        self, sequences: dict[SequenceIdType, npt.NDArray[np.str_]]
    ) -> dict[SequenceIdType, npt.NDArray[np.uint8]]:
        """
        ## Encodes sequences using multiprocessing
        """
        # Split the data into chunks
        split_data = self._split_dict(dictionary=sequences, chunks=self.n_jobs * 20)

        # Progress bar (inner is in single_process)
        outer_progress_bar = tqdm(total=len(split_data))

        all_results: dict[SequenceIdType, npt.NDArray] = {}

        with Pool(processes=self.n_jobs) as pool:
            # Workers do not open their own bars
            processing_function_without_bar = partial(
                self._encode_single_process, show_progress=False
            )
            # Multiprocess
            for chunk_result in pool.imap(processing_function_without_bar, split_data):
                all_results.update(chunk_result)
                outer_progress_bar.update()

        if self.flatten:
            all_results = self._flatten_arrays(data=all_results)
        return all_results

    @staticmethod
    def _flatten_arrays(
        data: dict[SequenceIdType, npt.NDArray],
    ) -> dict[SequenceIdType, npt.NDArray]:
        """
        ## Flattens all the arrays in the provided data
        """
        for sequence_id, encoding in data.items():
            data[sequence_id] = encoding.flatten()
        return data

    @staticmethod
    def _split_dict(dictionary: dict, chunks: int) -> list[dict]:
        """
        ## Splits dictionary into N chunks
        Repeats cycle until all key, value pairs have been allocated.
        """
        iterator = cycle(range(chunks))
        split_dict = [dict() for _ in range(chunks)]
        for key, value in dictionary.items():
            split_dict[next(iterator)][key] = value
        return split_dict


def estimate_data_size(
    data: ArrayDict, *, data_format: SaveFormat, json_options: Optional[int] = None
) -> int:
    """
    ## Estimates the size of encoded data
    Can distinguish between json and numpy formats. Json options allowed.
    ### Args:
        \t-data {ArrayDict} -- Mapping from keys to numpy array \n
        \t-format {SaveFormat} -- Format in which the data is saved \n
            -Options: json, numpy, npy, npz \n
        \t-json_options {int} -- Passed to orjson.dumps
    """
    if data_format == "json":

        if json_options is None:
            json_options = (
                orjson.OPT_SERIALIZE_NUMPY  # pylint: disable=maybe-no-member
                | orjson.OPT_NON_STR_KEYS  # pylint: disable=maybe-no-member
            )

        return len(
            orjson.dumps(data, option=json_options)  # pylint: disable=maybe-no-member
        )

    elif data_format in {"numpy", "npy", "npz"}:

        # Convert keys from tuple to str
        data = {f"({','.join(map(str, key))})": value for key, value in data.items()}

        buffer: io.BytesIO = io.BytesIO()

        if data_format == "npz":
            np.savez(buffer, **data)

        else:
            np.save(buffer, data)

        return buffer.getbuffer().nbytes

    else:
        raise ValueError("Format must be 'json', 'numpy', 'npy' or 'npz'")
