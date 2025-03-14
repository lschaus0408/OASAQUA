"""
---------------------------------- Observed Antibody Space CS -----------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics(doi: 10.1002/pro.4205)\n
--------------------------------------- Encode via One-Hot ---------------------------------------\n
Performs encoding of antibody sequences in the dataset via One-Hot encoding. A One-Hot encoding
assigns a vector of size 21 to each position, i, in a protein sequence to create a Nx21 matrix.
Each position, j, in the vector corresponds to one of the 20 essential amino-acids + one position
for an empty position. When an amino acid is present at position i, all of the values of the vector
are set to 0 except for j, which takes the value of 1.
"""

import json

from pathlib import Path
from typing import Literal, Optional, Callable
from itertools import cycle
from multiprocessing import Pool
from sys import getsizeof

import pandas as pd
import numpy as np
import numpy.typing as npt

from tqdm import tqdm

from postprocessing.post_processing import PostProcessor, DTYPE_DICT
from postprocessing.sequence_tracker import SequenceIdType


class EncodeOneHot(PostProcessor):
    """
    ## Encoder Using One-Hot
    Simple encoding that converts each amino acid into a size 20 array.
    Array will be 1 at positions that correspond to the amino acid and
    0 at all other positions. Encoding can either be stored as a 20xN
    matrix or a 20N vector.
    Data can be saved in json format or in numpy format.
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
        save_format: Literal["json", "numpy"],
        category_column: Optional[str] = None,
        n_jobs: int = 1,
        maximum_file_size_gb: Optional[int] = None,
        flatten: bool = False,
        output_file_prefix: str = "one_hot_encoded_sequences",
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
            index_col=0,
            dtype=DTYPE_DICT,
        )

    def save_file(self, file_path: Path, data: npt.ArrayLike):
        return

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
            # Remove indices where the sequence is greater than the pad_size
            data["sequence_lenght"] = data["Sequence_aa"].str.len()
            filtered_data = data[data["sequence_lenght"] > self.pad_size]

            # Create SequenceIDType column
            filtered_data.reset_index(drop=True, inplace=True)
            filtered_data["file_id"] = index
            sequence_ids = list[SequenceIdType] = list(
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

            # Determine the size of the dict and save
            json_string = json.dumps(encoded_sequence_dict)
            if (
                self.maximum_file_size is not None
                and getsizeof(json_string) > self.maximum_file_size
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
            return self._encode_single_process
        else:
            return self._encode_multi_process

    def _encode_single_process(
        self, sequences: dict[SequenceIdType, npt.NDArray[np.str_]]
    ) -> dict[SequenceIdType, npt.NDArray[np.uint8]]:
        """
        ## Encodes sequences using a single process
        """
        encoded_sequences: dict[SequenceIdType, npt.NDArray[np.uint8]] = {}
        for sequence_id, sequence in sequences.items():
            # Setup empty array
            encoded = np.zeros(
                shape=(self.pad_size, len(self.ALL_AMINO_ACIDS)), dtype=np.uint8
            )
            sequence_lenght = len(sequence)
            # Pad with the pad token
            if sequence_lenght < self.pad_size:
                difference = self.pad_size - sequence_lenght
                sequence = sequence + "X" * difference

            # Get two position lists
            residue_positions = range(sequence_lenght)
            residue_id_numbers = list(
                map(lambda residue: self.ALL_AMINO_ACIDS[residue], sequence)
            )
            # Advanced indexing to populate array
            encoded[residue_positions, residue_id_numbers] = 1

            encoded_sequences[sequence_id] = encoded

        return encoded_sequences

    def _encode_multi_process(
        self, sequences: dict[SequenceIdType, npt.NDArray[np.str_]]
    ) -> dict[SequenceIdType, npt.NDArray[np.uint8]]:
        """
        ## Encodes sequences using multiprocessing
        """
        # Split the data into chunks
        split_data = self._split_dict(dictionary=sequences, chunks=self.n_jobs * 20)
        len_split_data = [len(item) for item in split_data]

        all_results: dict[SequenceIdType, npt.NDArray] = {}

        with Pool(processes=self.n_jobs) as pool:
            # Multiprocess
            results = list(
                tqdm(
                    pool.imap(self._encode_single_process, split_data),
                    total=len_split_data,
                )
            )

            # Reassemble results
            for encoded_sequences in results:
                all_results.update(encoded_sequences)

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
