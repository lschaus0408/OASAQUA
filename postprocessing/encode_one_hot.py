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

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt

from tqdm import tqdm

from postprocessing.encoder import Encoder
from postprocessing.sequence_tracker import SequenceIdType


class EncodeOneHot(Encoder):
    """
    ## Encoder Using One-Hot
    Simple encoding that converts each amino acid into a size 20 array.
    Array will be 1 at positions that correspond to the amino acid and
    0 at all other positions. Encoding can either be stored as a 20xN
    matrix or a 20N vector.
    Data can be saved in json format or in numpy format.
    """

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
            pad_size=pad_size,
            save_format=save_format,
            category_column=category_column,
            n_jobs=n_jobs,
            maximum_file_size_gb=maximum_file_size_gb,
            flatten=flatten,
            output_file_prefix=output_file_prefix,
        )

    def _encode_single_process(
        self,
        sequences: dict[SequenceIdType, npt.NDArray[np.str_]],
        show_progress: bool = True,
    ) -> dict[SequenceIdType, npt.NDArray[np.uint8]]:
        """
        ## One-Hot Encodes sequences using a single process
        """
        encoded_sequences: dict[SequenceIdType, npt.NDArray[np.uint8]] = {}

        # Setting up tqdm to work with multiprocessing as well
        sequence_iterator = sequences.items()
        if show_progress:
            sequence_iterator = tqdm(sequence_iterator, total=len(sequences))

        for sequence_id, sequence in sequence_iterator:
            # Setup empty array
            encoded = np.zeros(
                shape=(self.pad_size, len(self.ALL_AMINO_ACIDS)), dtype=np.uint8
            )
            sequence_length = len(sequence)
            # Pad with the pad token
            if sequence_length < self.pad_size:
                difference = self.pad_size - sequence_length
                sequence = sequence + "X" * difference

            # Get two position lists
            residue_positions = np.arange(len(sequence))
            residue_id_numbers = [self.ALL_AMINO_ACIDS[residue] for residue in sequence]
            # Advanced indexing to populate array
            encoded[residue_positions, residue_id_numbers] = 1

            encoded_sequences[sequence_id] = encoded

        return encoded_sequences
