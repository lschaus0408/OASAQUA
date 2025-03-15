"""
---------------------------------- Observed Antibody Space CS -----------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics(doi: 10.1002/pro.4205)\n
---------------------------------- Non-Canonical Characters -------------------------------------\n
Removes sequences with non-canonical amino acid characters.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from postprocessing.post_processing import PostProcessor, DTYPE_DICT
from postprocessing.encoder import Encoder


class NCCharacters(PostProcessor):
    """
    ## Non-Canonical Character Filter for OASCS
    Removes sequences containing non-canonical characters.
    Users can provide a list of characters that are excepted from
    the non-canonical characters list.
    """

    CANONICAL_CHARACTERS = list(Encoder.ALL_AMINO_ACIDS.keys())

    def __init__(
        self,
        directory_or_file_path: Path,
        output_directory: Path,
        exceptions: Optional[list[str]] = None,
    ):
        super().__init__(
            directory_or_file_path=directory_or_file_path,
            output_directory=output_directory,
        )
        if exceptions:
            self.CANONICAL_CHARACTERS.extend(exceptions)

    def load_file(self, file_path: Path, overwrite=False):
        """
        ## Loads File
        """
        return pd.read_csv(
            filepath_or_buffer=file_path,
            index_col=0,
            dtype=DTYPE_DICT,
        )

    def save_file(self, file_path: Path, data: pd.DataFrame):
        """
        ## Saves file
        """
        # If an output directory is provided, it will save the file there
        if self.output_directory is not None:
            file_name = file_path.name
            file_path = Path(self.output_directory, file_name)

        data.reset_index(drop=True, inplace=True)
        data.to_csv(path_or_buf=file_path)

    def process(self):
        # Loads all files
        self.get_files_list(directory_or_file_path=self.directory_or_file_path)

        for file in self.all_files:
            data = self.load_file(file_path=file)
            filtered_data = data[
                data["Sequence_aa"].apply(self.contains_canonical_characters)
            ]

            self.save_file(file_path=file, data=filtered_data)

    def contains_canonical_characters(self, sequence: str) -> bool:
        """
        ## Returns False if the sequence contains non-canonical characters
        """
        return all(residue in self.CANONICAL_CHARACTERS for residue in sequence)
