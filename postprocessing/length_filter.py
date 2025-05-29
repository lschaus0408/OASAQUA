"""
---------------------------------- Observed Antibody Space CS -----------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics(doi: 10.1002/pro.4205)\n
---------------------------------------- Length Filter ------------------------------------------\n
Splits the dataset into training, test, and validation sets.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from abnumber import Chain, ChainParseError

from postprocessing.post_processing import PostProcessor, DTYPE_DICT


class LengthFilter(PostProcessor):
    """
    ## LengthFilter for OASCS
    Removes sequences outside of the length specifications of this program.
    Sequences can be filtered if they are larger than N, or smaller than M,
    or outside of N and M.
    """

    def __init__(
        self,
        directory_or_file_path: Path,
        output_directory: Optional[Path] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        max_cdr3_length: Optional[int] = None,
    ):
        super().__init__(
            directory_or_file_path=directory_or_file_path,
            output_directory=output_directory,
        )
        assert (
            max_length is not None or min_length is not None
        ), "One of the two, \
            or both, of max_length and min_length needs to be provided!"

        self.max_length = max_length
        self.min_lenght = min_length
        self.max_cdr3_length = max_cdr3_length

    def load_file(self, file_path: Path, overwrite=False):
        """
        ## Loads file
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

        file_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path_or_buf=file_path)

    def process(self):
        """
        ## Processes the files
        Loads all files iteratively, filters them and then saves them.
        """
        # Load list of files
        self.get_files_list(directory_or_file_path=self.directory_or_file_path)

        for file in self.all_files:
            data = self.load_file(file_path=file)

            # Copy data in order to keep the original file
            filtered_data = data.copy()

            if self.max_length is not None:
                filtered_data = filtered_data[
                    filtered_data["Sequence_aa"].str.len() <= self.max_length
                ]
            if self.min_lenght is not None:
                filtered_data = filtered_data[
                    filtered_data["Sequence_aa"].str.len() >= self.min_lenght
                ]

            if self.max_cdr3_length is not None:
                filtered_data = self._filter_by_cdr3_lenght(data=filtered_data)

            self.save_file(file_path=file, data=filtered_data)

    def _filter_by_cdr3_lenght(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ## Removes sequences with CDR3 lengths that are too long
        """
        if "cdr3_aa" in data.columns:
            data = data[data["cdr3_aa"].str.len() <= self.max_cdr3_length]
            return data

        # Else, extract cd3 information
        sequences = data["Sequence_aa"].to_list()
        cdr3_list = []
        for sequence in sequences:

            # Extract cd3 sequence using abnumber
            try:
                chain = Chain(sequence=sequence, scheme="imgt")
                cdr3_list.append(chain.cdr3_seq)

            # If it's not recognized as an antibody, skip
            except ChainParseError:
                cdr3_list.append("")

        data["cdr3_aa"] = cdr3_list
        # Call again to filter
        self._filter_by_cdr3_lenght(data=data)
