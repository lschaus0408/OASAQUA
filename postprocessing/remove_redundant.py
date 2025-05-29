"""
---------------------------------- Observed Antibody Space API -----------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics(doi: 10.1002/pro.4205)\n
---------------------------- Remove Redundant Sequences Post Processing --------------------------\n
Removes all redundant sequences from OAS API files. Can either run on a directory level or on single
files.\n
"""

from pathlib import Path
from typing import Literal, Optional

from tqdm import tqdm
import pandas as pd

from postprocessing.post_processing import PostProcessor, DTYPE_DICT


class RemoveRedundant(PostProcessor):
    """
    ## Removes Redundant Sequences
    Removes all redundant sequences from OAS API files.
    Can either run on a directory level or on single files.
    """

    def __init__(
        self,
        directory_or_file_path: Path,
        mode: Optional[Literal["directory", "file"]] = None,
    ) -> None:
        super().__init__(
            directory_or_file_path=directory_or_file_path, output_directory=""
        )
        self.mode = mode
        self.hashes = {}

    def save_file(self, file_path: Path, data: pd.DataFrame):
        """
        ## Saves the file
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path_or_buf=file_path)

    def load_file(self, file_path: Path, overwrite: bool = False):
        """
        ## Returns file at file_path as a dataframe
        """
        return pd.read_csv(
            filepath_or_buffer=file_path,
            index_col=0,
            dtype=DTYPE_DICT,
        )

    def process(self):
        """
        ## Processes directory or files
        """
        process_factory = {
            "file": self.process_file,
            "directory": self.process_directory,
        }
        # Figure out the processing mode
        if self.mode is None:
            if self.directory_or_file_path.is_file():
                tqdm.write("Processing a file for redundant sequences...")
                self.mode = "file"
            else:
                tqdm.write(
                    "Processing a directory of OAS API files for redundant sequences..."
                )
                self.mode = "directory"

        process_factory[self.mode]()

    def process_file(self, filename: Optional[Path] = None):
        """
        ## Processes single OAS API files
        """
        # Load data, else statement allows for directory processing
        if filename is None:
            data = self.load_file(self.directory_or_file_path)
            data.reset_index(inplace=True)
        else:
            data = self.load_file(filename)
            data.reset_index(inplace=True)

        # Register indices that need to be dropped
        drop_indexes = []
        for iterator, sequence in enumerate(data["Sequence_aa"]):
            hashed_sequence = hash(sequence)
            if hashed_sequence in self.hashes:
                drop_indexes.append(iterator)
            else:
                self.hashes[hashed_sequence] = 1

        data = data.drop(drop_indexes)
        if filename is None:
            self.save_file(data=data, file_path=self.directory_or_file_path)
        else:
            self.save_file(data=data, file_path=filename)

    def process_directory(self):
        """
        ## Processes directory of OAS API files
        """
        all_files = self.directory_or_file_path.glob("**/*")
        all_files = [file for file in all_files if file.is_file()]

        for file in tqdm(all_files):
            self.process_file(filename=file)


if __name__ == "__main__":
    print("This is the remove redundant post processor!")
