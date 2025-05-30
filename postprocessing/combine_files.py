"""
---------------------------------- Observed Antibody Space API -----------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics(doi: 10.1002/pro.4205)\n
---------------------------------- Combine Files Post Processing ---------------------------------\n
This module is used to combine files from OAS API to all have a similar size. This is important to
be able to reliably use the antibody viability module. If the files for that module are too small,
the sample to figure out the mutation rate is too small making filtering noisy. If the file is too
big, antibody viability might crash due to lacking memory to process everything. If the files
are all approximately the same size, the antibody viability module can always be run with
the same number of CPUs and batch sizes.
This can also be used to package data into desired data chunks for model training.\n
"""

import math
import warnings

from pathlib import Path

from tqdm import tqdm
import pandas as pd
import numpy as np

from postprocessing.post_processing import PostProcessor, DTYPE_DICT


class CombineFiles(PostProcessor):
    """
    ## Combine OAS API Files
    This module is used to combine files from OAS API to all have a similar size. This is important
    to be able to reliably use the antibody viability module.
    If the files for that module are too small, the sample to figure out the mutation rate is too
    small making filtering noisy. If the file is too big, antibody viability might crash due to
    lacking memory to process everything. If the files are all approximately the same size,
    the antibody viability module can always be run with the same number of CPUs and batch sizes.
    ### Args:
        \t directory {Path} -- Path to the directory to be processsed
        \t output_filename_prefix {str} -- Desired name for the output files
        \t file_size {int} -- Upper limit for file size in GB
    """

    def __init__(
        self,
        directory_or_file_path: Path,
        output_directory: Path,
        output_filename_prefix: str = "OAS_Combined_Files",
        desired_file_size_gb: int = 1.5,
    ) -> None:
        super().__init__(
            directory_or_file_path=directory_or_file_path,
            output_directory=output_directory,
        )
        self.desired_file_size = desired_file_size_gb
        self.output_filename = output_filename_prefix
        self.too_small = {}
        self.too_big = {}
        self.split_data = []
        self.split_trigger = 0
        self._fileindex = 0

    def save_file(self, file_path: Path, data: pd.DataFrame):
        data.reset_index(drop=True, inplace=True)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path_or_buf=file_path)

    def load_file(self, file_path: Path, overwrite: bool = False) -> pd.DataFrame:
        """
        ## Returns file at file_path as a dataframe
        """
        return pd.read_csv(filepath_or_buffer=file_path, index_col=0, dtype=DTYPE_DICT)

    def process(self):
        """
        ## Processes directory
        Checks file sizes in the directory and sorts them into a too large category or a
        too small category. The one's in the too small category are combined up to the
        desired file size. The one's in the too large category are split into smaller files.
        """
        # Grab all files inside directory
        self.get_files_list(directory_or_file_path=self.directory_or_file_path)

        # Sort files
        tqdm.write("Sorting Files...")
        self.sort_files(list_of_files=self.all_files)

        # Process big files
        tqdm.write("Processing Large Files...")
        self.process_big_files()

        # Process small files
        tqdm.write("Processing Small Files...")
        self.process_small_files()

        # Clean up
        tqdm.write("Cleaning Up Old Files...")
        for file in tqdm(self.all_files):
            file.unlink()

    def process_big_files(self):
        """
        ## Processes files that are too big
        """
        for big_file_key in tqdm(self.too_big.keys()):
            file_size = self.too_big[big_file_key]
            split_ratio = file_size / self.desired_file_size
            split_floor = math.floor(split_ratio)
            # What fraction to set apart before splitting into chunks of split_floor
            split_frac = 1 - (split_floor / split_ratio)

            # Load file
            data = self.load_file(file_path=big_file_key)  # type: pd.DataFrame
            first_split_index = math.floor(len(data) * split_frac)

            # Store small data temporarily
            data_small = data.iloc[:first_split_index, :]
            self.split_data.append(data_small)
            self.split_trigger += file_size * split_frac

            # If the split_trigger is larger than file_size, merge the files
            if self.split_trigger > file_size:
                self.process_split_data()

            # Save the rest of the data
            other_data = data.iloc[first_split_index:, :]
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                chunked_large_data = np.array_split(other_data, split_floor)
            for chunk in chunked_large_data:
                current_file_path = self._get_file_name()
                self.save_file(current_file_path, chunk)

        # Save whatever is left over in split data
        if self.split_data:
            self.process_split_data()

    def process_small_files(self):
        """
        ## Processes files that are too small
        """
        small_file_sizes = 0
        small_file_list = []
        # Iterate through small files
        for small_file_key in tqdm(self.too_small.keys()):
            # Keep track of file size if the files were merged
            small_file_sizes += self.too_small[small_file_key]
            # Temporarily store small files
            small_file_list.append(self.load_file(file_path=small_file_key))

            # If the files merged would be big enough, then merge the files and save them
            if small_file_sizes > self.desired_file_size:
                with warnings.catch_warnings():
                    warnings.simplefilter(action="ignore", category=FutureWarning)
                    merged_data = pd.concat(small_file_list)
                current_file_path = self._get_file_name()
                self.save_file(current_file_path, merged_data)
                small_file_list = []
                small_file_sizes = 0

        # Clean up whatever is left in the file list
        if small_file_list:
            merged_data = pd.concat(small_file_list)
            current_file_path = self._get_file_name()
            self.save_file(current_file_path, merged_data)

    def process_split_data(self):
        """
        ## Processes data currently in self.split_data
        """
        merged_data = pd.concat(self.split_data)
        current_file_path = self._get_file_name()
        self.save_file(current_file_path, merged_data)
        self.split_data = []

    def sort_files(self, list_of_files: list[Path]):
        """
        ## Sorts Files into "too big" or "too small"
        """
        for file in list_of_files:
            size = file.stat().st_size / (1024**3)
            if size > self.desired_file_size:
                self.too_big[file] = size
            else:
                self.too_small[file] = size

    def _get_file_name(self) -> str:
        """
        ## Returns the next file name
        """
        filename = self.output_filename + "_" + str(self._fileindex).zfill(5) + ".csv"
        self._fileindex += 1
        return self.output_directory / filename


if __name__ == "__main__":
    directory_test = Path("/home/lschaus/vscode/data/OAS_API_Processed/")
    A = CombineFiles(
        directory_or_file_path=directory_test,
        output_directory=directory_test,
        output_filename_prefix="OAS_Combined_Files_3",
    )
    A.process()
