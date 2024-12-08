"""
---------------------------------- Observed Antibody Space API -------------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics (doi: 10.1002/pro.4205)\n
---------------------------------- Combine Files Post Processing -----------------------------------\n
This module is used to combine files from OAS API to all have a similar size. This is important to 
be able to reliably use the antibody viability module. If the files for that module are too small,
the sample to figure out the mutation rate is too small making filtering noisy. If the file is too
big, antibody viability might crash due to lacking memory to process everything. If the files are 
all approximately the same size, the antibody viability module can always be run with the same number
of CPUs and batch sizes. This can also be used to package data into desired data chunks for model training.\n
"""
from OAS_API.post_processing import PostProcessor
from pathlib import Path
from tqdm import tqdm

import math
import warnings

import pandas as pd
import numpy as np

DTYPE_DICT = {
    "sequence": "string",
    "locus": "category",
    "stop_codon": "category",
    "vj_in_frame": "category",
    "v_frameshift": "category",
    "productive": "category",
    "rev_comp": "category",
    "complete_vdj": "category",
    "v_call": "category",
    "d_call": "category",
    "j_call": "category",
    "sequence_alignment": "string",
    "germline_alignment": "string",
    "sequence_alignment_aa": "string",
    "germline_alignment_aa": "string",
    "v_alignment_start": "Int16",
    "v_alignment_end": "Int16",
    "d_alignment_start": "Int16",
    "d_alignment_end": "Int16",
    "j_alignment_start": "Int16",
    "j_alignment_end": "Int16",
    "v_sequence_alignment": "string",
    "v_sequence_alignment_aa": "string",
    "v_germline_alignment": "string",
    "v_germline_alignment_aa": "string",
    "d_sequence_alignment": "string",
    "d_sequence_alignment_aa": "string",
    "d_germline_alignment": "string",
    "d_germline_alignment_aa": "string",
    "j_sequence_alignment": "string",
    "j_sequence_alignment_aa": "string",
    "j_germline_alignment": "string",
    "j_germline_alignment_aa": "string",
    "fwr1": "string",
    "fwr1_aa": "string",
    "fwr2": "string",
    "fwr2_aa": "string",
    "fwr3": "string",
    "fwr3_aa": "string",
    "fwr4": "string",
    "fwr4_aa": "string",
    "cdr1": "string",
    "cdr1_aa": "string",
    "cdr2": "string",
    "cdr2_aa": "string",
    "cdr3": "string",
    "cdr3_aa": "string",
    "junction": "string",
    "junction_aa": "string",
    "junction_length": "Int16",
    "junction_aa_length": "Int16",
    "v_score": "Float32",
    "d_score": "Float32",
    "j_score": "Float32",
    "v_cigar": "category",
    "d_cigar": "category",
    "j_cigar": "category",
    "v_support": "Float32",
    "d_support": "Float32",
    "j_support": "Float32",
    "v_identity": "Float32",
    "d_identity": "Float32",
    "j_identity": "Float32",
    "v_sequence_start": "Int16",
    "v_sequence_end": "Int16",
    "d_sequence_start": "Int16",
    "d_sequence_end": "Int16",
    "j_sequence_start": "Int16",
    "j_sequence_end": "Int16",
    "v_alignment_start": "Int16",
    "v_alignment_end": "Int16",
    "d_alignment_start": "Int16",
    "d_alignment_end": "Int16",
    "j_alignment_start": "Int16",
    "j_alignment_end": "Float32",
    "fwr1_start": "Float32",
    "fwr1_end": "Float32",
    "fwr2_start": "Int16",
    "fwr2_end": "Int16",
    "fwr3_start": "Int16",
    "fwr3_end": "Int16",
    "fwr4_start": "Int16",
    "fwr4_end": "Int16",
    "cdr1_start": "Int16",
    "cdr1_end": "Int16",
    "cdr2_start": "Int16",
    "cdr2_end": "Int16",
    "cdr3_start": "Int16",
    "cdr3_end": "Int16",
    "np1": "string",
    "np1_length": "Int16",
    "np2": "string",
    "np2_length": "Int16",
    "c_region": "string",
    "Redundancy": "category",
    "ANARCI_numbering": "string",
    "ANARCI_status": "string",
    "Run": "category",
    "Link": "category",
    "Author": "category",
    "Species": "category",
    "BSource": "category",
    "BType": "category",
    "Age": "category",
    "Longitudinal": "category",
    "Vaccine": "category",
    "Disease": "category",
    "Subject": "category",
    "Chain": "category",
    "Unique sequences": "Int16",
    "Isotype": "category",
    "Total sequences": "Int16",
    "Sequence_aa": "string",
    "Sequence_dna": "string",
}

class CombineFiles(PostProcessor):
    """
    ## Combine OAS API Files
    This module is used to combine files from OAS API to all have a similar size. This is important to
    be able to reliably use the antibody viability module. If the files for that module are too small,
    the sample to figure out the mutation rate is too small making filtering noisy. If the file is too
    big, antibody viability might crash due to lacking memory to process everything. If the files are
    all approximately the same size, the antibody viability module can always be run with the same number
    of CPUs and batch sizes.
    ### Args:
        \t directory {Path} -- Path to the directory to be processsed
        \t output_filename_prefix {str} -- Desired name for the output files
        \t file_size {int} -- Upper limit for file size in KB
    """

    def __init__(
        self,
        directory: Path,
        output_filename_prefix: str = "OAS_Combined_Files",
        desired_file_size: int = 150_000,
    ) -> None:
        self.directory = directory
        self.desired_file_size = desired_file_size
        self.output_filename = output_filename_prefix
        self.too_small = {}
        self.too_big = {}
        self.split_data = []
        self.split_trigger = 0
        self._fileindex = 0

    def save_file(self, data: pd.DataFrame):
        data.to_csv(path_or_buf=self._get_file_name())

    def load_file(self, file_path: Path) -> pd.DataFrame:
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
        all_files = self.directory.glob("**/*")
        all_files = [file for file in all_files if file.is_file()]

        # Sort files
        tqdm.write("Sorting Files...")
        self.sort_files(list_of_files=all_files)

        # Process big files
        tqdm.write("Processing Large Files...")
        self.process_big_files()

        # Process small files
        tqdm.write("Processing Small Files...")
        self.process_small_files()

        # Clean up
        tqdm.write("Cleaning Up Old Files...")
        for file in tqdm(all_files):
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
                self.save_file(chunk)

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
                self.save_file(merged_data)
                small_file_list = []
                small_file_sizes = 0

        # Clean up whatever is left in the file list
        if small_file_list:
            merged_data = pd.concat(small_file_list)
            self.save_file(merged_data)

    def process_split_data(self):
        """
        ## Processes data currently in self.split_data
        """
        merged_data = pd.concat(self.split_data)
        self.save_file(merged_data)
        self.split_data = []

    def sort_files(self, list_of_files: list[Path]):
        """
        ## Sorts Files into "too big" or "too small"
        """
        for file in list_of_files:
            size = file.stat().st_size / 1024
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
        return self.directory / filename


if __name__ == "__main__":
    directory = Path("/home/lschaus/vscode/data/OAS_API_Processed/")
    A = CombineFiles(
        directory=directory, output_filename_prefix="OAS_Combined_Files_3"
    )
    A.process()