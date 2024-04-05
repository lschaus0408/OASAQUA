"""
---------------------------------- Observed Antibody Space API -------------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics (doi: 10.1002/pro.4205)\n
---------------------------- Remove Redundant Sequences Post Processing ----------------------------\n
Removes all redundant sequences from OAS API files. Can either run on a directory level or on single 
files.\n
"""

from OAS_API.post_processing import PostProcessor
from pathlib import Path
from typing import Literal, Optional
from tqdm import tqdm

import pandas as pd

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


class RemoveRedundant(PostProcessor):
    """
    ## Removes Redundant Sequences
    Removes all redundant sequences from OAS API files. Can either run on a directory level or on single
    files.
    """

    def __init__(
        self,
        directory_or_file: Path,
        mode: Optional[Literal["directory", "file"]] = None,
    ) -> None:
        self.path = directory_or_file
        self.mode = mode
        self.hashes = {}

    def save_file(self, data: pd.DataFrame, filename: Path):
        data.to_csv(path_or_buf=filename)

    def load_file(self, file_path: Path):
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
            if self.path.is_file():
                tqdm.write("Processing a file for redundant sequences...")
                self.mode = "file"
            else:
                tqdm.write("Processing a directory of OAS API files for redundant sequences...")
                self.mode = "directory"

        process_factory[self.mode]()

    def process_file(self, filename: Optional[Path] = None):
        """
        ## Processes single OAS API files
        """
        # Load data, else statement allows for directory processing
        if filename is None:
            data = self.load_file(self.path)
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
            self.save_file(data=data, filename=self.path)
        else:
            self.save_file(data=data, filename=filename)

    def process_directory(self):
        """
        ## Processes directory of OAS API files
        """
        all_files = self.path.glob("**/*")
        all_files = [file for file in all_files if file.is_file()]

        for file in tqdm(all_files):
            self.process_file(filename=file)

if __name__ == "__main__":
    directory = Path("/home/lschaus/vscode/data/OAS_API_Processed/")
    A = RemoveRedundant(directory_or_file=directory)
    A.process()



