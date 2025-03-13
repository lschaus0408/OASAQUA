"""
---------------------------------- Observed Antibody Space API -----------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics(doi: 10.1002/pro.4205)\n
---------------------------------------- Post Processing -----------------------------------------\n
This module contains the post-processing tools for OAS API. Examples: Delete ambiguous sequences,
reconstruct sequences, cluster sequences + sample etc.\n
"""

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
import numpy.typing as npt

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


class PostProcessor(ABC):
    """
    ### OASCS Postprocessor Class
    FINISH DOCTSTRINGS
    """

    def __init__(self, directory_or_file_path: Path, output_directory: Path):
        self.directory_or_file_path = directory_or_file_path
        self.output_directory = output_directory

        self.all_files: list[Path] = []

    @abstractmethod
    def load_file(self, file_path: Path, overwrite: bool = False):
        """
        ## Method to load the files into the post processing tool.
        """

    @abstractmethod
    def save_file(self, file_path: Path, data: Union[pd.DataFrame, npt.ArrayLike]):
        """
        ## Method to save the file.
        """

    @abstractmethod
    def process(self):
        """
        ## Process file according to the processor.
        """

    def get_files_list(self, directory_or_file_path: Path):
        """
        ## Populates all_files list with file paths
        """
        if directory_or_file_path.is_file():
            self.all_files.append(directory_or_file_path)
        else:
            files = directory_or_file_path.glob("**/*")
            self.all_files.extend(files)
