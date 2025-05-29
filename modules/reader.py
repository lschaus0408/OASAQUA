"""
---------------------------------- Observed Antibody Space API -------------------------------------
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics (doi: 10.1002/pro.4205)
-------------------------------------- CSV Reader Module -------------------------------------------
This module reads the csv file downloaded from OAS and exctracts the desired data: AA sequence/DNA
sequence from different regions of the antibody.
"""

# Preamb
import json
import sys
from pathlib import Path

sys.path.insert(0, "../mousify")

from typing import List, Optional

import numpy as np
import pandas as pd

from modules.helper_functions import (
    check_header_request,
    check_int_in_range,
    check_keywords,
    check_none_and,
    check_translation,
)


class CSVReader:
    """## CSVReader reads a csv file obtained from OAS.
    Object can separate the header from the rest of the data and extracts the desired information form the file and saves it as a pd.DataFrame.
    This DataFrame can then be saved as a new csv file.
    ### Args:\n
        \tpath {str} -- path location of the csv file to be read\n

    ### Attr:\n
            \tpath {str} -- path location of the csv file to be read \n
            \tdata {dict} -- dict with the information extracted from the csv file\n
            \tdf {pd.DataFrame} -- DataFrame that stores the information from data
                            after everything has been extracted.\n

    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path  # type: Optional[Path]
        self.data = {}
        self.table = None  # type: Optional[pd.DataFrame]

    def __call__(
        self, metadata: Optional[List[str]] = None, data: Optional[List[str]] = None
    ) -> None:
        """## Object call for CSV_Reader.
        Goes through the CSV pipeline as intended. First reads the
        csv located at self.path, then reads the metadata and the desired data from the original
        file and writes it to a pandas DataFrame, stored as self.table. Currently only AA
        sequences are extraced because fetch_data is not yet fully implemented as intended.
        Even though both metadata and data are optional, both cannot be ommited at the same time.\n

        ### Args:\n
                \tmetadata {Optional[List[str]]} -- List of metadata keywords that are supposed to
                                                be extracted. See fetch_metadata for details\n
                \tdata {Optional[List[str]]} -- List of data keywords that are supposed to be
                                                extracted. See fetch_data for details.\n

        ### Returns:
                \tNone\n
        """
        # Check that a path has been passed
        self._check_path()

        # Check that not both metadata and data are None, since that wouldn't do anything
        check_none_and(metadata, data)

        # Read the csv file stored at self.path
        self.read_csv(keywords=data)

        # Store the metadata in self.data
        if metadata is not None:
            self.fetch_metadata(info=metadata)

        # Store the data in self.data
        if data is not None:
            for keyword in data:
                self.fetch_data(keyword, 1)

        self.to_dataframe()

    def set_path(self, path: Path) -> None:
        """
        ## Setter for the path attribute
        """
        self.path = path

    def _check_path(self) -> None:
        """
        ## Asserts that self.path is not None.
        """
        if self.path is None:
            raise AssertionError(
                f"self.path needs to be set to a path with the location of the file to be read. Currently, self.path is set to {self.path}!"
            )
        else:
            assert (
                self.path.is_file()
            ), f"self.path needs to point towards a file. Currently, self.path is set to {self.path}"

    def read_csv(self, keywords: list[str], no_header: bool = True) -> None:
        """## Method that reads the csv files from OAS and turns them into a pandas dataframe for cleaning.
        It is recommended to keep no_header as True, as it can mess with downstream
        processes. Header information can be extracted using the extract_header_info method.

        ### Args:\n
                \tpath {str} -- Path from the working directory to the csv file.\n
                \tno_header {bool} -- Optional argument to keep the original header of the
                                        OAS csv files. Recommended to keep this as True for routine use.\n

        ### Updates:\n
                \tself.table -- Makes table a pandas dataframe that can be cleaned downstream
        """

        # Check if header should be kept. value = 1 removes the header.
        if no_header:
            value = 1
        else:
            value = 0

        dtype_dict = {
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
        }

        columns = [
            item
            for item in keywords
            if item != "full" and item != "fwr" and item != "cdr" and item != "junction"
        ]
        columns.extend(
            [
                "fwr1",
                "fwr1_aa",
                "cdr1",
                "cdr1_aa",
                "fwr2",
                "fwr2_aa",
                "cdr2",
                "cdr2_aa",
                "fwr3",
                "fwr3_aa",
                "cdr3",
                "cdr3_aa",
                "fwr4",
                "fwr4_aa",
                "junction",
                "junction_aa",
                "np1",
                "np2",
            ]
        )
        # Return the pandas DataFrame
        self.table = pd.read_csv(
            self.path, header=value, dtype=dtype_dict, usecols=columns
        )

    def fetch_metadata(self, info: List[str]) -> None:
        """## Method that extracts the desired header info and writes to the object dictionary.

        ### Args:\n
            \tinfo {List[str]} -- List of string arguments to be extracted from the header.\n
                                    \tPossible arguments: "Run",\n
                                                        "Link",\n
                                                        "Author",\n
                                                        "Species",\n
                                                        "BSource",\n
                                                        "BType",\n
                                                        "Age",\n
                                                        "Longitudinal",\n
                                                        "Disease",\n
                                                        "Vaccine",\n
                                                        "Subject",\n
                                                        "Chain",\n
                                                        "Unique sequences",\n
                                                        "Isotype",\n
                                                        "Total sequences"\n
        ### Returns:
            \tNone
        """

        possibilities = "Run, Link, Author, Species, BSource, BType, Age, Longitudinal, \
                        Disease, Vaccine, Subject, Chain, Unique sequences, Isotype, \
                        Total sequences"

        # Context manager to open the file
        with open(self.path, "r", encoding="UTF-8") as file:  # type: ignore
            # Only exract first line
            header = file.readline()
            # Go through search terms
            for term in info:
                # Using RE we can check if the exact search term is present in possibilities
                assert check_header_request(
                    possibilities, term
                ), f"Search term {term} not found in header"

                # Find the term in the header and extract the info.
                term_len = len(term) + 8
                index = header.find(f'""{term}"": ""')
                end = header[index:].find('"", ""')
                if end == -1:
                    end = header[index:].find("}") - 2
                self.data[term] = header[index + term_len : end + index]

    def fetch_data(
        self,
        keyword: str = "full",
        translate: int = 0,
        path_allowed_sequences: str = "./OAS_API/files/csvreader_sequence_set.json",
    ) -> None:
        """## Takes the original dataframe as input and writes the desired data to the object dictionary.

        ### Args:
                \tkeyword {str} --  \tKeyword that defined what sequence should be fetched.\n
                                    \tOptions (Default is 'full'):\n
                                        \t- 'full': Entire Ab sequence\n
                                        \t- 'fwr': Framework Region\n
                                        \t- 'cdr': CDR Region\n
                                        \t- 'junction': Junction Region\n
                                        \t- 'np': NP region\n
                                        \t- 'c_region': C region\n
                                        \t- 'sequence_alignment':   Sequence alignment of all
                                                                sequences in the dataset\n
                                        \t- 'germline_alignment':   Germline sequence alignment
                                                                of all sequences in the dataset\n
                                        \t- '[v,d,j]_sequence_alignment':   Sequence alignment
                                                                        of all the v-,d- or
                                                                        j_sequences in the
                                                                        dataset\n
                                        \t- '[v,d,j]_germline_alignment':   Germline sequence
                                                                        alignment of all
                                                                        the v-,d- or
                                                                        j_sequences in
                                                                        the dataset\n
                                        \t- 'locus':            Locus of the Ab gene (Heavy,
                                                            Lambda or Kappa)\n
                                        \t- '[v,d,j]_call':     Call ID of the v-,d- or j_sequences\n
                                        \t- '[v,d,j]_score':    Score of the v-, d- or j_sequences\n
                                        \t- '[v,d,j]_support':  Support score of the v-, d- or
                                                            j_sequences\n
                                        \t- '[v,d,j]_identity': Identity % for the v-, d- or
                                                            j_sequences\n


                \ttranslate {int} -- \tTranslates the DNA sequence into an AA sequence,
                                    depending on the input the output changes
                                    (Default is 0):\n
                                            \t- 0: Only DNA sequence\n
                                            \t- 1: Only AA sequence\n
                                            \t- 2: Both DNA and AA sequence\n
        ### Returns:
                \tNone
        """
        # Define sequence keyword set (I haven't found a more elegant solution yet)
        with open(path_allowed_sequences, "r") as infile:
            sequence_set = json.load(infile)

        # Make sure keywords are allowed
        check_keywords(keyword)

        # Make sure translate is within the correct range
        check_int_in_range(translate)

        # Make sure translate is not on for score fetches
        check_translation(keyword, translate)

        # If the entire sequence is wanted, NOT IMPLEMENTED
        if keyword == "full":
            self.data.update(self._fetch_ab_sequence(translate=translate))
        else:
            # Need to fix this to be able to distinguish between sequences and call IDs
            if keyword in sequence_set:
                # Provide only the AA sequences
                if translate == 1:
                    # Regular expression operators
                    self.data.update(
                        self.table.filter(  # type: ignore
                            regex=f"{keyword}([0-9]+)?_aa(?!_length)"
                        ).to_dict("list")
                    )

                # Provide the DNA and AA sequences
                elif translate == 2:
                    # Regular expression operators
                    self.data.update(
                        self.table.filter(  # type: ignore
                            regex=f"^{keyword}([0-9]+)?(_aa)?(?!_length)$"
                        ).to_dict("list")
                    )

                # Provide only DNA sequences
                else:
                    # Regular expression operators
                    self.data.update(
                        self.table.filter(  # type: ignore
                            regex=f"^{keyword}([0-9]+)?(?!_length)$"
                        ).to_dict("list")
                    )

            else:
                self.data.update({keyword: self.table[keyword].to_list()})  # type: ignore

    def _fetch_ab_sequence(self, translate: int) -> dict:
        """
        ## Adds the fwr, cdr and junction sequences together into one sequence.
        Depending on the value of translate, it returns the amino acid sequence, DNA sequence, or both.

        ### Args:
            \ttranslate {int} -- Defines if the function returns the AA sequence, DNA sequence or both.\n
                                \t Possible values:\n
                                \t- 0: Only DNA Sequence\n
                                \t- 1: Only AA Sequence\n
                                \t- 2: Both\n
        ### Returns:
            \tdict -- Dictionary containing the keyword 'sequence' and/or 'sequence_aa' together with the sequence
        """
        # Better do the check again in case someone uses this function
        check_int_in_range(translate)
        # Keywords in order of combination
        keys_dna = ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4", "junction"]
        keys_aa = [key + "_aa" for key in keys_dna]

        if translate == 1:
            keywords = keys_aa
            out_key = "Sequence_aa"
        elif translate == 0:
            keywords = keys_dna
            out_key = "Sequence_dna"
        else:
            seq_aa = self._fetch_ab_sequence(translate=1)
            seq_dna = self._fetch_ab_sequence(translate=0)
            # The following expression results in the merger of the dictionaries
            return seq_aa | seq_dna

        sequences = self.table.filter(items=keywords)
        # New pandas version does not keep order of keywords so it needs to be reordered
        sequences = sequences[keywords].values  # type: ignore
        out_list = []

        for subsequence in sequences:
            # Convert NaN to empty string
            subsequence = ["" if item is np.nan else item for item in subsequence]
            # Map converts the list into type str
            sequence = "".join(map(str, subsequence))
            out_list.append(sequence)

        return {out_key: out_list}

    def to_dataframe(self) -> None:
        """## Converts the dictionary structure of self.data into a dataframe.
        Overwrites self.table.
        """
        self.table = pd.DataFrame(data=self.data)

    def to_csv(self, path: str) -> None:
        """## Writes the data extracted from the original file to disk in form of a csv file.
        ### Args:\n
                \tpath {Optional[str]} -- Path where the data should be saved. If no path is
                                            passed as an argument, the path from which the original
                                            csv file was stored is used.
        ### Returns:\n
            \tNone
        """

        self.table.to_csv(path)  # type: ignore


if __name__ == "__main__":
    print("Ran csvreader as main. This does nothing right now.")
