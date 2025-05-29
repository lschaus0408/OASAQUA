import gzip
import json
import os
import re
import shutil
from typing import List, Optional
from pathlib import Path

from Bio.Seq import Seq


def check_header_request(text: str, request: str, ignore_case: bool = True) -> bool:
    """## Checks if the requested term to extract from the header is supported by OAS.

    ### Args: \n
        \ttext {str} -- Text of possible keywords. \n
        \trequest {str} -- Text of requested keyword info. \n
        \tignore_case {bool} -- Restrict the search to case sensitive terms. \n
    ### Returns: \n
        \tbool -- True if the requested term is in the provided text.
    """
    if ignore_case:
        # re.search returns a match object (Truthy) if match is found and None if no match is found
        return re.search(r"\b{}\b".format(request), text, re.IGNORECASE) is not None
    else:
        return re.search(r"\b{}\b".format(request), text) is not None


def translate_dna_rna(sequence: str, codon_table: int) -> str:
    """## Translates the provided DNA or RNA sequence to the corresponding AA sequence.
    Translation is performed according to the codon table provided.

    ### Args: \n
        \tsequence {str} -- Sequence that is to be translated.
        The sequence needs to be an RNA or DNA sequence to be valid. \n
        \tcodon_table {int} -- Choice of codon table to be used.
        Follows the NCBI codon table numbers, \n
                \twhere 1 is the standard table. \n
                \t(https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi) \n
                \tOne can also pass a CodonTable object from biopython if a custom table is needed.

    ### Returns: \n
        \tstr -- A string of the AA sequence.
    """

    # Create a Seq object from Biopython
    to_translate = Seq(sequence)

    # Return translated sequence as string
    return str(to_translate.translate(table=str(codon_table)))


def check_keywords(keyword: str, custom_keywords: Optional[List[str]] = None) -> None:
    """## Checks if the keywords are in the allowed list of keywords.
    ### Args: \n
        \tkeyword {str} -- Keyword passed from fetch_data that
        determines what column to extract.

    ### Returns:
        \tNone
    """

    keyword_list = [
        "full",
        "fwr",
        "cdr",
        "junction",
        "np",
        "c_region",
        "sequence_alignment",
        "germline_alignment",
        "v_sequence_alignment",
        "d_sequence_alignment",
        "j_sequence_alignment",
        "v_germline_alignment",
        "d_germline_alignment",
        "j_germline_alignment",
        "locus",
        "v_call",
        "d_call",
        "j_call",
        "v_support",
        "d_support",
        "j_support",
        "v_identity",
        "d_identity",
        "j_identity",
        "locus",
        "stop_codon",
        "vj_in_frame",
        "v_frameshift",
        "productive",
        "rev_comp",
        "complete_vdj",
        "v_alignment_start",
        "d_alignment_start",
        "j_alignment_start",
        "v_alignment_end",
        "d_alignment_end",
        "j_alignment_end",
        "junction_length",
        "junction_aa_length",
        "v_score",
        "d_score",
        "j_score",
        "v_cigar",
        "d_cigar",
        "j_cigar",
        "v_support",
        "d_support",
        "j_support",
        "v_identity",
        "d_identity",
        "j_identity",
        "v_seq_start",
        "d_seq_start",
        "j_sequence_start",
        "v_sequence_end",
        "d_sequence_end",
        "j_sequence_end",
        "v_germline_start",
        "v_germline_end",
        "d_germline_start",
        "d_germline_end",
        "j_germline_start",
        "j_germline_end",
        "fwr1_start",
        "fwr1_end",
        "fwr2_start",
        "fwr2_end",
        "fwr3_start",
        "fwr3_end",
        "fwr4_start",
        "fwr4_end",
        "cdr1_start",
        "cdr1_end",
        "cdr2_start",
        "cdr2_end",
        "cdr3_start",
        "cdr3_end",
        "cdr4_start",
        "cdr4_end",
        "np1_length",
        "np2_length",
        "Redundancy",
        "ANARCI_numbering",
        "ANARCI_status",
    ]

    if custom_keywords is not None:
        keyword_list = custom_keywords

    assert (
        keyword in keyword_list
    ), f"Keyword: {keyword} not recognized. Use either of {*keyword_list,}."


def check_translation(keyword: str, translate: int) -> None:
    """## Checks if translate was used in the correct situation.
    ### Args: \n
        \ttranslate {int} -- Number id gotten from fetch_data that determines the translation.

    ### Returns: \n
        \tNone
    """

    if translate in [1, 2]:
        # assert "call" not in keyword, f"Keyword: {keyword} not translatable."
        assert "support" not in keyword, f"Keyword: {keyword} not translatable."
        assert "identity" not in keyword, f"Keyword: {keyword} not translatable."
        assert "np" not in keyword, f"Keyword: {keyword} not translatable."
        assert "c_region" not in keyword, f"Keyword: {keyword} not translatable."


def check_int_in_range(value: int, lst: Optional[List[int]] = None) -> None:
    """## Check if the provided integer is in the list of allowed values.
    ### Args:
        \tvalue {int} -- Value to be checked. \n
        \tlst {List[int]} -- List of integers to be checked against. Default is [0,1,2].

    ### Returns:
        \tNone
    """
    if lst is None:
        lst = [0, 1, 2]

    assert value in lst, f"{value} is not an element of [0,1,2]"


def check_none_and(
    metadata: Optional[List[str]] = None, data: Optional[List[str]] = None
) -> None:
    """If both metadata and data are None, raise an AssertionError."""

    if metadata is None and data is None:
        raise ValueError("Both metadata and data cannot be None at the same time.")


def flatten(lst: list) -> list:
    """
    Returns the flattened list.
    """
    return [item for sublist in lst for item in sublist]


def gunzip(gz_file_name: str, work_dir: str) -> None:  # pragma: no cover
    """
    ## Unzips a .gz file.
    By default shutil is not able to do that.
    Uses gzip library to open the file, then open the file within the archive
    ### Args:
        \tgz_file_name {str} -- Name of the file to be extracted. \n
        \twork_dir {str} -- Name of the directory to be extracted to.
    """
    # Extracts the tail of gz_file_name in case the entire path is provided
    file_name = os.path.split(gz_file_name)[-1]
    # Extract the file_name without '.gz' so we can use the rest as the out-file name
    file_name = re.sub(r"\.gz$", "", file_name, flags=re.IGNORECASE)

    # Open the archive file with gzip
    with gzip.open(gz_file_name, "rb") as in_file:
        # Create the file using the directory name and the file name, set as the out-file
        with open(os.path.join(work_dir, file_name), "wb") as out_file:
            # Copy the file from in-file to the out-file
            shutil.copyfileobj(in_file, out_file)


def check_query(query: tuple[tuple[str, str]], database: str) -> None:
    """
    ## Checks if the query provided is a valid one.
    ### Args:
                \t query {tuple[str,str]} -- Query of categories and keys
    """
    current_file_path = Path(__file__).resolve()
    main_directory = current_file_path.parent.parent
    query_file_path = main_directory / "files" / "query_check_dictionary.json"
    with open(query_file_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    query_dict = data[database]
    for pair in query:
        try:
            assert (
                pair[1] in query_dict[pair[0]]
            ), f"{pair[1]} not found as value in query_dict."
        except IndexError:
            assert pair[0] in query_dict, f"{pair[0]} not found as key in query_dict"
            continue
