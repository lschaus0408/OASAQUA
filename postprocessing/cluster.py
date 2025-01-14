"""
---------------------------------- Observed Antibody Space CS -----------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics(doi: 10.1002/pro.4205)\n
------------------------------------------- Cluster ----------------------------------------------\n
Clusters Sequences of OAS CS files using Linclust (doi: 10.1038/s41467-018-04964-5). Due to the
possibly very large sequence datasets that one can pull from OAS, Linclust seems to be the best
clustering algorithm as of 2024 for this particular application of creating datasets. It is 
recommended to run the Cluster post-processing after removing redundant sequences,
filtering by length, and checking for sequence viability, because in Linclust the sequence 
representative is always selected as the longest sequence in the cluster. Therefore,
abnormal CDR3 length Ab sequences could be selected as representatives and removed from the data
later, removing all the sequence information of the cluster it represented.
"""

from pathlib import Path

import pandas as pd

from postprocessing.post_processing import PostProcessor


class Cluster(PostProcessor):
    """
    ## Cluster Postprocessor
    Uses Linclust to cluster sequences.
    FINISH DOCSTRINGS
    """

    def __init__(self):
        return

    def load_file(self, file_path: Path, overwrite=False):
        """
        ### Load File
        """
        return

    def save_file(self, file_path: Path, data: pd.DataFrame):
        """
        ### Save File
        """
        return

    def process(self):
        """
        ### Cluster Sequences
        """
        return
