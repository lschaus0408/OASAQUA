"""
---------------------------------- Observed Antibody Space CS -----------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics(doi: 10.1002/pro.4205)\n
------------------------------------------- Cluster ----------------------------------------------\n
Clusters Sequences of OAS CS files using fastBCR (doi: 10.1016/j.crmeth.2023.100601). Due to the
possibly very large sequence datasets that one can pull from OAS, fastBCR seems to be the best
clustering algorithm as of 2024 for clustering antibody sequences by their clonal family. It is
recommended to run the Cluster post-processing after removing redundant sequences,
filtering by length, and checking for sequence viability. Other clustering, such as Linclust can
be implemented upon demand.
"""

from pathlib import Path
from shutil import which
from typing import Optional

import pandas as pd

from postprocessing.post_processing import PostProcessor
from postprocessing.sequence_tracker import SequenceIdType


class Cluster(PostProcessor):
    """
    ## Cluster Postprocessor
    Requires the installation of R and fastBCR by the user.
    Please check out the github for instructions:
    https://github.com/ZhangLabTJU/fastBCR/
    """

    def __init__(
        self,
        directory_or_file_path: Path,
        output_directory: Path,
        path_to_fastbcr: Optional[Path] = None,
    ):
        super().__init__(
            directory_or_file_path=directory_or_file_path,
            output_directory=output_directory,
        )

        # Make sure fastBCR is installed

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
