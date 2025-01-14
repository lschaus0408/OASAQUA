"""
---------------------------------- Observed Antibody Space CS -----------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics(doi: 10.1002/pro.4205)\n
------------------------------------------- Data Maker ------------------------------------------\n
Splits the dataset into training, test, and validation sets.
"""

from pathlib import Path

import pandas as pd

from postprocessing.post_processing import PostProcessor


class DataMaker(PostProcessor):
    """
    ## DataMaker for OASCS
    Splits a dataset into training, test, and validation sets.
    FINISH DOCSTRINGS
    """

    def __init__(self):
        return

    def load_file(self, file_path: Path, overwrite=False):
        return

    def save_file(self, file_path: Path, data: pd.DataFrame):
        return

    def process(self):
        return
