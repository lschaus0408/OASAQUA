"""
---------------------------------- Observed Antibody Space CS -----------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics(doi: 10.1002/pro.4205)\n
--------------------------------------- Encode via ESM ------------------------------------------\n
Encodes antibody sequences with the Protein Language Model 'Evolutionary Scale Modelling 2' (ESM2).
(doi: 10.1126/science.ade2574). Contrary to other post-processing modules, this encoding-based
post-processor stores the encodings in a separate file, but in the same order as observed in the
original file. 
"""

from pathlib import Path

import pandas as pd

from postprocessing.post_processing import PostProcessor


class EncodeESM(PostProcessor):
    """
    ## PLM Encoder Using ESM
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
