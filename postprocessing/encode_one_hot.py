"""
---------------------------------- Observed Antibody Space CS -----------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics(doi: 10.1002/pro.4205)\n
--------------------------------------- Encode via One-Hot ---------------------------------------\n
Performs encoding of antibody sequences in the dataset via One-Hot encoding. A One-Hot encoding 
assigns a vector of size 21 to each position, i, in a protein sequence to create a Nx21 matrix.
Each position, j, in the vector corresponds to one of the 20 essential amino-acids + one position 
for an empty position. When an amino acid is present at position i, all of the values of the vector
are set to 0 except for j, which takes the value of 1.
"""

from pathlib import Path

import pandas as pd

from postprocessing.post_processing import PostProcessor


class EncodeOneHot(PostProcessor):
    """
    ## Encoder Using One-Hot
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
