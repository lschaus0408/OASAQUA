"""
---------------------------------- Observed Antibody Space CS -----------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics(doi: 10.1002/pro.4205)\n
-------------------------------------- Sequence Tracker ------------------------------------------\n
Tracks sequences across multiple files for postprocessors.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SequenceTracker:
    """
    ## Tracks Sequences Across Files
    FINISH DOCSTRINGS
    ### Args:
        \tsequences {list[tuple[str, str]]} --
        \tstatus {dict[str, bool]} --
    """

    sequences: list[tuple[str, str]]
    status: Optional[dict[str, bool]] = None
