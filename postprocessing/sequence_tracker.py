"""
---------------------------------- Observed Antibody Space CS -----------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics(doi: 10.1002/pro.4205)\n
-------------------------------------- Sequence Tracker ------------------------------------------\n
Tracks sequences across multiple files for postprocessors.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, TypeAlias, Iterable
from collections import defaultdict

SequenceIdType: TypeAlias = tuple[str, str]
StatusType: TypeAlias = Literal[
    "delete", "keep", "train", "test", "validate", "sampled"
]


@dataclass
class SequenceStatus:
    """
    ## Data Container for Individual Sequences and their Label
    Can contain the actual sequence or only their status. As the ID of the sequence is stored
    in the SequenceTracker, the status can be uniquely identified by the SequenceID and not
    only by its sequence.
    ### Args:
        \tstatus {Literal} -- Can be 'delete', 'train', 'test', 'validate', 'sampled'
        \tsequence {str} -- Optional: Sequence of the protein in str form
    """

    status: StatusType = "keep"
    sequence: Optional[str] = None


@dataclass
class SequenceTracker:
    """
    ## Tracks Sequences Across Files
    The SequenceTracker tracks sequences within a file or within a directory
    of OASCS files. The tracker assigns an ID to each sequence and its status.
    Each ID can also be assigned to a category for classifier purposes.
    The deleted list, is a list of sequences that are to-be-deleted by the
    postprocessor.
    ### Args:
        \tidentities {dict{SequenceID, SequenceStatus}} -- ID status pairing of the tracker.
            The ID contains information on the index and origin file of the sequence. \n
        \tcategories {dict[str, SequenceID]} -- Assignment of each sequence ID to a given
            category.
    """

    identities: dict[SequenceIdType, SequenceStatus] = field(
        default_factory=defaultdict(SequenceStatus)
    )
    categories: Optional[dict[str, list[SequenceIdType]]] = field(
        default_factory=defaultdict(list)
    )
    deleted: Optional[list[str]] = field(default_factory=list)

    def add_default_identities(
        self,
        ids: Iterable[SequenceIdType],
        default_status: StatusType,
        overwrite: bool = False,
    ) -> None:
        """
        ## From sequence ids, creates the identities dict
        By default does not overwrite the existing identities dict
        """
        if overwrite:
            self.identities = {}

        for identifier in ids:
            self.identities[identifier].status = default_status

    def update_status(self, new_status: dict[SequenceIdType, StatusType]):
        """
        ## Updates SequenceStatus for each provided SequenceID
        """
        for identity, updated_status in new_status.items():
            self.identities[identity].status = updated_status

    def update_deleted_sequences(self):
        """
        ## Updates the deleted list
        """
        for sequence_status in self.identities.values():
            if sequence_status.status == "delete":
                assert (
                    sequence_status.sequence
                ), "In order to update the deleted list, \
                the sequence must be known by SequenceStatus and cannot be an empty string."
                self.deleted.append(sequence_status.sequence)
