"""
---------------------------------- Observed Antibody Space API -----------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics(doi: 10.1002/pro.4205)\n
-------------------------------- Antibody Viability Post Processing ------------------------------\n
Performs post processing in OAS API. Checks antibody viability inspired by ABOSS:
doi: https://doi.org/10.4049%2Fjimmunol.1800669
"""

import math

from pathlib import Path
from dataclasses import dataclass
from typing import Union, Optional, Literal
from collections import defaultdict, Counter
from itertools import chain as iter_chain
from functools import partial
from multiprocessing import Pool

from anarci import anarci
from tqdm import tqdm

import pandas as pd
import numpy as np

from postprocessing.post_processing import PostProcessor, DTYPE_DICT


@dataclass
class SequenceTracker:
    """
    ## Dataclass for processing sequences
    """

    sequences: list[tuple[str, str]]
    status: Optional[dict[str, bool]] = None

    def __post_init__(self):
        # Make sure data is in correct format
        assert isinstance(self.sequences, list) and isinstance(
            self.sequences[0], tuple
        ), "Sequences attribute provided needs to be of type list[tuple[str, str]]"
        # Make more memory efficient, the length of the sequences attribute is now immutable
        self.sequences = np.array(self.sequences)
        if self.status is None:
            self.status = {}
            for identity, _ in self.sequences:
                self.status[identity] = True

        self.deleted = []

    def update_status(self, new_status: dict):
        """
        ## Updates status attribute
        """
        for key in new_status.keys():
            self.status[key] = new_status[key][0]

    def update_deleted_sequences(self):
        """
        ## Stores sequences if their status is false
        """
        for identity, sequence in self.sequences:
            if not self.status[identity]:
                self.deleted.append(sequence)


class AntibodyViability(PostProcessor):
    """
    ## Antibody Viability Post Processor for OAS API
    FINISH DOCSTRINGS
    """

    _assign_germline = True

    def __init__(
        self,
        directory_or_file_path: Path,
        output_directory: Path,
        data: Optional[pd.DataFrame] = None,
        filter_strictness: Literal["loose", "strict"] = "loose",
        batch_size: Union[int, Literal["dynamic"]] = "dynamic",
        ncpus: int = 1,
        max_batch_size: int = 10000,
    ) -> None:
        super().__init__(
            directory_or_file_path=directory_or_file_path,
            output_directory=output_directory,
        )
        if data is not None:
            assert isinstance(
                data, pd.DataFrame
            ), "Data needs to be a pandas DataFrame."
        self.data = data
        self.path: Optional[Path] = None
        self.tracker: Optional[SequenceTracker] = None
        self.filter_strictness = filter_strictness
        self.batch_size = batch_size
        self.ncpus = ncpus
        self.max_batch_size = max_batch_size

    def load_file(self, file_path: Path, overwrite: bool = False):
        """
        ## Loads OAS CS file provided
        ### Args:
                \t path {Path} -- Path to file \n
                \t overwrite {bool} -- Whether or not to overwrite an already loaded file
        ### Updates:
                \t self.data {DataFrame} -- Data from provided file
        """
        if not overwrite:
            assert (
                self.data is None
            ), "File has already been loaded, set overwrite to True"
            self.data = pd.read_csv(
                file_path,
                index_col=0,
                dtype=DTYPE_DICT,
                usecols=["Chain", "Sequence_aa"],
            )
        else:
            self.data = pd.read_csv(
                file_path, index_col=0, dtype=DTYPE_DICT, usecols=["Sequence_aa"]
            )

        self.path = file_path

    def save_file(self, file_path: Path, data: pd.DataFrame):
        """
        ## Saves processed file as OAS API file
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(file_path)

    def update_data(self):
        """
        ## Updates data attribute based on the current status of sequence tracker
        """
        tqdm.write("Removing Sequences from Dataset...")
        # We only load "Sequence_aa before to save memory"
        self.data = pd.read_csv(self.path, index_col=0, dtype=DTYPE_DICT)
        self.data = self.data[~self.data["Sequence_aa"].isin(self.tracker.deleted)]
        tqdm.write("Finished removing sequences!")

    # PROCESS SHOULDN'T HAVE ANY ARGUMENTS
    def process(
        self,
    ):
        """
        ## Processes currently loaded OAS API file
        Processes files according to the ABOSS workflow. First, sequences
        are aligned and numbered with ANARCI. Then, the antibody sequence
        is checked for anomalies such as missing conserved residues or
        abnormally sized frameworks/cdrs. Next, the sequence identity to
        the closest v & j genes are checked (>50%). Lastly, all abnormal
        mutations are filtered out by checking the sequencing error frequency
        in a batch of data through conserved residues. Only non-abnormal
        frequency residue sequences are kept.
        ### Args:
                \tbatch_size {int or "dynamic"} -- Number of sequences to process in a batch.
                Recommended >= 1000. Dynamic sets batch size optimally on a per-file basis. \n
                \tncpus {int} -- Number of cpus to process with \n
                \tmax_batch_size {int} -- Only used in dynamic mode.
                Sets upper bound on batch size to not overload memory.
        ### Updates:
                \tself.data {DataFrame} -- Dataframe containing filtered sequences
        """

        # Load if valid path has been provided
        if self.directory_or_file_path:
            self.get_files_list(directory_or_file_path=self.directory_or_file_path)

        for file in self.all_files:
            self.load_file(file_path=file)
            self.filter_sequences(
                filter_strictness=self.filter_strictness,
                batch_size=self.batch_size,
                ncpus=self.ncpus,
                max_batch_size=self.max_batch_size,
            )

    def filter_sequences(
        self,
        filter_strictness: Literal["loose", "strict"],
        batch_size: Union[int, Literal["dynamic"]],
        ncpus: int,
        max_batch_size: int,
    ):
        """
        ## Performs filtering of sequences
        """
        tqdm.write("Starting antibody viability filtering...")
        # Set batch size, dynamic is especially useful for files with few sequences
        if batch_size == "dynamic":
            batch_size = int(math.ceil(len(self.data) / ncpus))
            if batch_size > max_batch_size:
                batch_size = max_batch_size

        data_chunks = self._package_batches(
            sequences=self.data["Sequence_aa"], batch_size=batch_size
        )
        # Partially apply filter_strictness already
        partial_process_batch = partial(
            self.process_batch, filter_strictness=filter_strictness
        )
        all_results = []
        # Multiprocessing of sequence batches
        with Pool(processes=ncpus) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(partial_process_batch, data_chunks),
                    total=len(data_chunks),
                )
            )
            for sequence_status in results:
                all_results.append(sequence_status)

        # Remove sequences from Sequence Tracker that show up in all_results
        merged_results = self._combine_dictionaries(all_results)
        self.tracker.update_status(merged_results)
        self.tracker.update_deleted_sequences()
        self.update_data()

    def process_batch(
        self,
        batch: tuple[tuple[str, str]],
        filter_strictness: Literal["loose", "strict"],
    ):
        """
        ## Processes one batch of data
        First runs anarci, then checks if sequences could not be parsed by anarci.
        Next, the sequence is checked whether it aligns well to its designated germline.
        Next, the sequence is checked for issues with it on a per-position basis,
        as well as checking the lengths of the different frames.
        Lastly, the false-read rate is inferred and
        every mutation that occurs less often than the false read-rate is rejected.
        ### Args:
            \t batch {list} -- Batch of data in the anarci desired format
        """
        # Create status dict
        batch_status = {}
        # Keep track of how often cysteine is conserved at position 23 and 104
        conserved_23 = []
        conserved_104 = []
        # For Sequences that have passed the check, we want to know what AAs are at each position
        amino_acids_by_position_list = []

        numbered_batch = self.get_anarci_numbering(sequences=batch)
        for index, numbered_sequence in enumerate(numbered_batch[0]):
            # From simplest to calculate to hardest we filter out sequences

            # If anarci couldn't align the sequence, set status to false
            if numbered_sequence is None:
                batch_status[batch[index][0]] = False
                continue

            # Check if it aligns well to its designated germline
            for gene in ["v_gene", "j_gene"]:
                if numbered_batch[1][index][0]["germlines"][gene][1] < 0.5:
                    batch_status[batch[index][0]] = False

            # Check for by position issues with sequence
            positional_issues = self._flagged_by_position(
                numbered_sequence, sequence_info=numbered_batch[1][index]
            )
            # If first return value is ever True, the sequence has been flagged
            if positional_issues[0]:
                batch_status[batch[index][0]] = False
                continue

            conserved_23.append(positional_issues[1])
            conserved_104.append(positional_issues[2])
            amino_acids_by_position_list.append(positional_issues[3])

        # Check plausible probabilities of mutations
        if len(conserved_23) != 0:
            false_read_position_23 = 1 - (conserved_23.count(True) / len(conserved_23))
        else:
            false_read_position_23 = 0
        if len(conserved_104) != 0:
            false_read_position_104 = 1 - (
                conserved_104.count(True) / len(conserved_104)
            )
        else:
            false_read_position_104 = 0
        # Get false read probability (i.e. sequencing error as opposed to mutation)
        if filter_strictness == "loose":
            false_read_probability = min(
                false_read_position_23, false_read_position_104
            )
        else:
            false_read_probability = max(
                false_read_position_23, false_read_position_104
            )

        # Get probability of each amino acid at a given numbered position
        residue_probabilities_by_position = self.get_residue_probabilities(
            amino_acids_by_position_list
        )

        # Second round of filtering
        for index, numbered_sequence in enumerate(numbered_batch[0]):
            # Skip the ones that have been filtered already
            if batch[index][0] in batch_status:
                continue
            if self.filter_by_residue_probability(
                numbered_sequence=numbered_sequence,
                filter_strictness=filter_strictness,
                probability_threshold=false_read_probability,
                residue_probabilities=residue_probabilities_by_position,
            ):
                batch_status[batch[index][0]] = False

        return batch_status

    def filter_by_residue_probability(
        self,
        numbered_sequence: list[tuple],
        filter_strictness: Literal["loose", "strict"],
        probability_threshold: float,
        residue_probabilities: dict[dict[float]],
    ) -> bool:
        """
        ## Filters residues based on their per-residue probabilities
        """
        # Go through each position (Due to the structure the positions are at [0][0])
        for position in numbered_sequence[0][0]:
            if position[1] == "-":
                continue

            # Always delete non-cysteines at the following positions
            if position[0][0] in ["23", "104"] and position[1] != "C":
                return True

            # Skip CDRs since they can have much lower probabilities naturally
            if (
                position[0][0]
                in iter_chain(range(27, 39), range(56, 66), range(105, 118))
                and filter_strictness == "loose"
            ):
                continue

            position_name = str(position[0][0]) + position[0][1]
            # Check if the probability is below the threshold
            if (
                residue_probabilities[position_name][position[1]]
                <= probability_threshold
            ):
                return True
        return False

    def get_residue_probabilities(self, residue_list: list) -> dict:
        """
        # Returns the probabilities of each residue at each position
        """
        # Merge all dictionaries
        merged_residue_dictionary = self._combine_dictionaries(residue_list)

        # Iterate through all numbered positions
        for _, value in merged_residue_dictionary.items():
            # Get position counts
            position_counter = Counter(value)
            total_counts = position_counter.total()

            # Iterate through counter items to update with residue probabilities
            for item, _ in position_counter.items():
                position_counter[item] /= total_counts
            # Update dictionary
            value = position_counter
        return merged_residue_dictionary

    @staticmethod
    def _flagged_by_position(
        numbered_sequence: list[tuple], sequence_info: list[dict]
    ) -> tuple[bool, Optional[int], Optional[int], Optional[dict]]:
        """
        ## Flags based off of certain positions
        Although less readable if it was split into multiple functions,
        going through multiple 'for loops' is slow as well.
        First checks if the CDR3 has been flagged for an indel.
        Then checks conserved positions, special positions, light chain
        absent positions and special rabbit positions.
        Sorry for anyone who has to understand this... \n
        ### Notes:
            \t - return value True means that the sequence is faulty and will be skipped.\n
            \t - position`[0][0]` is the numbered index from anarci \n
            \t - position`[0][1]` is the lettered index from anarci e.g. the A in 112A from CDR3 \n
            \t - position`[1]` is the single-letter code amino acid \n
            \t - sequence_info`[0][Literal]` contains anarci info about the sequence
        """
        # Set values to compare to
        conserved_residues = 0
        absent_residues = 2
        fwr3_adjustment = 0
        rabbit_position_84 = 1
        start_found = False
        conserved_23 = False
        conserved_104 = False
        amino_acid_by_position = {}

        # Frame sequence registry of frames we want to check the length of
        frames = {"fwr2": 0, "fwr3": 0, "fwr4": 0, "cdr3": 0}

        for position in numbered_sequence[0][0]:
            # Check if framework 1
            if position[0][0] < 27:
                # Find the starting position as sequencing can sometimes start later than 1 in fwr1
                if position[1] != "-" and not start_found:
                    start_found = True

                # If the position is skipped and the start has been found
                if position[1] == "-" and start_found:
                    # Rabbit is a special case where positions 2 and 10 can be skipped
                    if sequence_info[0]["species"] == "rabbit":
                        if position[0][0] not in [2, 10]:
                            return (True, None, None, None)
                    # Else only position 10 can be skipped
                    if position[0][0] != 10:
                        return (True, None, None, None)

            # Unlabeled positions are skipped
            if position[1] == "-":
                continue

            # Register important frames
            if position[0][0] in range(39, 56):
                frames["fwr2"] += 1  # Register that AA is in fwr2

            if position[0][0] in range(66, 105):
                frames["fwr3"] += 1  # Register that AA is in fwr3
                # Check if fwr3 needs to be adjusted
                if position[0][0] == 73:
                    fwr3_adjustment += 1
                # These can be absent in the light chain, we only check the first possible alignment
                if position[0][0] in [81, 82] and sequence_info[0]["chain_type"] == "L":
                    absent_residues -= 1
                # This rabbit specific position can be absent
                if position[0][0] == 84 and sequence_info[0]["species"] == "rabbit":
                    rabbit_position_84 -= 1

            if position[0][0] in range(105, 118):
                frames["cdr3"] += 1  # Register that AA is in cdr3

            if position[0][0] in range(118, 129):
                frames["fwr4"] += 1  # Register that AA is in fwr4

            # Check if flagged by anarci
            if position[0][1] != " ":
                # CDR3 is between 105 and 117 in imgt
                if position[0][0] < 105 or position[0][0] > 117:
                    return (True, None, None, None)
            # Check conserved residues
            if position[0][0] in [23, 104, 118]:
                conserved_residues += 1
                if position[0][0] == 23 and position[1] == "C":
                    conserved_23 = True
                if position[0][0] == 104 and position[1] == "C":
                    conserved_104 = True

            # Register all positions and their amino acids in a dict
            amino_acid_by_position[str(position[0][0]) + position[0][1]] = position[1]

        # Maybe we can make this more efficient by pre-adjusting the lengths
        # Check lengths
        if frames["fwr2"] < 17:  # FWR2 of length less than 17 are faulty
            return (True, None, None, None)
        if sequence_info[0]["chain_type"] == "H":
            if frames["cdr3"] > 37:  # CDR3 of length more than 37 are chimeric
                return (True, None, None, None)
            if (
                frames["fwr3"] - fwr3_adjustment < 38
            ):  # FWR3 of adjusted length less than 38 are faulty
                return (True, None, None, None)
        if sequence_info[0]["species"] == "rabbit":
            if (
                frames["fwr3"] - fwr3_adjustment + rabbit_position_84 < 38
            ):  # FWR3 of rabbits adjusted with length less than 38 are faulty
                return (True, None, None, None)
            if frames["fwr4"] < 10:  # FWR4 of rabbits less than 10 are faulty
                return (True, None, None, None)
        if sequence_info[0]["chain_type"] == "L":
            if (
                frames["fwr3"] + absent_residues - fwr3_adjustment < 38
            ):  # Light chain FWR3 adjusted with length less than 38 are faulty
                return (True, None, None, None)
            if frames["fwr4"] < 10:
                return (True, None, None, None)
        if frames["fwr4"] < 11:
            return (True, None, None, None)
        if frames["fwr4"] > 13:
            return (True, None, None, None)

        return (False, conserved_23, conserved_104, amino_acid_by_position)

    def _package_batches(
        self, sequences: pd.Series, batch_size: int = 1000
    ) -> tuple[tuple[str, str]]:
        """
        ## Packages sequences to anarci desired format
        Format is [("seq_name_1", "SEQVENCE"), ... ("seq_name_n", "SEQVENCE")]
        """
        package = []
        # First check sequence is complete
        for iterator, sequence in enumerate(sequences):
            if self._legal_characters(sequence):
                package.append((f"Seq_{iterator}", sequence))
        if package:
            # Keep track of sequence status using sequence tracker
            self.tracker = SequenceTracker(sequences=package)
            # Free up memory (Important if files are big)
            self.data = pd.DataFrame()
            # Package in chunks of batch_size
            return tuple(
                package[n : n + batch_size] for n in range(0, len(package), batch_size)
            )
        else:
            raise IndexError("No valid sequences in provided dataset")

    @staticmethod
    def _legal_characters(sequence) -> bool:
        """
        ## Checks if sequence is made of amino acids
        """
        allowed = set(
            [
                "A",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "K",
                "L",
                "M",
                "N",
                "P",
                "Q",
                "R",
                "S",
                "T",
                "V",
                "W",
                "Y",
            ]
        )
        # If anything in sequence is not in allowed, it's not returned
        return set(sequence) <= allowed

    @staticmethod
    def _combine_dictionaries(list_of_dictionaries: list[dict]) -> dict:
        """
        ## Combines dictionaries and places values in a list
        """
        # Setup defaultdict
        output_dictionary = defaultdict(list)
        for data in list_of_dictionaries:
            for key, value in data.items():
                output_dictionary[key].append(value)
        return output_dictionary

    def get_anarci_numbering(self, sequences: tuple[tuple[str, str]]) -> tuple:
        """
        ## Runs ANARCI
        Runs anarci and returns numbered sequences, alignment matrix and hit table
        """
        return anarci(sequences=sequences, assign_germline=self._assign_germline)


if __name__ == "__main__":
    print("This is the antibody viability file")
    # oas_files_path = Path("/central/groups/smayo/lschaus/OAS_Files").glob("**/*")
    # list_of_files = [file for file in oas_files_path if file.is_file()]

    # for file in list_of_files:
    #     file_size = file.stat().st_size / 1024
    #     if file_size < 1000:
    #         NCPUS_TEST = 2
    #         MAX_BATCH_SIZE_TEST = 1000
    #     elif file_size < 300_000:
    #         NCPUS_TEST = 32
    #         MAX_BATCH_SIZE_TEST = 1000
    #     else:
    #         NCPUS_TEST = 10
    #         MAX_BATCH_SIZE_TEST = 500
    #     A = AntibodyViability(directory_or_file_path="", output_directory="")
    #     A.load_file(file)
    #     A.process(
    #         batch_size="dynamic", ncpus=NCPUS_TEST, max_batch_size=MAX_BATCH_SIZE_TEST
    #     )
    #     save_path = Path("/central/groups/smayo/lschaus/OAS_Processed")
    #     save_path = save_path / file.name
    #     A.save_file(save_path, A.data)

    # # First batch
    # A = AntibodyViability()
    # A.load_file(Path(path))
    # A.process(batch_size="dynamic", ncpus=1, max_batch_size=1000)
    # A.save_file(
    #     Path("/central/groups/smayo/lschaus/OAS_Processed/OAS_Human_IGHG_ASC_00001.csv")
    # )
