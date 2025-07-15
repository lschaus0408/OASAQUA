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
from typing import Union, Optional, Literal
from collections import defaultdict, Counter
from functools import partial

from anarci import anarci
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

import pandas as pd

from postprocessing.post_processing import PostProcessor, DTYPE_DICT
from postprocessing.sequence_tracker import SequenceTracker


class AntibodyViability(PostProcessor):
    """
    ## Antibody Viability Post Processor for OAS API
    FINISH DOCSTRINGS
    """

    # Defining constants
    _assign_germline = True
    CONSERVED_POSITIONS = {23, 104, 118}
    CRITICAL_CYS_POSITIONS = {"23", "104"}
    FWR2_POSITIONS = set(range(39, 56))
    FWR3_POSITIONS = set(range(66, 105))
    CDR3_POSITIONS = set(range(105, 118))
    FWR4_POSITIONS = set(range(118, 129))
    START_POSITION = 27
    SKIPPED_CDR_POSITIONS_LOOSE = {
        *map(str, range(27, 39)),
        *map(str, range(56, 66)),
        *map(CDR3_POSITIONS),  # type: ignore
    }

    def __init__(
        self,
        directory_or_file_path: Path,
        output_directory: Path,
        data: Optional[pd.DataFrame] = None,
        filter_strictness: Literal["loose", "strict"] = "loose",
        batch_size: Union[int, Literal["dynamic"]] = "dynamic",
        n_jobs: int = 1,
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
        self.sequence_tracker: SequenceTracker = SequenceTracker()
        self.filter_strictness = filter_strictness
        self.batch_size = batch_size
        self.n_jobs = n_jobs
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
                file_path,
                index_col=0,
                dtype=DTYPE_DICT,
                usecols=["Chain", "Sequence_aa"],
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
        save_data = self.data[
            ~self.data["Sequence_aa"].isin(self.sequence_tracker.deleted)
        ].reset_index(drop=True)
        self.save_file(file_path=self.path, data=save_data)
        tqdm.write("Finished removing sequences!")

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

        for _, file in enumerate(self.all_files):
            self.load_file(file_path=file, overwrite=True)
            if self.data.empty:
                file.unlink(missing_ok=True)
                continue
            tqdm.write(
                f"Starting antibody viability filtering for file {str(file.stem)}..."
            )
            self.filter_sequences(
                filter_strictness=self.filter_strictness,
                batch_size=self.batch_size,
                ncpus=self.n_jobs,
                max_batch_size=self.max_batch_size,
            )
            # Reset Sequence Tracker
            self.sequence_tracker = SequenceTracker()

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
        # Set batch size, dynamic is especially useful for files with few sequences
        if batch_size == "dynamic":
            batch_size = int(math.ceil(len(self.data) / ncpus))
            if batch_size > max_batch_size:
                batch_size = max_batch_size

        data_chunks = self._package_batches(
            sequences=self.data["Sequence_aa"], batch_size=batch_size  # type: ignore
        )
        # Partially apply filter_strictness alreadys
        partial_process_batch = partial(
            self.process_batch, filter_strictness=filter_strictness
        )

        all_results = []
        # Multiprocessing of sequence batches
        with Pool(processes=ncpus) as pool:
            results = list(
                tqdm(
                    pool.uimap(partial_process_batch, data_chunks),
                    total=len(data_chunks),
                )
            )
            for sequence_status in results:
                all_results.append(sequence_status)

        # Remove sequences from Sequence Tracker that show up in all_results
        merged_results = self._combine_dictionaries(all_results)
        self.sequence_tracker.update_status(merged_results)
        self.sequence_tracker.update_deleted_sequences()
        self.update_data()

    def process_batch(
        self,
        batch: tuple[tuple[str, str]],
        filter_strictness: Literal["loose", "strict"],
    ) -> dict[str, bool]:
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

        # Unpack sequence_ids for faster lookups
        batch_sequence_ids = [seq_id for seq_id, _ in batch]

        numbered_batch, batch_sequence_info = self.get_anarci_numbering(sequences=batch)
        for index, (sequence_id, numbered_sequence, sequence_info) in enumerate(
            zip(batch_sequence_ids, numbered_batch, batch_sequence_info)
        ):
            # From simplest to hardest to calculate we filter out sequences

            # If anarci couldn't align the sequence, set status to false
            if numbered_sequence is None:
                batch_status[sequence_id] = False
                continue

            # Check if it aligns well to its designated germline
            germline_scores = sequence_info[0].get("germlines", {})
            if any(
                germline_scores.get(gene, [None, 0])[1] < 0.5
                for gene in ("v_gene", "j_gene")
            ):
                batch_status[batch[index][0]] = False

            # Check for by position issues with sequence
            is_flagged, has_cys23, has_cys104, aa_by_position = (
                self._flagged_by_position(
                    numbered_sequence, sequence_info=numbered_batch[1][index]
                )
            )
            # If first return value is ever True, the sequence has been flagged
            if is_flagged:
                batch_status[sequence_id] = False
                continue

            conserved_23.append(has_cys23)
            conserved_104.append(has_cys104)
            amino_acids_by_position_list.append(aa_by_position)

        # Check plausible probabilities of mutations
        if conserved_23:
            false_read_position_23 = 1 - (sum(conserved_23) / len(conserved_23))
        else:
            false_read_position_23 = 0

        if conserved_104:
            false_read_position_104 = 1 - (sum(conserved_104) / len(conserved_104))
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
        for index, (sequence_id, numbered_sequence) in enumerate(
            zip(batch_sequence_ids, numbered_batch)
        ):
            # Skip the ones that have been filtered already
            if sequence_id in batch_status:
                continue
            if self.filter_by_residue_probability(
                numbered_sequence=numbered_sequence,
                filter_strictness=filter_strictness,
                probability_threshold=false_read_probability,
                residue_probabilities=residue_probabilities_by_position,
            ):
                batch_status[sequence_id] = False

        return batch_status

    def filter_by_residue_probability(
        self,
        numbered_sequence: list[tuple],
        filter_strictness: Literal["loose", "strict"],
        probability_threshold: float,
        residue_probabilities: dict[str, dict[str, float]],
    ) -> bool:
        """
        ## Filters residues based on their per-residue probabilities
        """
        # Precompute sets
        if filter_strictness == "loose":
            skip_positions = self.SKIPPED_CDR_POSITIONS_LOOSE
        else:
            skip_positions = set()

        # Go through each position (Due to the structure the positions are at [0][0])
        for position in numbered_sequence[0][0]:
            position_number, position_letter, aa = (
                position[0][0],
                position[0][1],
                position[1],
            )
            if aa == "-":
                continue

            position_string = str(position_number)
            position_name = position_string + position_letter

            # Always delete non-cysteines at the following positions
            if position_string in self.CRITICAL_CYS_POSITIONS and aa != "C":
                return True

            # Skip CDRs since they can have much lower probabilities naturally
            if position_string in skip_positions:
                continue

            # Check if the probability is below the threshold
            if (
                residue_probabilities.get(position_name, {}).get(aa, 1.0)
                <= probability_threshold
            ):
                return True
        return False

    def get_residue_probabilities(
        self, residue_list: list[dict[str, str]]
    ) -> dict[str, dict[str, float]]:
        """
        # Returns the probabilities of each residue at each position
        """
        # Merge all dictionaries
        merged_residue_dictionary = defaultdict(Counter)
        for sequence_position_data in residue_list:
            for key, value in sequence_position_data.items():
                merged_residue_dictionary[key][value] += 1

        # Normalize counts into probabilities
        output_dict = {
            position: {
                value: count / sum(counter.values()) for value, count in counter.items()
            }
            for position, counter in merged_residue_dictionary.items()
        }

        return output_dict

    def _flagged_by_position(
        self, numbered_sequence: list[tuple], sequence_info: list[dict]
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
        # Trackers
        conserved_23 = False
        conserved_104 = False
        amino_acid_by_position = {}

        # Inputs
        chain_type = sequence_info[0]["chain_type"]
        species = sequence_info[0]["species"]

        # Frame sequence registry of frames we want to check the length of
        frames = {"fwr2": 0, "fwr3": 0, "fwr4": 0, "cdr3": 0}

        for position in numbered_sequence[0][0]:
            position_index, position_letter, aa = (
                position[0][0],
                position[0][1],
                position[1],
            )
            position_string = str(position_index) + position_letter

            # Skip unlabeled positions
            if aa == "-":
                # Check for skips after seq starts fwr1
                if position_index < self.START_POSITION:
                    if start_found:
                        if species == "rabbit" and position_index not in {2, 10}:
                            return (True, None, None, None)
                        elif position_index != 10:
                            return (True, None, None, None)
                continue

            # Mark the start of the sequence
            if position_index < self.START_POSITION and not start_found:
                start_found = True

            # Register
            amino_acid_by_position[position_string] = aa

            # Register important frames
            if position_index in self.FWR2_POSITIONS:
                frames["fwr2"] += 1  # Register that AA is in fwr2

            if position_index in self.FWR3_POSITIONS:
                frames["fwr3"] += 1  # Register that AA is in fwr3
                # Check if fwr3 needs to be adjusted
                if position_index == 73:
                    fwr3_adjustment += 1
                # These can be absent in the light chain, we only check the first possible alignment
                if position_index in {81, 82} and chain_type == "L":
                    absent_residues -= 1
                # This rabbit specific position can be absent
                if position_index == 84 and species == "rabbit":
                    rabbit_position_84 -= 1

            if position_index in self.CDR3_POSITIONS:
                frames["cdr3"] += 1  # Register that AA is in cdr3

            if position_index in self.FWR4_POSITIONS:
                frames["fwr4"] += 1  # Register that AA is in fwr4

            # Check if flagged by anarci
            if position_letter != " ":
                # CDR3 is between 105 and 117 in imgt
                if position_index not in self.CDR3_POSITIONS:
                    return (True, None, None, None)

            # Check conserved residues
            if position_index in self.CONSERVED_POSITIONS:
                conserved_residues += 1
                if position_index == 23 and aa == "C":
                    conserved_23 = True
                if position_index == 104 and aa == "C":
                    conserved_104 = True

        # Maybe we can make this more efficient by pre-adjusting the lengths
        # Check lengths
        if frames["fwr2"] < 17:  # FWR2 of length less than 17 are faulty
            return (True, None, None, None)

        if chain_type == "H":
            if frames["cdr3"] > 37:  # CDR3 of length more than 37 are chimeric
                return (True, None, None, None)
            if (
                frames["fwr3"] - fwr3_adjustment < 38
            ):  # FWR3 of adjusted length less than 38 are faulty
                return (True, None, None, None)

        if chain_type == "L":
            if (
                frames["fwr3"] + absent_residues - fwr3_adjustment < 38
            ):  # Light chain FWR3 adjusted with length less than 38 are faulty
                return (True, None, None, None)
            if frames["fwr4"] < 10:
                return (True, None, None, None)

        if species == "rabbit":
            if (
                frames["fwr3"] - fwr3_adjustment + rabbit_position_84 < 38
            ):  # FWR3 of rabbits adjusted with length less than 38 are faulty
                return (True, None, None, None)
            if frames["fwr4"] < 10:  # FWR4 of rabbits less than 10 are faulty
                return (True, None, None, None)

        if frames["fwr4"] < 11 or frames["fwr4"] > 13:
            return (True, None, None, None)

        return (False, conserved_23, conserved_104, amino_acid_by_position)

    def _package_batches(
        self,
        sequences: pd.Series,
        batch_size: int = 1000,
        check_characters: bool = False,
    ) -> tuple[list[tuple[str, str]], ...]:
        """
        ## Packages sequences to anarci desired format
        Format is [("seq_name_1", "SEQVENCE"), ... ("seq_name_n", "SEQVENCE")]
        """
        package: list[tuple[str, str]] = []
        # First check sequence is complete
        for iterator, sequence in enumerate(sequences):
            if check_characters:
                if self._legal_characters(sequence=sequence):
                    package.append((f"Seq_{iterator}", sequence))
            else:
                package.append((f"Seq_{iterator}", sequence))

        if package:
            # Keep track of sequence status using sequence tracker
            self.sequence_tracker.add_default_identities(
                [(0, int(identifier.split("_")[1])) for identifier, _ in package],
                default_status="keep",
                sequences=[seq[1] for seq in package],
            )
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
    def _combine_dictionaries(list_of_dictionaries: list[dict[str, bool]]) -> dict:
        """
        ## Combines dictionaries and converts the values
        Convert from {"Seq_id": bool} to {(0,id): "delete"} to make
        it compatible with SequenceTracker
        """
        # Setup defaultdict
        output_dictionary = {}
        for data in list_of_dictionaries:
            for key, _ in data.items():
                index = int(key.split("_")[1])
                output_dictionary[(0, index)] = "delete"
        return output_dictionary

    def get_anarci_numbering(self, sequences: tuple[tuple[str, str]]) -> tuple:
        """
        ## Runs ANARCI
        Runs anarci and returns numbered sequences, alignment matrix and hit table
        """
        return anarci(sequences=sequences, assign_germline=self._assign_germline)


if __name__ == "__main__":
    print("This is the antibody viability file")
