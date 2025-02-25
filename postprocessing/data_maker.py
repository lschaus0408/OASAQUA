"""
---------------------------------- Observed Antibody Space CS -----------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics(doi: 10.1002/pro.4205)\n
------------------------------------------- Data Maker ------------------------------------------\n
Splits the dataset into training, test, and validation sets.
"""

import warnings
import random

from pathlib import Path
from typing import Union, Literal, Optional, Any
from math import isclose
from functools import partial
from collections import defaultdict

import pandas as pd

from tqdm import tqdm

from postprocessing.post_processing import PostProcessor, DTYPE_DICT
from postprocessing.sequence_tracker import (
    SequenceTracker,
    StatusType,
    SequenceStatus,
    SequenceIdType,
)

dataset_definition_type = Literal["train", "test", "validation"]
priority_type = tuple[dataset_definition_type, ...]


class DataMaker(PostProcessor):
    """
    ## DataMaker for OASCS
    Splits a dataset into training, test, and validation sets.
    Takes a folder with OAS API files and first loads only the desired category
    column in those files. The catgories are stored in a set to find out which one's exist.
    Users can provide aliases in a dictionary to combine categories(e.g. different mouse
    strain names to mouse). Then all sequences are stored in a dictionary of the structure:
        \t {"category": [(file_id,  index), (..., ...), ...]} \n
    Users have to provide what number of sequences the training/test set file should have.
    Users can provide the ratios of categories desired, if none is provided, categories are
    sampled evenly. If a test set is desired, this can be specified together with the number
    of sequences in the file or per category.
    ### Args:
        \t directory_or_file_path {Path} -- Path where file(s) to be processed are present \n
        \t dataset_numbers {dict[str,int]} -- Optional: Number of sequences in the train,
        test, and validation sets \n
        \t dataset_ratios {dict[str, float]} -- Optional: Data split ratio for train, test,
        and validation sets \n
        \t category_column {str} -- Optional: Column from which to pick categories from \n
        \t aliases {dict} -- Optional: Convert categories to different names \n
        \t category_ratios {dict} -- Optional: Ratio of categories in set \n
        \t verbose {bool} -- Optional: Provides additional information during processing. \n
        \t dataset_priority {priority_type} -- Optional: Changes the priority of data
        allocation when sampling
    Note: Only one of 'dataset_numbers' or 'dataset_ratios' can be passed as a value.
    Passing both, will default to ratios.
    """

    def __init__(
        self,
        directory_or_file_path: Path,
        dataset_numbers: Optional[dict[StatusType, int]] = None,
        dataset_ratios: Optional[dict[StatusType, float]] = None,
        category_column: Optional[str] = None,
        aliases: Optional[dict[str, str]] = None,
        category_ratios: Optional[dict[str, float]] = None,
        verbose: Optional[bool] = False,
        dataset_priority: Optional[priority_type] = None,
    ):
        self.directory_or_file_path = directory_or_file_path
        self.dataset_numbers = dataset_numbers
        self.dataset_ratios = dataset_ratios
        self.category_column = category_column
        self.aliases = aliases
        self.category_ratios = category_ratios
        self.verbose = verbose

        if self.dataset_numbers is None and self.dataset_ratios is None:
            raise ValueError(
                "dataset_numbers and dataset_ratios cannot be None at the same time. \
                 One has to be specified!"
            )
        if self.dataset_numbers is not None and self.dataset_ratios is not None:
            warnings.warn(
                "Both dataset_numbers and dataset_ratios have been defined. \
                By default only dataset_ratios will be used.",
                UserWarning,
            )
            self.dataset_numbers = None

        self.all_files: list = []
        self.category_set: set = set()
        self.sequence_tracker: dict[str, list[tuple[int, int]]] = {}
        self.sampled_training: dict = {}
        self.sampled_test: dict = {}
        self.sampled_validation: dict = {}
        self.training_data: pd.DataFrame = pd.DataFrame()
        self.test_data: pd.DataFrame = pd.DataFrame()
        self.validation_data: pd.DataFrame = pd.DataFrame()
        self.number_of_sequences: int = 0
        self.sequence_tracker: SequenceTracker = SequenceTracker()
        self.change_dataset_priority = dataset_priority
        self.adjustment_ratio: float = 1.0

    def load_file(self, file_path: Path, overwrite=False) -> pd.DataFrame:
        """
        ## Returns file at file_path as a dataframe
        """
        return pd.read_csv(
            filepath_or_buffer=file_path,
            index_col=0,
            dtype=DTYPE_DICT,
        )

    def save_file(self, file_path: Path, data: pd.DataFrame):
        """
        ## Saves file(s)
        TO DO: SAVE THREE DIFFERENT FILES
        """
        data.to_csv(path_or_buf=file_path)

    def process(self):
        """
        ## Processes files to produce train, test and validation set
        """
        # Get all files to process
        self.get_files_list(self.directory_or_file_path)

        if self.category_column is not None:
            tqdm.write("Collecting categories...")
            self.get_categories_and_sequence_number()
        else:
            # Need at least one category for sequence_tracker
            self.category_set.update("NA")

        # Print categories if verbose
        if self.verbose:
            self.print_data_info()

        if self.verbose:
            tqdm.write("Creating sequence tracker...")
        self.create_sequence_tracker()
        if self.verbose:
            tqdm.write("Sampling training data...")
        self.sample_data(data_type="training")
        if self.verbose:
            tqdm.write("Sampling test data...")
        self.sample_data(data_type="test")
        if self.verbose:
            tqdm.write("Sampling validation data...")
        self.sample_data(data_type="validation")
        return

    def get_files_list(self, directory_or_file_path: Path):
        """
        ## Populates all_files list with file paths
        """
        if directory_or_file_path.is_file():
            self.all_files.append(directory_or_file_path)
        else:
            files = directory_or_file_path.glob("**/*")
            self.all_files.extend(files)

    def get_categories_and_sequence_number(
        self,
    ):
        """
        ## Gets categoy and sequence number information on the dataset
        Calculates the number of sequences and gathers all the categories in the specified column.
        """
        for file in self.all_files:
            # Read file
            data = pd.read_csv(
                filepath_or_buffer=file,
                index_col=0,
                dtype=DTYPE_DICT,
                usecols=[self.category_column, "Chain"],
            )
            # Resetting index to make sure they are unique
            data.reset_index(inplace=True)
            # Grabbing unique categories
            self.category_set.update(list(data[self.category_column].unique()))
            # Calculate number of sequences
            self.number_of_sequences += len(data)

    def print_data_info(self):
        """
        ## Prints information about the dataset
        """
        if self.number_of_sequences < 1:
            for file in self.all_files:
                data = pd.read_csv(
                    filepath_or_buffer=file, dtype=DTYPE_DICT, usecols=[0]
                )
                self.number_of_sequences += len(data)

        tqdm.write("Categories: ", self.category_set)
        tqdm.write("Total number of sequences: ", self.number_of_sequences)

    def create_sequence_tracker(
        self,
    ):
        """
        ## Creates the sequence tracker
        Go through all files and store in what file and at what index do sequences exist.
        Always populate the 'NA' category to keep track of all IDs regardless of category.
        """
        for file_id, file in enumerate(self.all_files):
            # Read file
            data = self.load_file(file)
            data["file_id"] = file_id
            sequence_ids: list[tuple[str, str]] = list(zip(data.file_id, data.index))
            data["Tuple"] = sequence_ids

            self.sequence_tracker.add_default_identities(sequence_ids, "keep")

            # Skip sorting by category if it isn't specified
            if self.category_column is None:
                self.sequence_tracker.categories["NA"].extend(sequence_ids)
                continue

            # Go through all categories and write to the sequence tracker
            for category in self.category_set:
                category_indices = data.Tuple[
                    data[self.category_column] == category
                ].to_list()
                if category_indices:
                    self.sequence_tracker.categories[category].extend(sequence_ids)
                    self.sequence_tracker.categories["NA"].extend(sequence_ids)

    def sample_data(self, data_type: StatusType) -> pd.DataFrame:
        """
        ## Samples data from sequence tracker and packages into dataframes
        --> SEE PAPER SKETCH
        """

        # Sampling by numbers
        if self.dataset_numbers is not None:
            sample_number = round(
                self.dataset_numbers[data_type] * self.adjustment_ratio
            )

        # Sampling by ratio
        else:
            # Re-normalize ratios if necessary
            self.dataset_ratios = self._normalize_ratios(self.dataset_ratios)
            # Get ratio of the provided StatusType
            current_ratio = self.dataset_ratios[data_type]

            # Find numbers of IDs to sample
            total_sequences = len(self.sequence_tracker.categories["NA"])
            sample_number = round(
                total_sequences * current_ratio * self.adjustment_ratio
            )

        # Sampling factory
        sampled_ids = self._sampling_factory(
            mode=self.category_ratios, sample_number=sample_number, data_type=data_type
        )

        # Update Status
        for identifier in sampled_ids:
            current_sequence_status = self.sequence_tracker.identities[identifier]
            current_sequence_status.status = data_type
            # Remove generic category member for easier sampling
            self.sequence_tracker.categories["NA"].remove(identifier)
            """^^^ NOTE: SHOULDN'T I REPLACE THIS WITH A STATUS SWITCH?
                ISSUE: DOESN"T THAT ADD A SAMPLING ISSUE?
                SOLUTION: CREATE A NEW LIST OF THE NON-SAMPLED ID'S, THEN
                SAMPLE FROM THERE!
                >>> TEST AND THEN GENERALIZE THIS CODE
            """

        del self.dataset_ratios[data_type]

    def _sampling_factory(
        self, mode: Union[None, Any], sample_number: int, data_type: StatusType
    ) -> list[SequenceIdType]:
        """
        ## Factory to decide if simple sampling or category sampling
        """
        # Partially apply category sampling if necessary
        if self.change_dataset_priority is not None:
            default_function = partial(
                self._category_sampling, priority=self.change_dataset_priority
            )
        else:
            default_function = self._category_sampling

        # Setup factory
        factory_dictionary = defaultdict(default_function)
        factory_dictionary[None] = self._simple_sampling

        return factory_dictionary[mode](
            sample_number=sample_number, data_type=data_type
        )

    def _simple_sampling(
        self,
        sample_number: int,
        data_type: StatusType,  # pylint: disable=unused-argument
    ) -> list[SequenceIdType]:
        """
        ## Simple sampling
        I.e. any sequence of the tracker is valid
        """
        return random.sample(self.sequence_tracker.categories["NA"], sample_number)

    def _category_sampling(
        self,
        sample_number: int,
        data_type: StatusType,
        priority: priority_type = ("train", "test", "validate"),
    ) -> list[SequenceIdType]:
        """
        ## Category Sampling
        I.e. samples according to the ratios of each category provided.
        If the ratio provided results in a larger number of samples than
        available, the ratios are re-adjusted to the closest possible ratio.
        At low numbers of examples in a given category, the allocation priority
        is by default set as train > test > validation.
        Alternative allocations of priorities is done via functools partial when the
        factory is called.
        """
        # How much to sample per category
        sample_number_per_category = {}

        for key, identities in self.sequence_tracker.categories.items():
            # Find sample number and adjust if necessary
            current_sample_number = round(sample_number * self.category_ratios[key])
            overunder = current_sample_number - len(identities)
            """
            NEXT STEPS:
                - We're now going for a much more simplified version (no more stupid ratios)
                - If overunder is positive or zero:
                    - Check if there are more than 2 examples to allocate into the dataset
                        --> If it is priority 1: give it N-2
                        --> If it is priority 2: give it 1
                        --> If it is priority 3: give it 1
                    - If there are only 2 examples
                        --> Priority 1: 1
                        --> Priority 2: 1
                        --> Priority 3: 0
                    - If there is only 1 example
                        --> Priority 1: 1
                        --> Priority 2,3: 0
                - If overunder is -1:
                    - Priority 1: N
                    - Priority 2: 1
                    - Priority 3: 0
                If overunder is <-1:
                    - Continue
            """

        # TO DO NEXT --> Check sheet

    def _normalize_ratios(self, dictionary: dict[str, float]) -> dict[str, float]:
        """
        ## Normalizes the ratios of a given dictionary
        """
        ratio_sum = sum(dictionary.values())
        if isclose(1.0, ratio_sum, abs_tol=1e-3):
            return dictionary
        else:
            correction = 1 / ratio_sum
            for key, value in dictionary.items():
                dictionary[key] = value * correction
            return dictionary


if __name__ == "__main__":
    data_making_test = DataMaker(
        directory_or_file_path=Path("./"),
        dataset_ratios={"train": 0.8, "test": 0.1, "validate": 0.1},
    )
    tracking = SequenceTracker(
        identities={
            ("test_file1", "1"): SequenceStatus(sequence="A"),
            ("test_file1", "2"): SequenceStatus(sequence="C"),
            ("test_file1", "3"): SequenceStatus(sequence="D"),
            ("test_file1", "4"): SequenceStatus(sequence="E"),
            ("test_file1", "5"): SequenceStatus(sequence="F"),
            ("test_file1", "6"): SequenceStatus(sequence="G"),
            ("test_file1", "7"): SequenceStatus(sequence="H"),
            ("test_file1", "8"): SequenceStatus(sequence="I"),
            ("test_file1", "9"): SequenceStatus(sequence="K"),
            ("test_file1", "0"): SequenceStatus(sequence="L"),
        },
        categories={
            "NA": [
                ("test_file1", "1"),
                ("test_file1", "2"),
                ("test_file1", "3"),
                ("test_file1", "4"),
                ("test_file1", "5"),
                ("test_file1", "6"),
                ("test_file1", "7"),
                ("test_file1", "8"),
                ("test_file1", "9"),
                ("test_file1", "0"),
            ]
        },
    )
    data_making_test.sequence_tracker = tracking
    data_making_test.sample_data(data_type="train")
    data_making_test.sample_data(data_type="test")
    print(data_making_test.sequence_tracker)
