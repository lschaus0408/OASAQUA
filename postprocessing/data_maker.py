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
from collections import defaultdict
from functools import partial

import pandas as pd

from tqdm import tqdm

from postprocessing.post_processing import PostProcessor, DTYPE_DICT
from postprocessing.sequence_tracker import (
    SequenceTracker,
    StatusType,
    SequenceStatus,
    SequenceIdType,
)

DatasetDefinitionType = Literal["train", "test", "validation"]
PriorityType = tuple[DatasetDefinitionType, ...]


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
        dataset_priority: Optional[PriorityType] = ("train", "test", "validation"),
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

        if self.category_ratios is not None:
            self.sampling_mode = "category"
        else:
            self.sampling_mode = None

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

        for dataset in self.change_dataset_priority:
            if (
                dataset in self.dataset_numbers.keys()
                or dataset in self.dataset_ratios.keys()
            ):
                if self.verbose:
                    tqdm.write(f"Sampling {dataset} data...")
                self.sample_data(data_type={dataset})
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
            mode=self.sampling_mode, sample_number=sample_number, data_type=data_type
        )

        # Update Status
        for identifier in sampled_ids:
            current_sequence_status = self.sequence_tracker.identities[identifier]
            current_sequence_status.status = data_type
            # Remove generic category member for easier sampling
            # self.sequence_tracker.categories["NA"].remove(identifier)
            """^^^ NOTE: SHOULDN'T I REPLACE THIS WITH A STATUS SWITCH?
                ISSUE: DOESN"T THAT ADD A SAMPLING ISSUE?
                SOLUTION: CREATE A NEW LIST OF THE NON-SAMPLED ID'S, THEN
                SAMPLE FROM THERE!
                >>> TEST AND THEN GENERALIZE THIS CODE
                >>> len(self.dataset_ratios) is a pitfall with the way it works right now!
            """

        del self.dataset_ratios[data_type]

    def _sampling_factory(
        self, mode: Union[None, Any], sample_number: int, data_type: StatusType
    ) -> list[SequenceIdType]:
        """
        ## Factory to decide if simple sampling or category sampling
        """
        # Category sampling is the default function
        default_function = partial(
            self._category_sampling, sample_number=sample_number, data_type=data_type
        )

        # Setup factory
        factory_dictionary = defaultdict(default_function)
        factory_dictionary[None] = partial(
            self._simple_sampling, sample_number=sample_number, data_type=data_type
        )
        return factory_dictionary[mode]

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
    ) -> list[SequenceIdType]:
        """
        ## Category Sampling
        I.e. samples according to the ratios of each category provided.
        If the ratio provided results in a larger number of samples than
        available, the ratios are re-adjusted to the closest possible ratio.
        """
        if self.dataset_ratios is not None:
            looping_dict = self.dataset_ratios
        else:
            looping_dict = self.dataset_numbers

        # Keep track of how many allocation rounds
        allocation_round = 0
        # Keep track of the ratio allocated previously
        previous_round_allocation = 0

        # Setup defaultdict here to no overwrite later
        category_sample_status = defaultdict(list)

        # Allocate greedily from smallest ratio
        for data_key in self._sorted_keys_by_value(looping_dict):

            # Determine the allocation difference to not oversample
            current_round_allocation = looping_dict[data_key]
            corrected_round_allocaion = (
                current_round_allocation - previous_round_allocation
            )
            previous_round_allocation = looping_dict[data_key]

            # Determine the sampling case we are in
            if self.dataset_ratios is not None:

                # Get the maximum sample size for this allocation round
                maximum_sample_size = round(
                    len(self.sequence_tracker.identities) * corrected_round_allocaion
                )

            else:
                # Get the maximum sample size for this allocation round
                maximum_sample_size = corrected_round_allocaion

            # Iterate through cats for allocation
            for cat_key, cat_value in self.category_ratios.items():

                remaining_number_of_datasets = len(looping_dict) - allocation_round
                # One loop of sampling
                category_sample_status = self._single_loop_category_sampling(
                    category_key=cat_key,
                    category_value=cat_value,
                    category_sample_status_dict=category_sample_status,
                    maximum_sample_size=maximum_sample_size,
                    total_number_of_datasets=remaining_number_of_datasets,
                )
            allocation_round += 1

            # If we reach the desired number of samples, break the loop
            if len(category_sample_status["sampled"]) >= sample_number:
                break

            # Break the loop if we reached the desired data type
            if data_key == data_type:
                break

        return category_sample_status["sampled"]

    def _single_loop_category_sampling(
        self,
        category_key: str,
        category_value: float,
        category_sample_status_dict: dict[str, list[SequenceIdType]],
        maximum_sample_size: int,
        total_number_of_datasets: int,
    ) -> dict[str, list[SequenceIdType]]:
        """
        ## One loop per category for sampling
        """
        # Set of IDs in this category
        category_set = set(self.sequence_tracker.categories[category_key])
        # Remove reserved and sampled from unreserved set
        category_sample_status_dict["unreserved"] = category_set.difference(
            category_sample_status_dict["reserved"],
            category_sample_status_dict["sampled"],
        )

        # Multiply by number of datasets to ensure proper sampling
        category_sample_size = round(
            maximum_sample_size * category_value * total_number_of_datasets
        )

        # Check if this oversamples the category
        total_ids_in_category = len(category_sample_status_dict["unreserved"])
        if category_sample_size > total_ids_in_category:

            # Allocate single
            if total_ids_in_category <= total_number_of_datasets:
                # Make sure that there is something to sample, otherwise return
                if total_ids_in_category == 0:
                    return category_sample_status_dict
                # Sample just one in this case and shuffle around in dataset
                sampled_id = random.sample(
                    list(category_sample_status_dict["unreserved"]), 1
                )
                # Reserved ID allocation
                reserved_ids = set(category_sample_status_dict["unreserved"]) - set(
                    sampled_id
                )
                category_sample_status_dict = self._change_sampling_status(
                    category_sample_status_dict, sampled_id, reserved_ids
                )

            # Allocate the rest
            else:
                quotient, remainder = divmod(
                    total_ids_in_category, total_number_of_datasets
                )
                sampled_id = random.sample(
                    list(category_sample_status_dict["unreserved"]),
                    quotient + remainder,
                )
                # Reserved ID allocation
                reserved_ids = set(category_sample_status_dict["unreserved"]) - set(
                    sampled_id
                )
                category_sample_status_dict = self._change_sampling_status(
                    category_sample_status_dict, sampled_id, reserved_ids
                )

        # If the category isn't oversampled
        else:
            individual_sample_size = round(
                category_sample_size / total_number_of_datasets
            )
            # Sample for this dataset
            sampled_id = random.sample(
                list(category_sample_status_dict["unreserved"]), individual_sample_size
            )
            category_sample_status_dict = self._change_sampling_status(
                category_sample_status_dict=category_sample_status_dict,
                sampled_ids=sampled_id,
            )
            # Reserve for the remaining datasets (Subtract 1 to account for already sampled)
            reserved_ids = random.sample(
                list(category_sample_status_dict["unreserved"]),
                individual_sample_size * (total_number_of_datasets - 1),
            )
            category_sample_status_dict = self._change_sampling_status(
                category_sample_status_dict=category_sample_status_dict,
                reserved_ids=reserved_ids,
            )
        return category_sample_status_dict

    @staticmethod
    def _change_sampling_status(
        category_sample_status_dict: dict[str, list[SequenceIdType]],
        sampled_ids: Optional[list[SequenceIdType]] = None,
        reserved_ids: Optional[list[SequenceIdType]] = None,
    ) -> dict[str, list[SequenceIdType]]:
        """
        ## Changes the status of a category sample set
        """
        # Add IDs to sampled
        if sampled_ids is not None:
            category_sample_status_dict["sampled"].extend(sampled_ids)

        # Add IDs to reserved
        if reserved_ids is not None:
            category_sample_status_dict["reserved"].extend(reserved_ids)

        removal_list = []
        # Removed added IDs from unreserved
        for sequence_id in category_sample_status_dict["unreserved"]:
            if sampled_ids is not None and sequence_id in sampled_ids:
                removal_list.append(sequence_id)
            elif reserved_ids is not None and sequence_id in reserved_ids:
                removal_list.append(sequence_id)

        category_sample_status_dict["unreserved"] = set(
            category_sample_status_dict["unreserved"]
        ) - set(removal_list)
        return category_sample_status_dict

    @staticmethod
    def _sorted_keys_by_value(dictionary_with_ratios: dict[str, float]) -> list:
        """
        ## Returns a list of keys sorted by their value
        """
        sorted_output_keys = sorted(
            dictionary_with_ratios, key=dictionary_with_ratios.get
        )
        return sorted_output_keys

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
        category_column="species",
        category_ratios={
            "human": 0.5,
            "mouse": 0.2,
            "rat": 0.1,
            "rhesus": 0.15,
            "rabbit": 0.05,
        },
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
            ("test_file2", "0"): SequenceStatus(sequence="N"),
            ("test_file2", "1"): SequenceStatus(sequence="AM"),
            ("test_file2", "2"): SequenceStatus(sequence="CM"),
            ("test_file2", "3"): SequenceStatus(sequence="DM"),
            ("test_file2", "4"): SequenceStatus(sequence="EM"),
            ("test_file2", "5"): SequenceStatus(sequence="FM"),
            ("test_file2", "6"): SequenceStatus(sequence="GM"),
            ("test_file2", "7"): SequenceStatus(sequence="HM"),
            ("test_file2", "8"): SequenceStatus(sequence="IM"),
            ("test_file2", "9"): SequenceStatus(sequence="KM"),
            ("test_file3", "0"): SequenceStatus(sequence="NM"),
            ("test_file3", "1"): SequenceStatus(sequence="AMC"),
            ("test_file3", "2"): SequenceStatus(sequence="CMC"),
            ("test_file3", "3"): SequenceStatus(sequence="DMC"),
            ("test_file3", "4"): SequenceStatus(sequence="EMC"),
            ("test_file3", "5"): SequenceStatus(sequence="FMC"),
            ("test_file3", "6"): SequenceStatus(sequence="GMC"),
            ("test_file3", "7"): SequenceStatus(sequence="HMC"),
            ("test_file3", "8"): SequenceStatus(sequence="IMC"),
            ("test_file3", "9"): SequenceStatus(sequence="KMC"),
            ("test_file3", "0"): SequenceStatus(sequence="NMC"),
            ("test_file4", "0"): SequenceStatus(sequence="NM"),
            ("test_file4", "1"): SequenceStatus(sequence="AMCD"),
            ("test_file4", "2"): SequenceStatus(sequence="CMCD"),
            ("test_file4", "3"): SequenceStatus(sequence="DMCD"),
            ("test_file4", "4"): SequenceStatus(sequence="EMCD"),
            ("test_file4", "5"): SequenceStatus(sequence="FMCD"),
            ("test_file4", "6"): SequenceStatus(sequence="GMCD"),
            ("test_file4", "7"): SequenceStatus(sequence="HMCD"),
            ("test_file4", "8"): SequenceStatus(sequence="IMCD"),
            ("test_file4", "9"): SequenceStatus(sequence="KMCD"),
            ("test_file4", "0"): SequenceStatus(sequence="NMCD"),
            ("test_file5", "1"): SequenceStatus(sequence="AMCDE"),
            ("test_file5", "2"): SequenceStatus(sequence="CMCDE"),
            ("test_file5", "3"): SequenceStatus(sequence="DMCDE"),
            ("test_file5", "4"): SequenceStatus(sequence="EMCDE"),
            ("test_file5", "5"): SequenceStatus(sequence="FMCDE"),
            ("test_file5", "6"): SequenceStatus(sequence="GMCDE"),
            ("test_file5", "7"): SequenceStatus(sequence="HMCDE"),
            ("test_file5", "8"): SequenceStatus(sequence="IMCDE"),
            ("test_file5", "9"): SequenceStatus(sequence="KMCDE"),
            ("test_file5", "0"): SequenceStatus(sequence="NMCDE"),
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
                ("test_file2", "0"),
                ("test_file2", "1"),
                ("test_file2", "2"),
                ("test_file2", "3"),
                ("test_file2", "4"),
                ("test_file2", "5"),
                ("test_file2", "6"),
                ("test_file2", "7"),
                ("test_file2", "8"),
                ("test_file2", "9"),
                ("test_file3", "0"),
                ("test_file3", "1"),
                ("test_file3", "2"),
                ("test_file3", "3"),
                ("test_file3", "4"),
                ("test_file3", "5"),
                ("test_file3", "6"),
                ("test_file3", "7"),
                ("test_file3", "8"),
                ("test_file3", "9"),
                ("test_file3", "0"),
                ("test_file4", "0"),
                ("test_file4", "1"),
                ("test_file4", "2"),
                ("test_file4", "3"),
                ("test_file4", "4"),
                ("test_file4", "5"),
                ("test_file4", "6"),
                ("test_file4", "7"),
                ("test_file4", "8"),
                ("test_file4", "9"),
                ("test_file4", "0"),
                ("test_file5", "1"),
                ("test_file5", "2"),
                ("test_file5", "3"),
                ("test_file5", "4"),
                ("test_file5", "5"),
                ("test_file5", "6"),
                ("test_file5", "7"),
                ("test_file5", "8"),
                ("test_file5", "9"),
                ("test_file5", "0"),
            ],
            "human": [
                ("test_file1", "1"),
                ("test_file1", "2"),
                ("test_file1", "3"),
                ("test_file1", "4"),
                ("test_file1", "5"),
                ("test_file1", "6"),
                ("test_file1", "7"),
                ("test_file1", "8"),
                ("test_file1", "9"),
                ("test_file2", "0"),
                ("test_file2", "1"),
                ("test_file2", "2"),
                ("test_file2", "3"),
                ("test_file2", "4"),
                ("test_file2", "5"),
                ("test_file2", "6"),
                ("test_file2", "7"),
                ("test_file2", "8"),
                ("test_file2", "9"),
            ],
            "mouse": [
                ("test_file3", "0"),
                ("test_file3", "1"),
                ("test_file3", "2"),
                ("test_file3", "3"),
                ("test_file3", "4"),
                ("test_file3", "5"),
                ("test_file3", "6"),
                ("test_file3", "7"),
                ("test_file3", "8"),
                ("test_file3", "9"),
                ("test_file3", "0"),
            ],
            "rat": [
                ("test_file4", "0"),
                ("test_file4", "1"),
                ("test_file4", "2"),
                ("test_file4", "3"),
                ("test_file4", "4"),
                ("test_file4", "5"),
                ("test_file4", "6"),
                ("test_file4", "7"),
                ("test_file4", "8"),
                ("test_file4", "9"),
                ("test_file4", "0"),
            ],
            "rhesus": [
                ("test_file5", "1"),
                ("test_file5", "2"),
                ("test_file5", "3"),
                ("test_file5", "4"),
                ("test_file5", "5"),
            ],
            "rabbit": [
                ("test_file5", "6"),
                ("test_file5", "7"),
                ("test_file5", "8"),
                ("test_file5", "9"),
                ("test_file5", "0"),
            ],
        },
    )
    data_making_test.sequence_tracker = tracking
    data_making_test.sample_data(data_type="train")
    data_making_test.sample_data(data_type="test")
    data_making_test.sample_data(data_type="validate")
    training_examples = 0
    for key, value in data_making_test.sequence_tracker.identities.items():
        if value.status == "keep":
            print(key)
            training_examples += 1
    print(training_examples)
