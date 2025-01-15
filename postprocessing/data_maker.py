"""
---------------------------------- Observed Antibody Space CS -----------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics(doi: 10.1002/pro.4205)\n
------------------------------------------- Data Maker ------------------------------------------\n
Splits the dataset into training, test, and validation sets.
"""

import warnings

from pathlib import Path
from typing import Optional
from math import isclose

import pandas as pd

from tqdm import tqdm

from postprocessing.post_processing import PostProcessor, DTYPE_DICT


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
        \t number_of_sequences {dict[str,int]} -- Optional: Number of sequences in the train,
        test, and validation sets \n
        \t dataset_ratios {dict[str, float]} -- Optional: Data split ratio for train, test,
        and validation sets \n
        \t category_column {str} -- Optional: Column from which to pick categories from \n
        \t aliases {dict} -- Optional: Convert categories to different names \n
        \t category_ratios {dict} -- Optional: Ratio of categories in set \n
        \t verbose {bool} -- Optional: Provides additional information during processing. \n
    Note: Only one of 'number_of_sequences' or 'dataset_ratios' can be passed as a value.
    Passing both, will default to ratios.
    """

    def __init__(
        self,
        directory_or_file_path: Path,
        number_of_sequences: Optional[dict[str, int]] = None,
        dataset_ratios: Optional[dict[str, float]] = None,
        category_column: Optional[str] = None,
        aliases: Optional[dict[str, str]] = None,
        category_ratios: dict[str, float] = None,
        verbose: Optional[bool] = False,
    ):
        self.directory_or_file_path = directory_or_file_path
        self.number_of_sequences = number_of_sequences
        self.dataset_ratios = dataset_ratios
        self.category_column = category_column
        self.aliases = aliases
        self.category_ratios = category_ratios
        self.verbose = verbose

        if self.number_of_sequences is None and self.dataset_ratios is None:
            raise ValueError(
                "number_of_sequences and dataset_ratios cannot be None at the same time. \
                 One has to be specified!"
            )
        if self.number_of_sequences is not None and self.dataset_ratios is not None:
            warnings.warn(
                "Both number_of_sequences and dataset_ratios have been defined. \
                By default only dataset_ratios will be used.",
                UserWarning,
            )
        if self.dataset_ratios is not None:
            sum_to_one = sum(self.dataset_ratios.values())
            assert isclose(1.0, sum_to_one, abs_tol=1e-3)

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

        tqdm.write("Creating sequence tracker...")
        self.create_sequence_tracker()
        tqdm.write("Sampling training data...")
        self.sample_data(data_type="training")
        tqdm.write("Sampling test data...")
        self.sample_data(data_type="test")
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
        """
        for file_id, file in enumerate(self.all_files):
            # Read file
            data = self.load_file(file)
            data["file_id"] = file_id
            data["Tuple"] = list(zip(data.file_id, data.index))

            # Skip sorting by category if it isn't specified
            if self.category_column is None:
                self.write_to_sequence_tracker("NA", data.Tuple.to_list())
                continue

            # Go through all categories and write to the sequence tracker
            for category in self.category_set:
                category_indices = data.Tuple[
                    data[self.category_column] == category
                ].to_list()
                if category_indices:
                    self.write_to_sequence_tracker(
                        category=category, category_indices=category_indices
                    )

    def write_to_sequence_tracker(
        self, category: str, category_indices: tuple[str, int]
    ):
        """
        ## Writes data to sequence tracker
        --> CREATE SEQUENCE TRACKER DATACLASS
        """
        if category in self.sequence_tracker:
            self.sequence_tracker[category].extend(category_indices)
        else:
            self.sequence_tracker[category] = category_indices

    def sample_data(self, data_type: str) -> pd.DataFrame:
        """
        ## Samples data from sequence tracker and packages into dataframes
        MAKE A SCHEME FOR THIS
        TO DO:
            1) If category ratios have been set, determine the optimal ratio given the information in the sequence tracker
            2) Figure out how much to sample for the given data type (i.e. what is the total number of sequences available to sample)
            3) In the sequence tracker, sample the id<->file pairs, for each category, and mark the sampled one's as such
            4) Grab the data from the original files and add them to the data types dataset
        """
