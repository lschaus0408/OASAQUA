"""
---------------------------------- Observed Antibody Space CS -----------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics(doi: 10.1002/pro.4205)\n
------------------------------------------- Cluster ----------------------------------------------\n
Clusters Sequences of OAS CS files using fastBCR (doi: 10.1016/j.crmeth.2023.100601). Due to the
possibly very large sequence datasets that one can pull from OAS, fastBCR seems to be the best
clustering algorithm as of 2024 for clustering antibody sequences by their clonal family. It is
recommended to run the Cluster post-processing after removing redundant sequences,
filtering by length, and checking for sequence viability. Other clustering, such as Linclust can
be implemented upon demand.
"""

import tempfile
import subprocess
import ast

from pathlib import Path
from typing import Optional, Literal

import pandas as pd

from postprocessing.post_processing import PostProcessor, DTYPE_DICT
from postprocessing.sequence_tracker import SequenceIdType, SequenceTracker, StatusType


class Cluster(PostProcessor):
    """
    ## Cluster Postprocessor
    Requires the installation of R and fastBCR by the user.
    Please check out the github for instructions:
    https://github.com/ZhangLabTJU/fastBCR/
    """

    FBCR_COLUMNS = [
        "Species",
        "cdr3",
        "cdr3_aa",
        "v_call",
        "j_call",
        "c_call",
        "junction",
        "junction_aa",
    ]

    FBCR_MANDATORY = {"v_call", "j_call", "junction_aa"}

    BASE_DIRECTORY = Path("./R_scripts")

    RENAMING_EXCEPTIONS = ["Species"]

    def __init__(
        self,
        directory_or_file_path: Path,
        output_directory: Path,
        path_to_fastbcr: Optional[Path] = None,
        paired: bool = False,
        chain_type: Optional[Literal["heavy", "light"]] = None,
        min_depth: int = 3,
        max_depth: int = 1000,
        overlap_threshold: float = 0.1,
        consensus_threshold: float = 0.8,
        sample_per_cluster: int = 1,
        maximum_file_size_gb: Optional[float] = None,
        scramble_data_before_cluster: bool = True,
    ):
        super().__init__(
            directory_or_file_path=directory_or_file_path,
            output_directory=output_directory,
        )

        # Make sure fastBCR is installed
        self.path_to_fastbcr = path_to_fastbcr
        self.paired = paired
        if not paired:
            assert (
                chain_type is not None
            ), f"Chain type needs to be specified \
            if paired is False. Current chain_type: {chain_type}"
            assert chain_type in [
                "heavy",
                "light",
            ], f"Chain type needs to be 'heavy' or 'light', not: {chain_type}"
            self.chain_type = chain_type
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.overlap_threshold = overlap_threshold
        self.consensus_threshold = consensus_threshold
        self.sample_per_cluster = sample_per_cluster
        self.sequence_tracker: SequenceTracker = SequenceTracker()
        self.species_set: set = set()
        self.maximum_file_size = maximum_file_size_gb
        self.scramble_data_before_cluster = scramble_data_before_cluster

    def load_file(self, file_path: Path, overwrite=False):
        """
        ## Returns file at file_path as a dataframe
        """
        # Check the file header
        data_header = pd.read_csv(filepath_or_buffer=file_path, nrows=0)
        column_names_header = list(data_header.columns)

        # Get columns for fastBCR
        load_columns = [
            column for column in column_names_header if column in self.FBCR_COLUMNS
        ]

        # Check if mandatory items are present
        assert self.FBCR_MANDATORY.issubset(
            load_columns
        ), f"Mandatory columns need to be present in the data for fastBCR to work: \
        {self.FBCR_MANDATORY}"

        data = pd.read_csv(
            filepath_or_buffer=file_path,
            dtype=DTYPE_DICT,
            usecols=load_columns,
        )
        return data

    def save_file(self, file_path: Path, data: pd.DataFrame):
        """
        ### Save File
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path_or_buf=file_path)

    def process(self):
        """
        ### Cluster Sequences
        """
        # Load Files
        self.get_files_list(self.directory_or_file_path)

        # Create Sequence Tracker
        self.create_sequence_tracker()

        # Create temp directory for fastbcr
        with tempfile.TemporaryDirectory(dir=self.BASE_DIRECTORY) as fastbcr_dir:

            # Separate Files by Species
            for species, identifiers in self.sequence_tracker.categories.items():
                # Sort identifiers by file
                identifiers.sort(key=lambda item: item[0])
                data = self.assemble_files(id_list=identifiers, species_id=species)
                # Scamble data if desired
                if self.scramble_data_before_cluster:
                    data = data.sample(frac=1).reset_index(drop=True)
                # Save file in folder for fastBCR
                data_path = Path(fastbcr_dir) / f"fastbcr_{species}_temp.csv"
                self.save_file(file_path=data_path, data=data)
            # Run fastBCR
            self.run_fastbcr(tempdir=fastbcr_dir)

            # Load output
            temp_files = Path(fastbcr_dir).glob("**/*")
            clonotype_files = [file for file in temp_files if "clonotype" in file.name]
            sampled_ids = self.sample_clonotypes(files=clonotype_files)
            # Load originals
            self.sequence_tracker.update_status(new_status=sampled_ids)
            file_assembly_data = self.sequence_tracker.restructure_data()
            self.assemble_datasets_and_save(assembly_datastructure=file_assembly_data)

    def create_sequence_tracker(
        self,
    ):
        """
        ## Creates the Sequence Tracker
        """
        for file_id, file in enumerate(self.all_files):
            data = pd.read_csv(
                filepath_or_buffer=file,
                # index_col=0,
                dtype=DTYPE_DICT,
                usecols=["Species"],
            )

            data["file_id"] = file_id
            sequence_ids: list[SequenceIdType] = list(zip(data.file_id, data.index))
            data["sequence_id"] = sequence_ids

            # Setup sequence tracker
            self.sequence_tracker.add_default_identities(sequence_ids, "keep")

            # Setup Species Category for Sequence Tracker
            self.species_set.update(list(data["Species"].unique()))

            for species in self.species_set:
                species_ids = data["sequence_id"][data["Species"] == species].to_list()
                if species_ids:
                    self.sequence_tracker.categories[species].extend(sequence_ids)

    def assemble_files(
        self, id_list: list[SequenceIdType], species_id: str
    ) -> pd.DataFrame:
        """
        ## Creates one file for usage in fastBCR out of the id_list
        """
        filtered_data = None
        seen_files = set()

        for file_id, _ in id_list:
            if file_id in seen_files:
                continue
            else:
                seen_files.add(file_id)
            unfiltered_data = self.load_file(file_path=self.all_files[file_id])
            current_id_set = [id_pair for id_pair in id_list if id_pair[0] == file_id]
            unfiltered_data["sequence_id"] = current_id_set
            # Create dataframe if it doesn't exist
            if filtered_data is None:
                filtered_data = unfiltered_data.loc[
                    unfiltered_data["Species"] == species_id
                ]
            else:
                filtered_data = pd.concat(
                    [
                        unfiltered_data.loc[unfiltered_data["Species"] == species_id],
                        filtered_data,
                    ],
                    ignore_index=True,
                )
        return filtered_data

    def run_fastbcr(self, tempdir: Path):
        """
        ## Runs fastBCR via Rscript
        """
        command_list = [
            "Rscript",
            "./R_scripts/run_fastBCR.R",
            "-d",
            str(tempdir),
            "-c",
            str(self.min_depth),
            "-x",
            str(self.max_depth),
            "-o",
            str(self.overlap_threshold),
            "-n",
            str(self.consensus_threshold),
        ]
        if self.paired:
            command_list.append("-p")
            subprocess.run(
                command_list,
                check=True,
            )
        else:
            subprocess.run(
                command_list,
                check=True,
            )

    def sample_clonotypes(self, files: list[Path]) -> dict[SequenceIdType, StatusType]:
        """
        ## From fastBCR output files, samples IDs
        """
        samples = []
        for file in files:
            data = pd.read_csv(
                filepath_or_buffer=file,
                usecols=["clonotype_index", "sequence_id", "raw_index", "raw_indices"],
            )
            try:
                dataframe_sample = data.sample(n=self.sample_per_cluster)
            except ValueError:
                print(
                    f"Couldn't sample {self.sample_per_cluster} for this clonotype, defaulting to 1"
                )
                dataframe_sample = data.sample(n=1)
            samples.append(dataframe_sample)
        all_samples = pd.concat(samples, ignore_index=True)
        # Since read_csv messes up the dtypes, need to adjust the type here
        all_samples["sequence_id"] = all_samples["sequence_id"].apply(ast.literal_eval)
        return dict.fromkeys(all_samples["sequence_id"].tolist(), "sampled")

    def assemble_datasets_and_save(
        self,
        assembly_datastructure: dict[str, dict[str, list]],
    ) -> None:
        """
        ## Assembles datasets into DataFrames
        This method also saves the dataframe as a csv if it exceeds
        the maximum_file_size threshold.
        """

        def save_if_exceeds_size(
            data: list[pd.DataFrame], name: str = "fastBCR_Cluster"
        ) -> list[pd.DataFrame]:
            """
            ## Saves file if it exceeds the maximum file size
            """
            # Create dataframe
            current_dataframe = pd.concat(data, ignore_index=True)
            pd_type_converter = {
                key: DTYPE_DICT[key] for key in current_dataframe.columns
            }
            current_dataframe = current_dataframe.astype(dtype=pd_type_converter)
            # Get size in GB
            current_dataframe_size = current_dataframe.memory_usage(deep=True).sum() / (
                1024**3
            )
            if (
                self.maximum_file_size is not None
                and current_dataframe_size >= self.maximum_file_size
            ):
                output_file_name = Path(
                    self.output_directory,
                    f"{name}_Chunk_{save_iteration}.csv",
                )
                self.save_file(file_path=output_file_name, data=current_dataframe)
                return []
            # Return original if it doesnt exceed size
            return data

        data_list = []
        save_iteration = 0

        # Open each file in order
        for file_index, dataset in assembly_datastructure.items():
            data = pd.read_csv(
                self.all_files[file_index], dtype=DTYPE_DICT, index_col=0
            )
            # Add rows from the dataset to the dict according to the key
            for _, dataset_value in dataset.items():
                data_list.append(data.iloc[dataset_value])

                data_list = save_if_exceeds_size(data=data_list)
                if not data_list:
                    save_iteration += 1

        # Finally create remaining dataframes and save
        if data_list:
            last_dataframe = pd.concat(data_list, ignore_index=True)
            output_file_name = Path(
                self.output_directory, f"fastBCR_Cluster_chunk_{save_iteration}.csv"
            )
            self.save_file(file_path=output_file_name, data=last_dataframe)
