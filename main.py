"""
---------------------------------- Observed Antibody Space API -------------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics (doi: 10.1002/pro.4205)\n
------------------------------------------- OAS Main -----------------------------------------------\n
This module is the main program of OAS API. It manages the communication between the different modules
to create the dataset. It will use oasdownload to download the files, csvreader to process the data and
filemanager to save the data/add data to existing files. Main is also responsible for communicating with
the post-processing tools.\n
"""

import os
import warnings
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Callable, Literal, Optional

from tqdm import tqdm

from csvreader import CSVReader
from filemanager import FileManager
from helper_functions import check_query
from oasdownload import DownloadOAS
from post_processing import PostProcessor

CATEGORY = Literal[
    "Chain",
    "Isotype",
    "Age",
    "Disease",
    "B-Source",
    "B-Type",
    "Longitudinal",
    "Organism",
    "Species",
    "Vaccine",
    "Subject",
]
KEY = Literal[
    "Heavy",
    "Light",
    "Bulk",
    "IGHA",
    "IGHD",
    "IGHE",
    "IGHG",
    "IGHM",
    "Defined",
    "Undefined",
    "HCV",
    "None",
    "SARS-COV-2",
    "SLE",
    "CLL",
    "HIV",
    "MS",
    "Asthma",
    "POEMS",
    "Allergic-Rhinitis-Out-Of-Season",
    "Allergic-Rhinitis-In-Season",
    "EBV",
    "CMV-EBV",
    "MuSK-MG",
    "AChR-MG",
    "Dengue",
    "Non-Dengue-Febrile-Illness",
    "Light-Chain-Amyloidosis",
    "Ebola",
    "Allergy-NoSIT",
    "Obstructive-Sleep-Apnea",
    "Tonsilitis",
    "Tonsilitis-Obstructive-Sleep-Apnea",
    "Healthy-Celiac-Disease",
    "Allergy-SIT",
    "PBMC",
    "Spleen",
    "Bone-Marrow",
    "Lymph",
    "Cerebrospinal-Fluid",
    "Lamina-Propria",
    "Biopsy",
    "Spleen-Bone-Marrow",
    "Biopsy-Small-Intestine",
    "Peritoneal-Cavity",
    "Nasopharyngeal-Swab",
    "PBMC-Nasal-Biopsy",
    "Lung",
    "Jejunum",
    "Ileum",
    "Colon",
    "Mesenteric-Lymph-Node",
    "LeukoPak",
    "Tonsillectomy",
    "Nasal-Biopsy",
    "Cord-Blood-Cells",
    "Cortex",
    "Choroid-Plexus",
    "Brain-Lesion",
    "Cervical-Lymph-Node",
    "Pia-Mater",
    "Unsorted-B-Cells",
    "Naive-B-Cells",
    "Memory-B-Cells",
    "Immature-B-Cells",
    "Plasma-B-Cells",
    "Pre-B-Cells",
    "ASC",
    "Plasmablast",
    "Germinal-Center-B-Cells",
    "Pro-B-Cells",
    "Plasmablast-Plasma-B-Cells",
    "B-1b-Cells",
    "B-2-Cells",
    "B-1a-Cells",
    "FO-Cells",
    "MZ-Cells",
    "RV+B-Cells",
    "Naive-B-Cell-Plasmablast",
    "Human",
    "Rat",
    "Kymouse",
    "Mouse-C57BL-6",
    "Mouse-BALB-c",
    "Mouse",
    "Mouse-Swiss-Webster",
    "Mouse-RAG2-GFP-129Sve",
    "Rabbit",
    "Rhesus",
    "Camel",
    "HIS-Mouse",
    "Rat-SD",
    "None",
    "Flu",
    "Tetanus",
    "HuD",
    "DNP",
    "HepB",
    "OVA",
    "NP-HEL",
    "MenACWY-Polysaccharide",
    "MenACWY-Conjugate",
    "NP-CGG",
    "NP-CGG-Bacterial-Colonization",
    "EColi-Lactobacillus-Clostridia",
    "RSV",
    "TIV",
    "HepB-HepA-Flu",
    "pH1N1",
    "pH1N1-AS03",
    "Tetanus-Flu",
    "Plasmodium",
    "Sheep-Erythrocytes",
]

METADATA = Literal[
    "Run",
    "Link",
    "Author",
    "Species",
    "BSource",
    "BType",
    "Age",
    "Longitudinal",
    "Disease",
    "Vaccine",
    "Subject",
    "Chain",
    "Unique Sequence",
    "Isotype",
    "Total Sequence",
]

DATA = Literal[
    "full",
    "fwr",
    "cdr",
    "junction",
    "np",
    "c_region",
    "sequence_alignment",
    "germline_alignment",
    "v_sequence_alignment",
    "d_sequence_alignment",
    "j_sequence_alignment",
    "v_germline_alignment",
    "d_germline_alignment",
    "j_germline_alignment",
    "locus",
    "v_call",
    "d_call",
    "j_call",
    "v_support",
    "d_support",
    "j_support",
    "v_identity",
    "d_identity",
    "j_identity",
    "locus",
    "stop_codon",
    "vj_in_frame",
    "v_frameshift",
    "productive",
    "rev_comp",
    "complete_vdj",
    "v_alignment_start",
    "d_alignment_start",
    "j_alignment_start",
    "v_alignment_end",
    "d_alignment_end",
    "j_alignment_end",
    "junction_length",
    "junction_aa_length",
    "v_score",
    "d_score",
    "j_score",
    "v_cigar",
    "d_cigar",
    "j_cigar",
    "v_support",
    "d_support",
    "j_support",
    "v_identity",
    "d_identity",
    "j_identity",
    "v_seq_start",
    "d_seq_start",
    "j_sequence_start",
    "v_sequence_end",
    "d_sequence_end",
    "j_sequence_end",
    "v_germline_start",
    "v_germline_end",
    "d_germline_start",
    "d_germline_end",
    "j_germline_start",
    "j_germline_end",
    "fwr1_start",
    "fwr1_end",
    "fwr2_start",
    "fwr2_end",
    "fwr3_start",
    "fwr3_end",
    "fwr4_start",
    "fwr4_end",
    "cdr1_start",
    "cdr1_end",
    "cdr2_start",
    "cdr2_end",
    "cdr3_start",
    "cdr3_end",
    "cdr4_start",
    "cdr4_end",
    "np1_length",
    "np2_length",
    "Redundancy",
    "ANARCI_numbering",
    "ANARCI_status",
]


class API:
    """
    ### Manages all the OAS API modules together in one main class.
    """

    def __init__(
        self,
        filemanager: FileManager,
        csvreader: CSVReader,
        downloader: DownloadOAS,
        query: tuple[tuple[CATEGORY, KEY], ...],
        metadata: list[METADATA],
        data: list[DATA],
        database: Literal["paired", "unpaired"] = "unpaired",
        keep_downloads: Literal["keep", "delete", "move"] = "delete",
    ):
        self.filemanager = filemanager
        self.csvreader = csvreader
        self.downloader = downloader
        self.query = query
        self.database = database
        self.metadata = metadata
        self.data = data
        self.keep_downloads = keep_downloads

    def get_OAS_files(self):
        """
        ## Method that calls DownloadOAS to download the files from the query.
        Check out oasdownload.py for more information on this method.
        """
        # Check if queries are correct
        check_query(database=self.database, query=self.query)

        # Provide the query to the downloader
        self.downloader.set_search(self.query)

        # Download files
        self.downloader()

    def set_database_path(self, custom_path: Optional[str] = None):
        """
        ## Sets the path for the database that is to be accessed.
        ### Args:
                    \t -custom_path {Optional[str]} -- If specified, files dictionary is taken from that file.
        """
        # Set database path
        if custom_path is not None:
            self.downloader.file_path = custom_path
        elif self.database == "unpaired":
            self.downloader.file_path = (
                "./OAS_API/files/OAS_files_dictionary_unpaired.json"
            )
        else:
            self.downloader.file_path = (
                "./OAS_API/files/OAS_files_dictionary_paired.json"
            )

    def get_dataframe(self):
        """
        ## Method that calls csvreader to create the dataframe in the desired format.
        ### Args:
                \t - metadata {List[str]} -- List of metadata keywords to pull from the csv file \n
                \t - data {List[str]} -- List of data keywords to pull from the csv file \n
                \t --> See CSVReader for what keywords are allowed
        ### Updates:
                \t - self.csvreader.table -- A pd.DataFrame with the desired data
        """
        self.csvreader(metadata=self.metadata, data=self.data)

    def set_reader(self) -> None:
        """
        ## Sets a CSVReader to the FileManager as the data attribute. \n
        Only use, once the data has been processed by get_dataframe or by calling CSVReader object.
        """
        self.filemanager.data = self.csvreader

    def save_file(self):
        """
        ## Method that calls the file manager to save the files as csv.
        """
        self.filemanager.save_as_csv()

    def process_folder(
        self,
        mode: Literal["Individual", "Bulk", "Split"] = "Split",
        max_file_size: Optional[int] = 4_000_000_000,
    ):
        """
        ## Method to process the folder of files created by get_OAS_files.
        This method will go through the list of files and process them through CSVReader individually
        and will save them using the file manager. Once two files are created, the file manager will
        check whether merging them will exceed the max_file_size limit. If not they will be merged.
        This will assure that the file will not crash the program once loaded into RAM, as the files
        can become big.
        ### Args:
                    \t mode {str} -- Processing mode to process the folder.\n
                                     Options: \n
                                        'Individual' -- Each file is processed to an individual file. \n
                                        'Bulk' -- Every file is processed into the same file. WARNING: Might cause memory issues. \n
                                        'Split' -- Every file is processed into the same file until a certain file size is reached. Then a new file is created. \n
                    \t max_file_size {int} -- Maximum file size in bytes. Only relevant in 'Split' mode.
        """
        # Factory to select correct folder processing method.
        processors = {
            "Individual": self._individual_processing,
            "Bulk": self._bulk_processing,
            "Split": self._split_processing,
        }
        # Set the method
        processing_mode = processors[mode]

        # Make sure 'Split' mode is used correctly
        if mode == "Split":
            assert (
                max_file_size is not None
            ), f"File size limit needed for 'Split' mode processing. Currently max_file_size is set to {max_file_size}."
            assert (
                isinstance(max_file_size, int) and max_file_size > 0
            ), f"max_file_size needs to be of type int and greater than 0"
            processing_mode(max_file_size=max_file_size)

        processing_mode()

    def _individual_processing(self) -> None:
        """
        ## Processes each downloaded file individually and saves it individually.
        This mode will likely not cause any issues with memory, as the individual files should be small enough.
        """
        # Create list of file names in the directory
        files = self._list_files(self.downloader.file_destination, "DL")

        # Process files
        tqdm.write("Processing files... \n")
        processed_size = 0
        for file in tqdm(files, ascii=" ."):
            self.csvreader.set_path(file)
            self.get_dataframe()
            self.set_reader()
            self.save_file()
            saved_filepath = Path(
                str(self.filemanager.path)
                + "/"
                + self.filemanager.filename
                + "_"
                + str(self.filemanager._fileindex - 1).zfill(5)
                + ".csv"
            )
            processed_size += self.filemanager.file_size(path=saved_filepath)

        tqdm.write(
            f"Finished processing files! All files together are {processed_size/1048576} MB large."
        )

    def _bulk_processing(self) -> None:
        """
        ## Processes each downloaded file and merges them into one file.
        Warning! This mode will cause memory issues if too many files were downloaded.
        Once the processing starts, memory will fill up rapidly due to all the files being processed.
        It is recommended to use 'Split' mode up to the size of the machine's memory, instead of this.
        """
        # Warning again just to make sure the user knows what they are doing
        warnings.warn(
            "This method might make the machine run out of memory!", ResourceWarning
        )

        # NEED TO IMPLEMENT
        raise NotImplementedError("Method has not been implemented yet")

    def _split_processing(self, max_file_size: int) -> None:
        """
        ## Processes each downloaded file and splits them according to the final file size.
        This is the recommended way of processing the data. Default file size if 4GB
        ### Args:
                \t max_file_size {int} -- Maximum file size for the processed file in bytes.
        """
        raise NotImplementedError("Method has not been implemented yet")

    def add_file(self):
        """
        ## Method that calls the file manager and adds new data to the provided csv.
        """
        # NEED TO IMPLEMENT
        raise NotImplementedError("Method has not been implemented yet")

    def process_file(self, processor: PostProcessor):
        """
        ## Method that calls the post processor provided and reprocesses the data.
        """
        # NEED TO IMPLEMENT
        raise NotImplementedError("Method has not been implemented yet")

    def DL_factory(self) -> Callable:
        """
        ## Factory for what to do with the downloaded files after processing.
        Only use after processing files. Otherwise things might break.
        ### Returns:
                    \t{Callable} -- Instance method that takes care of the downloaded files.
        """
        factory = {
            "keep": self._keep_downloads,
            "delete": self._delete_downloads,
            "move": self._move_downloads,
        }
        return factory[self.keep_downloads]

    @staticmethod
    def _keep_downloads() -> None:
        """
        ## Does nothing. Just here to make the types work
        """
        return None

    def _delete_downloads(self) -> None:
        """
        ## Deletes the downloaded files.
        """
        # Create list of file names in the directory
        files = self._list_files(self.downloader.file_destination, "DL")

        for file in files:
            Path(join(self.downloader.file_destination, file)).unlink()

    def _move_downloads(self, new_directory: str = "Downloads") -> None:
        """
        ## Moves files to a 'Downloads' folder.
        """
        files = self._list_files(self.downloader.file_destination, "DL")
        # os.mkdir will throw an error if it already exists
        try:
            os.mkdir(join(self.downloader.file_destination, new_directory))
        # finally to have this step happen either way
        finally:
            for file in files:
                new_path = new_directory + "/" + file
                os.rename(
                    join(self.downloader.file_destination, file),
                    join(self.downloader.file_destination, new_path),
                )

    @staticmethod
    def _list_files(directory: Path, condition: Optional[str] = None) -> list[Path]:
        """
        ## Provides a list of files given the conditions.
        ### Args:
                \tdirectory {str} -- Path of the directory
                \tcondition {str} -- If specified, additional condition in the file name
        ### Returns:
                \t {list[str]} -- List of files
        """
        if condition is None:
            return [
                file for file in directory.iterdir() if file.is_file()
            ]
        else:
            return [
                file
                for file in directory.iterdir()
                if file.is_file() and condition in str(file)
            ]


if __name__ == "__main__":

    def argparse_tuple_type(argument: str) -> tuple:
        """
        ## Takes an argument from argparse and transforms it into a tuple
        """
        argument = argument.replace("(", "").replace(")", "").replace(" ", "")
        argument = argument.split(",")
        mapped_argument = []
        for index in range(int(len(argument) / 2)):
            mapped_argument.append((argument[index * 2], argument[index * 2 + 1]))
        return tuple(mapped_argument)
    
    def argparse_list_type(argument: str) -> list:
        """
        ## Takes an argument from argparse and transforms it into a list
        """
        argument = argument.replace("(", "").replace(")", "").replace(" ", "")
        return argument.split(",")

    import argparse

    parser = argparse.ArgumentParser(
        prog="OAS API",
        description="A downloader, packaging tool, and dataset manager for OAS",
        epilog="For more information on OAS API visit: INSERT URL HERE",
    )
    parser.add_argument(
        "-p",
        "--paired",
        action="store_true",
        help="Download from the paired antibody database instead of the unpaired",
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        help="Output directory for downloaded files",
        default="./oas_api_downloads",
    )
    parser.add_argument("-f", "--filename", help="Prefix for all filenames")
    parser.add_argument(
        "-q",
        "--query",
        type=argparse_tuple_type,
        help="Download queries for OAS API files, see link below to find out what queries are allowed. \
                        Example: ((B-Type, Memory-B-Cells), (Chain, Light)) to query light chains from memory B-Cells",
    )
    parser.add_argument("-m", "--metadata", type=argparse_list_type, help="Determines what metadata from the files to keep. \
                        Example: (Author, Species, Chain)")
    parser.add_argument("-d", "--data", type=argparse_list_type, help="Determines what column data to keep. \
                        Example: (aa_sequence, fwr, cdr, v_call)")
    parser.add_argument("-pm", "--processing_mode", choices=["Individual", "Bulk", "Split"], help="Determines how to process downloaded files. Individual is recommended, see link below for details")
    parser.add_argument("-k", "--keep_downloads", choices=["keep", "delete", "move"], help="Determines what to do with raw files from download.")
    args = parser.parse_args()

    # Set the download file reference paths
    if args.paired:
        dict_path = Path("./OAS_API/files/OAS_files_dictionary_paired.json")
    else:
        dict_path = Path("./OAS_API/files/OAS_files_dictionary_unpaired.json")

    # Set the output directory path
    output_directory = Path(args.output_directory)

    # Set the output filename
    if args.filename is None:
        if args.paired:
            filename = "OAS_Paired_Query"
        else:
            filename = "OAS_Unpaired_Query"
    else:
        filename = args.filename

    data_arguments = args.data
    # Translating argument for OAS API to understand
    data_arguments = ['full' if item == 'aa_sequence' else item for item in data_arguments]

    if args.processing_mode is None:
        processing_mode = "Individual"
    else:
        processing_mode = args.processing_mode

    if args.keep_downloads is None:
        keep_downloads = 'delete'
    else:
        keep_downloads = args.keep_downloads

    reader = CSVReader(output_directory)
    manager = FileManager(
        path=Path(output_directory),
        data=reader,
        filename=filename,
    )
    downloader = DownloadOAS(
        file_path=dict_path,
        file_destination=output_directory,
    )
    api = API(
        filemanager=manager,
        csvreader=reader,
        downloader=downloader,
        query=args.query,
        metadata=args.metadata,
        data=data_arguments,
        keep_downloads=keep_downloads
    )

    api.get_OAS_files()
    api.process_folder(mode=processing_mode)
    api.DL_factory()()
