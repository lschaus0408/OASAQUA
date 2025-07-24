"""
---------------------------------- Observed Antibody Space API -----------------------------------
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics (doi:10.1002/pro.4205)
-------------------------------------- OAS Download Module ---------------------------------------
This module should download files from the OAS server. The user can provide keys for the queries and
the module will download all the files provided by that set of keys. The raw data files will be
saved in a temp folder with a naming structure that is recognized by the csvreader module.
"""

import shutil
import re
import warnings

from random import sample
from os import listdir, rename
from os.path import isfile, join
from typing import Optional, Union, Literal
from urllib.error import URLError
from pathlib import Path
from copy import deepcopy

import json
import requests

from tqdm import tqdm
from wget import bar_adaptive, download

from modules.helper_functions import gunzip


class EmptyRequestWarning(Warning):
    """Warning for Empty HTTP Request"""

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class DownloadOAS:
    """
    ## Class that takes as arguments the categories and keys that should be downloaded from OAS.
    When the object is called, the program will download all relevant files and store them in the
    specified directory.
    ### Args:
        \tsearch{Tuple[Tuple[str,str]]} -- Tuple containing a tuple-pair of category and key \n
        \tfile_path{str} -- Path to the OAS_files_dictionary \n
        \tfile_destination{str} -- Destination path for the downloaded files
    ### Returns:
        \tRaw files from the OAS Database, downloaded into the specified folder
    """

    def __init__(
        self,
        file_destination: Path,
        paired: Literal["paired", "unpaired"],
        query_check_file_path: Path = Path("./files/query_check_dictionary.json"),
        search: Optional[tuple[tuple[str, str]]] = None,
        sample_size: Union[None, int] = None,
    ):
        self.query_check_file_path = query_check_file_path
        self.search = search
        self.files = []
        self.file_destination = file_destination
        self.sample_size = sample_size

        if paired == "paired":
            self.paired = True
        else:
            self.paired = False

        if not self.file_destination.is_dir():
            self.file_destination.mkdir()

        # ### CHANGE THIS FOR CONNECTION REQUEST
        assert (
            self.query_check_file_path.is_file()
        ), f"File {self.query_check_file_path} not found."

        with open(self.query_check_file_path, "r", encoding="utf-8") as infile:
            self.dictionary = json.load(infile)

    def __call__(self):
        """
        ## Method that runs the object as intended.
        Grabs the files, makes a list with the union of all OAS files,
        then it creates the 'wget' commands. The final step is to download the files,
        unzip the components and rename the files within the directory,
        followed by the cleanup of the archive files.
        """
        # Make union of files to be downloaded
        self.make_request()

        # If one would like only a sample of the database
        if self.sample_size is not None:
            self.sample()

        # Download all files in the file list
        self.download_files()

        # Unpack the archived filed into the desired folder
        self.unpack()

        # Rename the files to a systematic naming scheme
        self.name_files()

        # Clean up
        self.clean_up()

    def set_search(self, search: list[tuple[tuple[str, str], ...]]) -> None:
        """Setter for search attribute"""
        self.search = search

    def make_request(self) -> None:
        """
        ## Opens multiple files based on the cats/keys and creates a list of files to download.
        This list should not contain any duplicates, so we use sets to create the intersection.
        ### Args: \n
            \tNone \n
        ### Updates: \n
            \tself.files
        """
        if self.paired:
            request_url = "https://opig.stats.ox.ac.uk/webapps/oas/oas_paired/"
        else:
            request_url = "https://opig.stats.ox.ac.uk/webapps/oas/oas_unpaired/"

        if self.search is None:
            raise TypeError("Please set the search terms before using OASDownload.")

        queries = self._create_queries()
        combined_urls = set()
        # Process each query individually
        for query in queries:
            req = requests.post(
                request_url,
                data=query,
                timeout=1,
            )
            # Narrow-down where in the page the URLs are
            page_text = req.text
            start_index = page_text.find("var CSV = [")
            end_index = page_text.find("].join")
            raw_requests = page_text[start_index + 10 : end_index]
            urls = self._extract_urls(raw_requests)
            # Warn if the URL list is empty
            if not urls:
                warnings.warn(
                    f"Query: {query} resulted in an empty request! \
                    \nPlease check if the query was valid.",
                    EmptyRequestWarning,
                )
            combined_urls.update(urls)
        self.files = combined_urls

    def _create_queries(self):
        """
        ## Creates Queries from parsed user-queries
        """
        assert self.search is not None
        queries_list = []
        for pair in self.search:
            # Make sure a category is actually passed
            assert (
                pair[0] is not None
            ), "None passed as category search term. Categories are always needed."
            # Make sure missing query is interpreted as any query
            try:
                value = pair[1]
            except IndexError:
                value = "*"

            # Keep track of previously seen categories
            if any(pair[0] in dictionary for dictionary in queries_list):
                queries_list_extension = []
                for query in queries_list:
                    # Copy each query and change the desired value
                    copied_query = deepcopy(query)
                    copied_query[pair[0]] = value
                    # Make queries list extension only have unique dict entries
                    queries_list_extension.append(copied_query)
                    queries_list_extension = [
                        dict(set_of_dict)
                        for set_of_dict in set(
                            frozenset(dictionary.items())
                            for dictionary in queries_list_extension
                        )
                    ]
                # Extend queries with queries list extension
                queries_list.extend(queries_list_extension)
            else:
                # Create list entry if none exist
                if not queries_list:
                    queries_list.append({pair[0]: value})
                # Add key:value to all existing members
                else:
                    for query in queries_list:
                        query[pair[0]] = value
        return queries_list

    @staticmethod
    def _extract_urls(raw_requests: str):
        """
        ## Extracts URLs from the raw requests
        ### Args:
            \traw_requests{str} -- raw data from HTTP request
        ### Returns:
            \turl_list -- list of URLs
        """
        command_start = [
            message.start() for message in re.finditer("wget", raw_requests)
        ]
        file_suffix = [
            message.start() for message in re.finditer("csv.gz", raw_requests)
        ]
        url_list = []
        for index, command in enumerate(command_start):
            url = raw_requests[command + 5 : file_suffix[index] + 6]
            url_list.append(url)
        return url_list

    def make_shell_file(self, filename: str = "OASdownload.sh") -> None:
        """
        ## Creates a shell file to be executed later from self.files.
        Grabs each element of self.files and uses create_shell_command to create the commands.
        Every command is added to a shell file and saved to self.file_destination.
        I kept this in case someone prefers having the shell files instead of immediately running
        the download inside python.
        ### Args: \n
            \tfilename {str} -- Name of the file to be created
        """
        # Check if file exists. If so, delete the file and create a new one. Else create the file.
        check = Path(str(self.file_destination) + "/" + filename)
        if check.is_file():
            check.unlink()

        with check.open(mode="w", encoding="utf-8") as file:
            for url in self.files:
                file.write("wget " + url)
                file.write("\n")

    def download_files(self):
        """
        ## Excecutes and creates the commands contained in self.files.
        """
        tqdm.write(f"Downloading {len(self.files)} files... \n")
        # Using range to print progess to console
        for index, file in enumerate(self.files):
            # Use wget to download files, bar_adaptive is from the package
            try:
                download(file, out=str(self.file_destination), bar=bar_adaptive)
            except URLError as err:
                print(
                    f"\n WARNING: The following URL is skipped due to an exception occuring: \
                    \n {file}!"
                )
                print(err)
                continue
            # Currently set to every 10th download to not clog the console too much
            if (index + 1) % 10 == 0:
                tqdm.write(f"\n Downloaded {index+1} file(s) out of {len(self.files)}")
        # End message
        tqdm.write("\nFinished Downloading Files")

    def unpack(self, path: Optional[Path] = None) -> None:
        """
        ## Unzips the files at self.file_destination and extracts the information.
        ### Args: \n
            \tpath {Optional[str]} -- Path where the .gz file is located
        """
        # Default scenario leaves path as file_destination (see download_files)
        if path is None:
            path = self.file_destination
        # List all files in the directory that contain a .gz
        filepath = path.glob("*.gz")

        try:
            # Registers the gz unpacker from the helper_functions module
            shutil.register_unpack_format(
                "gz",
                [
                    ".gz",
                ],
                gunzip,
            )
        except shutil.RegistryError as err:
            print(err)

        # Unpack the files
        for file in filepath:
            shutil.unpack_archive(file, path, "gz")

    def name_files(self, prefix: str = "OASQuery") -> None:
        """
        ## Renames the files in self.file_destination to a systematic name.
        ### Args: \n
            \tprefix {str} -- Prefix for the systematic name.
                            All file names will include this prefix with 01,02 etc.
        """
        # Get all the files contained in self.file_destination that have been extracted already
        files = [
            file
            for file in listdir(self.file_destination)
            if isfile(join(self.file_destination, file)) and ".csv.gz" not in file
        ]

        # Iterate over files and rename
        i = 1
        for file in files:
            # Above file list will rename if anything is a non csv.gz file, which is a bug
            if ".csv" in file and "OAS" not in file:
                # zfill(5) just adds leading zeros up to 5 digits
                suffix = str(i).zfill(5)
                new_name = prefix + "_DL" + "_" + suffix + ".csv"
                rename(
                    join(self.file_destination, file),
                    join(self.file_destination, new_name),
                )
                i += 1

    def clean_up(self, archives: bool = True, shell: bool = False) -> None:
        """
        ## Removes the files created in oasdownload.
        In order not to clutter self.file_destination with unnecessary files.
        ### Args: \n
            \tarchives {bool} -- Decides to delete the .csv.gz archive files \n
            \tshell {bool} -- If generated, deletes the shell file
        """
        for file in listdir(self.file_destination):
            if archives:
                # Archive files will have .gz suffix
                if ".gz" in file:
                    # Delete file
                    Path(str(self.file_destination) + "/" + file).unlink()

            elif shell:
                # Shell files will have .sh suffix
                if ".sh" in file:
                    # Delete file
                    Path(str(self.file_destination) + "/" + file).unlink()

    def sample(self) -> None:
        """
        ## Takes a sample from the files attribute.
        ### Updates:
                \tself.files {list[str]} -- Takes a sample of the files and updates it
        """
        # Make sure everything is configures alright
        assert (
            self.sample_size is not None
        ), f"Cannot use sample if sample_size is {self.sample_size}. Value needs to be of type int."
        assert (
            self.files
        ), "Please run make_union first. \
        Cannot run sample without having grabbed the file names first."
        assert isinstance(
            self.sample_size, int
        ), f"sample_size needs to be of type int, currently type is {type(self.sample_size)}"

        # Sample
        self.files = sample(self.files, self.sample_size)


if __name__ == "__main__":
    print("Called OAS Download as main. This does nothing.")
