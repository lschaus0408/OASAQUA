""" 
---------------------------------- Observed Antibody Space API -------------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics (doi: 10.1002/pro.4205)\n
-------------------------------------- OAS Download Module -------------------------------------------\n
This module should download files from the OAS server. The user can provide keys for the queries and 
the module will download all the files provided by that set of keys. The raw data files will be saved 
in a temp folder with a naming structure that is recognized by the csvreader module.\n
"""

from random import sample
import shutil

from os import listdir, rename
from os.path import isfile, join
from typing import Optional, Union

from tqdm import tqdm

import json
from urllib.error import URLError
from pathlib import Path

from wget import bar_adaptive, download

from OAS_API.helper_functions import gunzip


class DownloadOAS:
    """
    ## Class that takes as arguments the categories and keys that should be downloaded from OAS.
    When the object is called, the program will download all relevant files and store them in the
    specified directory.
    ### Args:
        \tsearch {Tuple[Tuple[str,str]]} -- Tuple containing a tuple-pair of category and key keywords \n
        \tfile_path {str} -- Path to the OAS_files_dictionary \n
        \tfile_destination {str} -- Destination path for the downloaded files
    ### Returns:
        \tRaw files from the OAS Database, downloaded into the specified folder
    """

    def __init__(
        self,
        file_destination: Path,
        file_path: Path = Path(""),
        search: Optional[tuple[tuple[str, str]]] = None,
        sample_size: Union[None, int] = None,
    ):
        self.file_path = file_path
        self.search = search
        self.files = []
        self.file_destination = file_destination
        self.sample_size = sample_size

        if not self.file_destination.is_dir():
            self.file_destination.mkdir()

        assert self.file_path.is_file(), f"File {self.file_path} not found."

        with open(self.file_path, "r") as infile:
            self.dictionary = json.load(infile)

    def __call__(self):
        """
        ## Method that runs the object as intended.
        Grabs the files, makes a list with the union of all OAS files,
        then it creates the 'wget' commands. The final step is to download the files, unzip the components
        and rename the files within the directory, followed by the cleanup of the archive files.
        """
        # Make union of files to be downloaded
        self.make_union()

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

    def set_search(self, search: tuple[tuple[str, str]]) -> None:
        self.search = search

    def make_union(self) -> None:
        """
        ## Opens multiple files based on the cats/keys and creates a list of files to download.
        This list should not contain any duplicates, so we use sets to create the intersection.
        ### Args: \n
            \tNone \n
        ### Updates: \n
            \tself.files
        """
        if self.search is None:
            raise TypeError("Please set the search terms before using OASDownload.")
        # Grab a (cat, key) pair
        for pair in self.search:
            # Make sure a cat is actually passed
            assert (
                pair[0] is not None
            ), "None passed as category search term. Categories are always needed."
            try:
                value = pair[1]
            except IndexError:
                value = None
            # If a key has been passed, then use the key
            if value is not None:
                file_list = self._find_data(
                    key=pair[1], values=self.dictionary[pair[0]]
                )
                # Update the main list with intersection of both
                if self.files:
                    self.files = list(set(self.files) & set(file_list))
                    assert self.files, f"Query combination {pair} with previous queries returns an empty list"
                else:
                    self.files = file_list

            # If no key has been passed, use all the keys under the passed cat
            else:
                file_list = [element[1] for element in self.dictionary[pair[0]]]
                # Flatten list
                file_list = [item for sublist in file_list for item in sublist]
                # Update the main list with intersection of both
                if self.files:
                    self.files = list(set(self.files) & set(file_list))
                    assert self.files, f"Query combination {pair} with previous queries returns an empty list"
                else:
                    self.files = file_list

    def _find_data(self, key: str, values: list) -> list[str]:
        """
        ## Private function only to be used by make_union.
        Iterates through values to find the key and returns the list of data paired with that key.
        ### Args: \n
            \tkey {str} -- Key that defines the data \n
            \tvalues {list} --  List containind the key/data pairs
        ### Returns:\n
            \tlist[str] -- Desired data
        """
        for i in values:
            if i[0] == key:
                return i[1]

        raise AssertionError(f"Key {key} not in the provided dataset")

    def create_url(self, url: str) -> str:
        """
        ## Creates a shell command given a file name that one wants to download.
        OAS is divided into two databases: OAS Paired and OAS Unpaired.
        We use the json file to determine whether the download is performed from either of the
        databases.
        ### Args: \n
            \tcmd {str} -- String that is to be converted in to a shell command
        ### Returns: \n
            \tstr -- String converted to a shell command
        """
        # Figure out if the database to be called is OAS Paired or OAS Unpaired
        if "_unpaired." in str(self.file_path):
            url = "http://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired" + url

        elif "_paired." in str(self.file_path):
            url = "http://opig.stats.ox.ac.uk/webapps/ngsdb/paired" + url

        else:
            raise ValueError(
                f"Could not resolve the OAS database from the provided json file or string. {self.file_path} contains neither keywords '_paired.' or '_unpaired.'!"
            )

        return url

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

        with check.open(mode="w") as f:
            for dataset in self.files:
                f.write(self.create_url(dataset))
                f.write("\n")

    def download_files(self):
        """
        ## Excecutes and creates the commands contained in self.files.
        """
        tqdm.write(f"Downloading {len(self.files)} files... \n")
        # Using range to print progess to console
        for i in range(len(self.files)):
            url = self.create_url(self.files[i])
            # Use wget to download files, bar_adaptive is from the package
            try:
                download(url, out=str(self.file_destination), bar=bar_adaptive)
            except URLError as err:
                print(
                    f"\n WARNING: The following URL is skipped due to an exception occuring: \n {url}!"
                )
                print(err)
                continue
            # Currently set to every 10th download to not clog the console too much
            if (i + 1) % 10 == 0:
                tqdm.write(f"\n Downloaded {i+1} file(s) out of {len(self.files)}")
        # End message
        tqdm.write("\nFinished Downloading Files")

    def unpack(self, path: Optional[str] = None) -> None:
        """
        ## Unzips the files at self.file_destination and extracts the information.
        ### Args: \n
            \tpath {Optional[str]} -- Path where the .gz file is located
        """
        # Default scenario leaves path as file_destination (see download_files)
        if path is None:
            path = self.file_destination
        # List all files in the directory that contain a .gz
        p = Path(path).glob("*.gz")

        try:
            # Registers the gz unpacker from the helper_functions module
            shutil.register_unpack_format(
                "gz",
                [
                    ".gz",
                ],
                gunzip,
            )
        except:
            pass

        # Unpack the files
        for file in p:
            shutil.unpack_archive(file, path, "gz")

    def name_files(self, prefix: str = "OASQuery") -> None:
        """
        ## Renames the files in self.file_destination to a systematic name for csv reader to understand.
        ### Args: \n
            \tprefix {str} -- Prefix for the systematic name. All file names will include this prefix with 01,02 etc.
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
            # Above file list will rename if anything is a non csv.gz file, which is a bug. This is a quick fix
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
        ), "Please run make_union first. Cannot run sample without having grabbed the file names first."
        assert isinstance(
            self.sample_size, int
        ), f"sample_size needs to be of type int, currently type is {type(self.sample_size)}"

        # Sample
        self.files = sample(self.files, self.sample_size)


if __name__ == "__main__":
    print("Called OAS Download as main. This does nothing.")
