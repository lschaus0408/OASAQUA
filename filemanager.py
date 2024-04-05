"""
---------------------------------- Observed Antibody Space API -------------------------------------
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics (doi: 10.1002/pro.4205)
-------------------------------------- File Manager Module -------------------------------------------
This module takes a CSVReader object and turns it into a file. The file manager can merge multiple 
objects into one file of a given approximate size. When the file exists and/or an update to the files
is necessary, objects can be written into existing files to the desired size.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from csvreader import CSVReader



@dataclass
class FileManager:
    """ ### FileManager takes paths and CSVReader objects and creates files containing the CSVReader information.
        ## Args:
                \t - path {str} --  Path where the files are to be saved or read from. \n
                                    \tAll files should be in this directory. \n
                                    \tIf the path does not exist, it is created.\n
                \t - filename {str} -- Prefix of the filenames to be created.\n 
                                    \tAll files will have this filename in common with a numbered index suffix.\n
                \t - data {CSVReader} -- CSVReader object that is to be written to a file.\n 
                                    \tIf none is provided, then a file can be loaded to define self.data.\n
    """

    path: Path
    filename: str
    data: Optional[CSVReader] = None #type: CSVReader
    _fileindex: int = 1

    def __add__(self, cls: FileManager) -> pd.DataFrame:
        """ ## Merges two pandas Dataframes from the Filemanager together into one.
            ### Args:
                        \t - cls {FileManager} -- Filemanager object with a CSVReader loaded
            ### Returns:
                        \t - pd.DataFrame -- A Dataframe that should be saved to replace current self.data"""
        return pd.concat([self.data.table, cls.data.table], ignore_index = True)  # type: ignore

    def save_as_csv(self, max_file_size: Optional[int] = None):
        """
            ## Saves the data stored in CSVReader as a csv file if the file is empty.
            If the file contains data then the size is checked. If adding the data
            would surpass filesize, the data is saved into a new file.

            In future the CSVReader should be split such that the files size comes as close
            as possible to the self.filesize.
            ### Args:
                        \t -max_file_size {int} -- Maximum allowed file size in bytes. 
                                                \tIf the file size exceeds this value, then the file is shortened until\n
                                                \tthe final file size is reached. In this case multiple files will be saved\n
                                                \tto disk.\n
            
        """
        # STILL HAVE TO IMPLEMENT THE SPLIT DATASET FUNCTION USING MAX_FILE_SIZE
        assert self.data is not None, "No data as been loaded into self.data"
        path_name = Path(
            str(self.path) + "/" + self.filename + "_" + str(self._fileindex).zfill(5) + ".csv"
        )

        self.data.to_csv(path=path_name) #type: ignore
        self._fileindex += 1

    def load_file(self, path: str, merge: bool = True) -> None:
        """ ## Load existing file to add data.
            If self.data is none, then a CSVReader object is created containing the data found at path.
            Else, the data at path is merged into the current self.data attribute if merge is True.
            If merge is False and self.data is populated, then it overwrites the data in self.data.
            ### Args:
                        \t - path {str} -- Path where the data is located.
                        \t - merge {bool} -- Whether or not to merge the data into the current object. \n
                                            \t Default = True   
        """
        # If merge is True it doesn't matter is self.data is None or not since pd.concat can handle it
        if merge:
            csv_data = CSVReader(path = path)
            csv_data.read_csv(no_header = False)
            self.data = pd.concat([self.data.table, csv_data.table], ignore_index = True)  # type: ignore

        # If merge is False, self.data has to be populated in order to do something
        elif self.data is None and not merge:
            raise ValueError("self.data cannot be None and merge False at the same time")

        # If merge is False and self.data is populated, then overwrite the data in self.data
        elif self.data is not None and not merge:
            self.data = CSVReader(path = path)
            self.data.read_csv(no_header = False)

    def file_size(self, path: Optional[Path] = None) -> int:
        """ 
            ## Returns the filesize of a given file.
            ### Args:
                        \t path {str} -- Path of the file to be evaluated/
            ### Returns:
                        \t {int} -- File size in bytes
        """
        if path is not None:
            return path.stat().st_size

        else:
            return self.path.stat().st_size
