"""
---------------------------------- Observed Antibody Space CS -----------------------------------\n
Module can download different datasets from the OAS and write them into a compressed file for local
storage. Files can be separated into paired and unpaired sequences as well as by their antibody
region. OAS is a database from the University of Oxford Dept. of Statistics(doi: 10.1002/pro.4205)\n
--------------------------------------- Encode via ESM ------------------------------------------\n
Encodes antibody sequences with the Protein Language Model 'Evolutionary Scale Modelling 2' (ESM2).
(doi: 10.1126/science.ade2574). Contrary to other post-processing modules, this encoding-based
post-processor stores the encodings in a separate file, but in the same order as observed in the
original file.
"""

from pathlib import Path
from typing import Literal, Optional, TypeAlias, get_args
from os import environ

import torch

import numpy as np
import numpy.typing as npt

from postprocessing.encoder import Encoder
from postprocessing.sequence_tracker import SequenceIdType, sequence_id_to_str


class EncodeESM(Encoder):
    """
    ## PLM Encoder Using ESM
    Encodes Sequences using ESM2 protein language model.
    The specific model of ESM can be specified.
    """

    ESM_MODELS: TypeAlias = Literal[
        "esm2_t48_15B_UR50D",
        "esm2_t36_3B_UR50D",
        "esm2_t33_650M_UR50D",
        "esm2_t30_150M_UR50D",
        "esm2_t12_35M_UR50D",
        "esm2_t6_8M_UR50D",
    ]

    def __init__(
        self,
        directory_or_file_path: Path,
        output_directory: Path,
        pad_size: int,
        save_format: Literal["json", "numpy"],
        category_column: Optional[str] = None,
        n_jobs: int = 1,
        maximum_file_size_gb: Optional[int] = None,
        flatten: bool = False,
        output_file_prefix: str = "esm_encoded_sequences",
        model: ESM_MODELS = "esm2_t6_8M_UR50D",
    ):
        super().__init__(
            directory_or_file_path=directory_or_file_path,
            output_directory=output_directory,
            pad_size=pad_size,
            save_format=save_format,
            category_column=category_column,
            n_jobs=n_jobs,
            maximum_file_size_gb=maximum_file_size_gb,
            flatten=flatten,
            output_file_prefix=output_file_prefix,
        )

        assert model in get_args(
            self.ESM_MODELS
        ), f"{model} is not a valid model. \
            Please provide a valid ESM-2 model. Visit the ESM github for a list of models."

        self.model, self.alphabet = torch.hub.load("facebookresearch/esm:main", model)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        self.representation_lenght = self._find_representation_layer_length(model=model)

        # In the case that multiple jobs are used, use CPU for ESM
        if self.n_jobs > 1:
            environ["CUDA_VISIBLE_DEVICES"] = ""

    def _encode_single_process(
        self, sequences: dict[SequenceIdType, npt.NDArray[np.str_]]
    ) -> dict[SequenceIdType, npt.NDArray[np.uint8]]:
        """
        ## Encodes via ESM encoder
        """
        encoded_sequences: dict[SequenceIdType, npt.NDArray[np.float32]] = {}
        data_batch = []
        # Generate data batch
        for sequence_id, sequence in sequences.items():
            # Convert sequence ID to a string
            index = sequence_id_to_str(sequence_id=sequence_id)
            data_batch.append((f"Protein_{index}", sequence))
        _, __, batch_tokens = self.batch_converter(data_batch)

        # Extract per-residue representations
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.representation_lenght])
        token_representations = results["representations"][self.representation_lenght]

        # Convert data to numpy
        token_representations = token_representations.numpy()

        # Flatten
        if self.flatten:
            token_representations = self._flatten_arrays(token_representations)

        # Convert to internal data structure
        encoded_sequences = self._convert_to_dict(
            data_batch=data_batch, token_representations=token_representations
        )

        return encoded_sequences

    def _convert_to_dict(
        self, data_batch: list[tuple[str, str]], token_representations: npt.NDArray
    ) -> dict[SequenceIdType, npt.NDArray[np.float32]]:
        """
        ## Converts token representations to a dict with SequenceID
        """
        output_dict = {}
        for index, data in enumerate(data_batch):
            batch_id = data[0]
            # Split by _
            parts_batch_id = batch_id.split("_")
            file_id, sequence_id = parts_batch_id[1], parts_batch_id[2]
            # Create datastructure
            output_dict[(file_id, sequence_id)] = token_representations[index]
        return output_dict

    @staticmethod
    def _find_representation_layer_length(model: ESM_MODELS) -> int:
        """
        ## Based on the input model, finds the representation layer lenght
        """
        start_index = model.find("t")
        end_index = model.find("_", start_index)
        representation_lenght = int(model[start_index + 1 : end_index])
        return representation_lenght
