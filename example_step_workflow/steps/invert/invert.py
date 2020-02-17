#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from datastep import Step, log_run_params
from tqdm import tqdm

from ..raw import Raw

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Invert(Step):
    def __init__(self, direct_upstream_tasks: List["Step"] = [Raw]):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks)

    @log_run_params
    def run(
        self,
        matrices: Optional[Union[Union[str, Path], List[Path]]] = None,
        filepath_column: str = "filepath",
        **kwargs
    ) -> List[Path]:
        """
        Invert the list of matrices provided.

        If running in the command line, this will lookup the prior step's produced
        manifest for matrice retrieval. If running in the workflow, uses the direct
        output of the prior step.

        Parameters
        ----------
        matrices: Optional[Union[Union[str, Path], List[Path]]]
            A path to a csv manifest to use or directly a list of paths of serialized
            arrays to invert.
            Default: self.step_local_staging_dir.parent / "raw" / manifest.csv
        filepath_column: str
            If providing a path to a csv manifest, the column to use for matrices.
            Default: "filepath"

        Returns
        -------
        inverted: List[Path]
            The list of paths to the inverted matrices.
        """
        # Default matrices value
        if matrices is None:
            matrices = self.step_local_staging_dir.parent / "raw" / "manifest.csv"

        # Get the matrices from the csv if provided a path
        if isinstance(matrices, (str, Path)):
            # Resolve the filepath and check for existance
            matrices = Path(matrices).resolve(strict=True)

            # Read csv
            raw_data = pd.read_csv(matrices)

            # Convert the specified column into a list of paths
            matrices = [Path(f) for f in raw_data[filepath_column]]

        # Configure manifest dataframe for storage tracking
        self.manifest = pd.DataFrame(index=range(len(matrices)), columns=["filepath"])

        # Storage dir
        inverted_dir = self.step_local_staging_dir / "inverted"
        inverted_dir.mkdir(exist_ok=True)

        # Invert the matrices
        inversions = []
        for i, matrix in tqdm(
            enumerate(matrices), desc="Loading, inverting and saving matrices"
        ):
            # Load matrix
            mat = np.load(matrix)

            # Invert
            inv = np.linalg.inv(mat)

            # Configure save path and save
            inv_save_path = inverted_dir / matrix.name
            np.save(inv_save_path, inv)

            # Add the path to manifest
            self.manifest.at[i, "filepath"] = inv_save_path

            # Append the inversion save path to the list of inversions
            inversions.append(inv_save_path)

        # Save the manifest
        self.manifest.to_csv(self.step_local_staging_dir / "manifest.csv", index=False)

        return inversions
