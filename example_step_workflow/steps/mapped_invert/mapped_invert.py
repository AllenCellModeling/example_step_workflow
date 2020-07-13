#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datastep import Step, log_run_params
from distributed import worker_client

from ..mapped_raw import MappedRaw

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class MappedInvert(Step):
    def __init__(self, direct_upstream_tasks: List["Step"] = [MappedRaw]):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks)

    @staticmethod
    def _invert_array(read_path: Union[str, Path], save_dir: Path) -> Tuple[int, Path]:
        if isinstance(read_path, str):
            read_path = Path(read_path)

        # Load matrix
        mat = np.load(read_path)

        # Invert
        inv = np.linalg.inv(mat)

        # Configure save path and save
        save_dir.mkdir(parents=True, exist_ok=True)
        inv_save_path = save_dir / read_path.name
        np.save(inv_save_path, inv)

        # Important:
        # Because we are running in a distributed fashion, we need to track
        # the correct matrices through their processing.
        # We can't just use a random i index, we should grab the index from
        # the filename provided

        # Split the filename by the suffix and the name
        # Then split by the datalabel and the index
        i = int(read_path.name.split(".")[0].split("_")[1])

        return i, inv_save_path

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
            Default: self.step_local_staging_dir.parent / "mappedraw" / manifest.csv
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
            matrices = self.step_local_staging_dir.parent / "mappedraw" / "manifest.csv"

        # Get the matrices from the csv if provided a path
        if isinstance(matrices, (str, Path)):
            # Resolve the filepath and check for existance
            matrices = Path(matrices).resolve(strict=True)

            # Read csv
            raw_data = pd.read_csv(matrices)

            # Convert the specified column into a list of paths
            matrices = [Path(f) for f in raw_data[filepath_column]]

        # Storage dir
        inverted_dir = self.step_local_staging_dir / "inverted"

        # Connect to an executor
        with worker_client() as client:
            # Create random arrays
            futures = client.map(
                self._invert_array,
                matrices,
                [inverted_dir for i in range(len(matrices))],
            )

            # Blocking until all are done
            inversion_infos = client.gather(futures)

        # Configure manifest dataframe for storage tracking
        self.manifest = pd.DataFrame(index=range(len(matrices)), columns=["filepath"])
        for i, path in inversion_infos:
            self.manifest.at[i, "filepath"] = path

        # Save the manifest
        self.manifest.to_csv(self.step_local_staging_dir / "manifest.csv", index=False)

        # Return list of paths
        return list(self.manifest["filepath"])
