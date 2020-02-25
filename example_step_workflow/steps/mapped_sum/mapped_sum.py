#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datastep import Step, log_run_params
from distributed import worker_client

from ..mapped_invert import MappedInvert

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class MappedSum(Step):
    def __init__(self, direct_upstream_tasks: List["Step"] = [MappedInvert]):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks)

    @staticmethod
    def _sum_array(read_path: Path, save_dir: Path) -> Tuple[int, Path]:
        # Load matrix
        mat = np.load(read_path)

        # Sum
        vec = np.amax(mat, 0)
        vec = np.sort(vec)
        vec = np.cumsum(vec)

        # Configure save path and save
        save_dir.mkdir(parents=True, exist_ok=True)
        vec_save_path = save_dir / read_path.name
        np.save(vec_save_path, vec)

        # Important:
        # Because we are running in a distributed fashion, we need to track
        # the correct matrices through their processing.
        # We can't just use a random i index, we should grab the index from
        # the filename provided

        # Split the filename by the suffix and the name
        # Then split by the datalabel and the index
        i = int(read_path.name.split(".")[0].split("_")[1])

        return i, vec_save_path

    @log_run_params
    def run(
        self,
        matrices: Optional[Union[Union[str, Path], List[Path]]] = None,
        filepath_column: str = "filepath",
        **kwargs,
    ) -> List[Path]:
        """
        Sum the list of matrices provided.

        If running in the command line, this will lookup the prior step's produced
        manifest for matrice retrieval. If running in the workflow, uses the direct
        output of the prior step.

        Parameters
        ----------
        matrices: Optional[Union[Union[str, Path], List[Path]]]
            A path to a csv manifest to use or directly a list of paths of serialized
            arrays to sum.
            Default: self.step_local_staging_dir.parent / "mappedinvert" / manifest.csv

        filepath_column: str
            If providing a path to a csv manifest, the column to use for matrices.
            Default: "filepath"

        Returns
        -------
        vectors: List[Path]
            The list of paths to the produced vectors.
        """
        # Default matrices value
        if matrices is None:
            matrices = (
                self.step_local_staging_dir.parent / "mappedinvert" / "manifest.csv"
            )

        # Get the matrices from the csv if provided a path
        if isinstance(matrices, (str, Path)):
            # Resolve the filepath and check for existance
            matrices = Path(matrices).resolve(strict=True)

            # Read csv
            raw_data = pd.read_csv(matrices)

            # Convert the specified column into a list of paths
            matrices = [Path(f) for f in raw_data[filepath_column]]

        # Storage dir
        sum_dir = self.step_local_staging_dir / "sum"

        # Connect to an executor
        with worker_client() as client:
            # Create random arrays
            futures = client.map(
                self._sum_array, matrices, [sum_dir for i in range(len(matrices))],
            )

            # Blocking until all are done
            sum_infos = client.gather(futures)

        # Configure manifest dataframe for storage tracking
        self.manifest = pd.DataFrame(index=range(len(matrices)), columns=["filepath"])
        for i, path in sum_infos:
            self.manifest.at[i, "filepath"] = path

        # Save the manifest
        self.manifest.to_csv(self.step_local_staging_dir / "manifest.csv", index=False)

        # Return list of paths
        return list(self.manifest["filepath"])
