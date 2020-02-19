#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from datastep import Step, log_run_params
from distributed import worker_client

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class MappedRaw(Step):
    @staticmethod
    def _generate_array(i: int, m: int, save_dir: Path) -> Tuple[int, Path]:
        # Generate array
        x = np.random.rand(m, m)

        # Configure save path
        save_dir.mkdir(parents=True, exist_ok=True)
        matrix_save_path = save_dir / f"matrix_{i}.npy"
        np.save(matrix_save_path, x)

        return i, matrix_save_path

    @log_run_params
    def run(self, n: int = 100, m: int = 100, seed: int = 1, **kwargs) -> List[Path]:
        """
        Generates n random arrays of shape (m, m) and saves them to /matrices

        Parameters
        ----------
        n: int
            Number of random arrays to generate.
            Default: 100
        m: int
            Squared shape of the array.
            Default: 100 (100 x 100)
        seed: int
            Seed for numpy's random number generator

        Returns
        -------
        arrays: List[Path]
            The paths to the generated arrays.
        """
        # Configure random seed
        np.random.seed(seed=seed)

        # Storage dir
        matrices_dir = self.step_local_staging_dir / "matrices"

        # Connect to an executor
        with worker_client() as client:
            # Create random arrays
            futures = client.map(
                self._generate_array,
                range(n),
                [m for i in range(n)],  # Must have an arg for every n
                [matrices_dir for i in range(n)],  # Must have an arg for every n
            )

            # Blocking until all are done
            array_infos = client.gather(futures)

        # Configure manifest dataframe for storage tracking
        self.manifest = pd.DataFrame(index=range(n), columns=["filepath"])
        for i, path in array_infos:
            self.manifest.at[i, "filepath"] = path

        # Save the manifest
        self.manifest.to_csv(self.step_local_staging_dir / "manifest.csv", index=False)

        # Return list of paths
        return list(self.manifest["filepath"])
