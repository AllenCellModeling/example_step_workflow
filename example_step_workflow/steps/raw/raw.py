#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from datastep import Step, log_run_params
from tqdm import tqdm

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Raw(Step):
    @log_run_params
    def run(self, n: int = 100, m: int = 100, seed: int = 1, **kwargs) -> List[Path]:
        """
        Generates n random arrays of shape (m, m) and saves them to /matrices
        """
        # Configure random seed
        np.random.seed(seed=seed)

        # Configure manifest dataframe for storage tracking
        self.manifest = pd.DataFrame(index=range(n), columns=["filepath"])

        # Storage dir
        matrices_dir = self.step_local_staging_dir / "matrices"
        matrices_dir.mkdir(exist_ok=True)

        # Generate random arrays
        arrs = []
        for i in tqdm(range(n), desc="Creating and saving matrices"):
            # Generate random m by m array
            x = np.random.rand(m, m)

            # Configure save path and save
            matrix_save_path = matrices_dir / f"matrix_{i}.npy"
            np.save(matrix_save_path, x)

            # Add the path to the manifest
            self.manifest.at[i, "filepath"] = matrix_save_path

            # Append the array save path to the list of arrays
            arrs.append(matrix_save_path)

        # Save the manifest
        self.manifest.to_csv(
            self.step_local_staging_dir / "manifest.csv",
            index=False
        )

        return arrs
