#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datastep import Step, log_run_params
from tqdm import tqdm

from ..sum import Sum

matplotlib.use('agg')
plt.style.use("seaborn-whitegrid")


###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Plot(Step):
    def __init__(self, direct_upstream_tasks: List["Step"] = [Sum]):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks)

    @log_run_params
    def run(
        self,
        vectors: Optional[Union[Union[str, Path], List[Path]]] = None,
        filepath_column: str = "filepath",
        **kwargs,
    ) -> List[Path]:
        """
        Plot and save the list of vectors provided.

        If running in the command line, this will lookup the prior step's produced
        manifest for vector retrieval. If running in the workflow, uses the direct
        output of the prior step.

        Parameters
        ----------
        vectors: Optional[Union[Union[str, Path], List[Path]]]
            A path to a csv manifest to use or directly a list of paths of serialized
            vectors to plot.
            Default: self.step_local_staging_dir.parent / "sum" / manifest.csv

        filepath_column: str
            If providing a path to a csv manifest, the column to use for vectors.
            Default: "filepath"

        Returns
        -------
        plots: List[Path]
            The list of paths to the produced plots.
        """
        # Default vectprs value
        if vectors is None:
            vectors = self.step_local_staging_dir.parent / "sum" / "manifest.csv"

        # Get the matrices from the csv if provided a path
        if isinstance(vectors, (str, Path)):
            # Resolve the filepath and check for existance
            vectors = Path(vectors).resolve(strict=True)

            # Read csv
            raw_data = pd.read_csv(vectors)

            # Convert the specified column into a list of paths
            vectors = [Path(f) for f in raw_data[filepath_column]]

        # Storage dir
        plot_dir = self.step_local_staging_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        # Plot the vectors
        ax = plt.axes()
        for i, vec in tqdm(enumerate(vectors), desc="Plotting vectors"):
            # Load vector
            vec = np.load(vec)

            # Append ax
            ax.plot(vec)

        # Configure manifest dataframe for storage tracking
        self.manifest = pd.DataFrame(index=range(1), columns=["filepath"])

        # Configure save path and save
        plot_save_path = plot_dir / f"plot.png"
        plt.savefig(plot_save_path, format="png")

        # Add the path to manifest
        self.manifest.at[0, "filepath"] = plot_save_path

        # Save the manifest
        self.manifest.to_csv(self.step_local_staging_dir / "manifest.csv", index=False)

        return plot_save_path
