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

from example_step_workflow.steps.fancyplot.plot_utils import gradient_fill

matplotlib.use("agg")
plt.style.use("seaborn-whitegrid")

###############################################################################

log = logging.getLogger(__name__)


###############################################################################


class Fancyplot(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [Sum],
        config: Optional[Union[str, Path]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        vectors: Optional[Union[Union[str, Path], List[Path]]] = None,
        filepath_column: str = "filepath",
        **kwargs,
    ) -> List[Path]:
        """
        Run a pure function.

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
        # Default vectors value
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
        plot_dir = self.step_local_staging_dir / "fancyplots"
        plot_dir.mkdir(exist_ok=True)

        # First make matrix from plotting vectors
        plot_matrix = np.nan
        for i, vec in tqdm(enumerate(vectors), desc="Plotting vectors"):
            # Load vector
            vec = np.load(vec)

            # check length of vector and make plotting matrix
            if np.any(np.isnan(plot_matrix)):
                n = len(vectors)
                m = len(vec)
                plot_matrix = np.zeros([n, m])

            # fill plotting matrix
            plot_matrix[i, :] = vec

        # reorder the matrix
        plot_matrix = plot_matrix[plot_matrix[:, m - 1].argsort()]
        max_pm = np.amax(plot_matrix)

        # Plot the vectors as fancy fills
        fig_fill, ax_fill = plt.subplots()  # the second figure, fancy fill plot
        cmap = plt.cm.get_cmap("gnuplot")
        for i in range(n):
            y = plot_matrix[i, :]
            max_y = np.amax(y)

            # Fancy plotting
            x = np.arange(m)
            lc = cmap(max_y / max_pm)
            gradient_fill(x, y, lc, ax=ax_fill)

        # set axes limits
        plt.sca(ax_fill)
        plt.xlim(1, m)
        plt.ylim(0, np.amax(plot_matrix))

        # Configure manifest dataframe for storage tracking
        self.manifest = pd.DataFrame(index=range(1), columns=["filepath"])

        # Configure save path and save
        plot_save_path = plot_dir / f"plot_fancy.png"
        plt.sca(ax_fill)
        plt.savefig(plot_save_path, format="png")

        # Add the path to manifest
        self.manifest.at[0, "filepath"] = plot_save_path

        # Save the manifest
        self.manifest.to_csv(self.step_local_staging_dir / "manifest.csv", index=False)

        return plot_save_path
