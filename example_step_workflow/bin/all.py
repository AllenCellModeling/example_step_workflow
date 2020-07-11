#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script will run all tasks in a prefect Flow.

When you add steps to you step workflow be sure to add them to the step list
and configure their IO in the `run` function.
"""

import logging
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, List, NamedTuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask_jobqueue import SLURMCluster
from distributed import LocalCluster
from prefect import Flow, task, unmapped
from prefect.engine.executors import DaskExecutor
from prefect.engine.results import LocalResult
from prefect.engine.serializers import Serializer
from tqdm import tqdm

matplotlib.use("agg")
plt.style.use("seaborn-whitegrid")

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class ArrayManager(NamedTuple):
    index: int
    arr: np.ndarray


@task(
    result=LocalResult(dir="local_staging/raw/"),
    target=lambda **kwargs: "{}.npy".format(kwargs["n"]),
)
def generate_array(n: int) -> ArrayManager:
    return ArrayManager(n, np.random.rand(n, n))


@task(
    result=LocalResult(dir="local_staging/inv/"),
    target=lambda **kwargs: "{}.npy".format(kwargs.get("am").index),
)
def invert_array(am: ArrayManager) -> ArrayManager:
    return ArrayManager(am.index, np.linalg.inv(am.arr))


@task(
    result=LocalResult(dir="local_staging/sum/"),
    target=lambda **kwargs: "{}.npy".format(kwargs.get("am").index),
)
def sum_array(am: ArrayManager) -> ArrayManager:
    vec = np.amax(am.arr, 0)
    vec = np.sort(vec)
    vec = np.cumsum(vec)

    return ArrayManager(am.index, vec)


@task
def basic_plot(ams: List[ArrayManager]):
    # Plot the vectors as red lines
    plt.figure(figsize=(10, 10))
    for i, am in tqdm(enumerate(ams), desc="Plotting vectors"):
        plt.plot(np.linspace(0, 1, am.arr.shape[0]), am.arr)

    # Save
    plots = Path("local_staging/plots")
    plots.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots / "basic.png", format="png")


###############################################################################

class All:

    def run(
        self,
        distributed: bool = False,
        clean: bool = False,
        debug: bool = False,
        **kwargs,
    ):
        """
        Run a flow with your steps.

        Parameters
        ----------
        distributed: bool
            Create a SLURMCluster to use for job distribution.
            Default: False (do not create a cluster)
        clean: bool
            Should the local staging directory be cleaned prior to this run.
            Default: False (Do not clean)
        debug: bool
            A debug flag for the developer to use to manipulate how much data runs,
            how it is processed, etc.
            Default: False (Do not debug)

        Notes
        -----
        Documentation on prefect:
        https://docs.prefect.io/core/

        Basic prefect example:
        https://docs.prefect.io/core/
        """
        # Choose executor
        if distributed:
            # Log dir settings
            log_dir_name = datetime.now().isoformat().split(".")[0]  # Do not include ms
            log_dir = Path(f".logs/{log_dir_name}/")
            log_dir.mkdir(parents=True)

            # Spawn cluster
            cluster = SLURMCluster(
                cores=2,
                memory="32GB",
                walltime="10:00:00",
                queue="aics_cpu_general",
                local_directory=str(log_dir),
                log_directory=str(log_dir),
            )

            # Set adaptive scaling
            cluster.adapt(minimum_jobs=1, maximum_jobs=40)

        else:
            # Stop conflicts between Dask and OpenBLAS
            # Info here:
            # https://stackoverflow.com/questions/45086246/too-many-memory-regions-error-with-dask
            os.environ["OMP_NUM_THREADS"] = "1"

            # Spawn local cluster
            cluster = LocalCluster()

        # Log bokeh info
        if cluster.dashboard_link:
            log.info(f"Dask UI running at: {cluster.dashboard_link}")

        # Start local dask cluster
        exe = DaskExecutor(cluster.scheduler_address)

        # Configure your flow
        with Flow("example_step_workflow") as flow:
            ams = generate_array.map(range(10, 100))
            ams = invert_array.map(ams)
            ams = sum_array.map(ams)
            basic_plot(ams)

        # Run flow and get ending state
        state = flow.run(executor=exe)


    def pull(self):
        """
        Pull all steps.
        """
        for step in self.step_list:
            step.pull()

    def checkout(self):
        """
        Checkout all steps.
        """
        for step in self.step_list:
            step.checkout()

    def push(self):
        """
        Push all steps.
        """
        for step in self.step_list:
            step.push()

    def clean(self):
        """
        Clean all steps.
        """
        for step in self.step_list:
            step.clean()




class PandasParquetSerializer(Serializer):

    def serialize(self, value: pd.DataFrame) -> bytes:
        b = BytesIO()
        value.to_parquet(b)
        return b.getvalue()

    def deserialize(self, value: bytes) -> pd.DataFrame:
        return pd.read_parquet(value)


class NumpySerializer(Serializer):

    def serialize(self, value: np.ndarray) -> bytes:
        b = BytesIO()
        np.save(b, value, allow_pickle=True)
        return b.getvalue()

    def deserialize(self, value: bytes) -> np.ndarray:
        b = BytesIO(value)
        return np.load(b, allow_pickle=True)
