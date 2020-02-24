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
from pathlib import Path

from dask_jobqueue import SLURMCluster
from distributed import LocalCluster
from prefect import Flow
from prefect.engine.executors import DaskExecutor

from example_step_workflow import steps

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class All:
    def __init__(self):
        """
        Set all of your available steps here.
        This is only used for data logging operations, not running.
        """
        self.step_list = [
            steps.MappedRaw(),
            steps.MappedInvert(),
            steps.Sum(),
            steps.Plot(),
        ]

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
        # Initalize steps
        raw = steps.MappedRaw()
        invert = steps.MappedInvert()
        cumsum = steps.Sum()
        plot = steps.Plot()

        # Choose executor
        if distributed:
            # Log dir settings
            log_dir_name = datetime.now().isoformat().split(".")[0]  # Do not include ms
            log_dir = Path(f".logs/{log_dir_name}/")
            log_dir.mkdir(parents=True)

            # Spawn cluster
            cluster = SLURMCluster(
                cores=2,
                memory="4GB",
                walltime="01:00:00",
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
            # If your step utilizes a secondary flow with dask pass the executor address
            # If you want to clean the local staging directories pass clean
            # If you want to utilize some debugging functionality pass debug
            # If you don't utilize any of these, just pass the parameters you need.
            matrices = raw(
                distributed_executor_address=cluster.scheduler_address,
                clean=clean,
                debug=debug,
                **kwargs,  # Allows us to pass `--n {some integer}` or other params
            )
            inversions = invert(
                matrices,
                distributed_executor_address=cluster.scheduler_address,
                clean=clean,
                debug=debug,
            )
            vectors = cumsum(
                inversions,
                distributed_executor_address=cluster.scheduler_address,
                clean=clean,
                debug=debug,
            )
            plot(
                vectors,
                distributed_executor_address=cluster.scheduler_address,
                clean=clean,
                debug=debug,
            )

        # Run flow and get ending state
        state = flow.run(executor=exe)

        # Get plot location
        log.info(f"Plot stored to: {plot.get_result(state, flow)}")

        # Close cluster
        if distributed:
            cluster.close()

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
