# Example Step Workflow

[![Build Status](https://github.com/AllenCellModeling/example_step_workflow/workflows/Build%20Master/badge.svg)](https://github.com/AllenCellModeling/example_step_workflow/actions)
[![Code Coverage](https://codecov.io/gh/AllenCellModeling/example_step_workflow/branch/master/graph/badge.svg)](https://codecov.io/gh/AllenCellModeling/example_step_workflow)

An example step workflow that does a whole bunch of simple random matrix inversion.

---

## Workflow
This example step workflow generates random matrices, inverts them, does a cumulative sum over an axis,
the finally plots them.

![DAG](docs/resources/workflow.png)

By default the workflow and all sub tasks are configured to run in a single process. However, there are example
steps called `mapped_raw` and `mapped_invert` to give an idea of how to switch from single threaded / process
to parallel data gathering and processing.

## Distributed
If you want to run this in a distributed fashion be sure install the distributed dependencies
(`pip install -e .[distributed]`) and additionally create a `workflow_config.json` file with the following contents:
```json
{
    "project_local_staging_dir": "/allen/aics/modeling/{your_name}/results/example_step_workflow/"
}
```

_**Note:** The path your provide in the JSON file can point to wherever you want, the path simply must be available to
all the workers._

## Installation
`pip install git+https://github.com/AllenCellModeling/example_step_workflow.git`

## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

### Developer installation
`pip install -e .[dev]`


***Free software: Allen Institute Software License***
