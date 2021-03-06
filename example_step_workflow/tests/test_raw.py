#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple example of a test file using a function.
NOTE: All test file names must have one of the two forms.
- `test_<XYY>.py`
- '<XYZ>_test.py'

Docs: https://docs.pytest.org/en/latest/
      https://docs.pytest.org/en/latest/goodpractices.html#conventions-for-python-test-discovery
"""

from example_step_workflow.steps import Raw


# This test just checks to see if the raw step instantiates and runs
def test_raw_run(n=3):
    raw = Raw()
    arrs = raw.run(n=n)
    assert len(raw.manifest) == n
    assert len(arrs) == n
