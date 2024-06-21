# Copyright © <2024> Idiap Research Institute <contact@idiap.ch>

# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>

# SPDX-License-Identifier: AGPL-3.0-only
"""Imports module from Picoclvr.

As picoclvr is not a package, we need to add its path to the sys.path in order to import its modules.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "picoclvr")))

import maze
import mygpt
import problems
import tasks

__all__ = ["maze", "mygpt", "problems", "tasks"]
