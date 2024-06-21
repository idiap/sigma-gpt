# Copyright © <2024> Idiap Research Institute <contact@idiap.ch>

# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>

# SPDX-License-Identifier: AGPL-3.0-only
"""Imports module from nanoGPT.

As nanoGPT is not a package, we need to add its path to the sys.path in order to import its modules.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "nanoGPT")))

import model

__all__ = ["model"]
