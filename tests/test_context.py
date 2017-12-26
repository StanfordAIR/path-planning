from . import context

import sys
import numpy as np
import matplotlib.pyplot as plt

import nav

def test_nav_import():
    assert 'nav' in sys.modules
