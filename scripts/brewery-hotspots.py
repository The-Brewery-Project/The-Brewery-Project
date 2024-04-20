# -*- coding: utf-8 -*-
'''
Python script for defining, classifying, and building models to address brewery hotspots.
'''

# import libraries
import numpy as np
import pandas as pd

# pull in city level data
city_df = pd.read_csv('../data/city_level.csv')
