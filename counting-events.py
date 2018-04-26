# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:50:20 2018

@author: IsaacShrestha
Counting number of Eventtype in the whole dataset
"""

import pandas as pd

# Import the dataset
dataset = pd.read_excel('C:\Users\IsaacShrestha\Desktop\data\EventCount\dataset.xlsx')

# Counting each eventtype and writing to separate file
df = dataset.groupby('Run')['Event'].value_counts().unstack().fillna(0)
df.to_excel('C:\Users\IsaacShrestha\Desktop\data\EventCount\EventCount.xlsx')