# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:46:05 2020

@author: crystal
"""

import pandas as pd
import csv
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

#return series
path_to_minprice = "./5minprice/"
path_to_5minprice = "./return_series_5min_move/"
ext_of_minprice = "_5min_moving_stock.csv"
ext_of_5minprice = "_5min_moving_stock_return_series.csv"

datelist = [f.split('_')[0] for f in os.listdir(path_to_minprice)]
print(datelist[0])
dateist1 = [datelist[0]]

for date in sorted(datelist):
    
    df = pd.read_csv(path_to_minprice+date+ext_of_minprice)
    df=df.pct_change()#對row做運算
    df.dropna(inplace=True)
    df.to_csv(path_to_5minprice+date+ext_of_5minprice,index = False)
    