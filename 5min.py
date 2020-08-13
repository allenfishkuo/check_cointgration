# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:21:48 2020

@author: User
"""

import numpy as np
import pandas as pd
import os
path_to_minprice = "./2016/minprice/"
path_to_5minprice = "./5minprice/"
ext_of_minprice = "_min_stock.csv"
ext_of_5minprice = "_5min_moving_stock.csv"

datelist = [f.split('_')[0] for f in os.listdir(path_to_minprice)]
print(datelist[0])
dateist1 = [datelist[0]]

for date in sorted(datelist):
    table = pd.read_csv(path_to_minprice+date+ext_of_minprice)
    col = table.columns
    print(col)
    table = table.values
    table = table.T
    new = np.zeros([table.shape[0],table.shape[1]//5])
    #print(table.shape)
    #print(new.shape)
    for i in range(table.shape[0]):
        for j in range(0,table.shape[1],5):
            total = 0
            for k in range(5):
                if j  == 270:
                    break
                new[i,j//5] += table[i,j+k]/5
                
    #print(new)
    new_data = pd.DataFrame(new,col)
    new_data = new_data.T
    new_data.to_csv(path_to_5minprice+date+ext_of_5minprice,index = False)
    
                
    
        
    #new_data = pd.DataFrame(np.random.randn(10/0,len(col)),columns=col)