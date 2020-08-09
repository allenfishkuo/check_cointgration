# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 20:01:30 2020

@author: user
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from MTSA import order_select 
from vecm import rank,eig, weigh,vecm
from VecmPvalue import vecm_pvalue
from scipy.stats import skew
import ADF
import math
import os

path_to_average = "./2016/averageprice/"
ext_of_average = "_averagePrice_min.csv"

path_compare = "/2016_table/"
path = os.getcwd()

def find_pairs( i , n , min_price ):

    local_select_model = []
    local_weight = []
    local_name = []
    local_pval = []
    for j in range(1):        
        stock1 = min_price[[i]]
        stock2 = min_price[[n]]          
        stock1_name = stock_number1
        stock2_name = stock_number2
        z = ( np.hstack( [stock1 , stock2] ) )
        model = VAR(z)
        p = order_select(z,5)
        if p < 1:                    
            continue            
        # portmanteau test
        if model.fit(p).test_whiteness( nlags = 5 ).pvalue < 0.05:               
            continue            
        # Normality test
        if model.fit(p).test_normality().pvalue < 0.05:                
            continue
        r1 = rank( pd.DataFrame(z) , 'H2' , p ) 
        r2 = rank( pd.DataFrame(z) , 'H1*' , p )
        r3 = rank( pd.DataFrame(z) , 'H1' , p )            
        if r3 > 0:                      # 在 model 3 上有 rank               
            if r2 > 0:                  # 在 model 2 上有 rank                   
                if r1 > 0:              # select model 1 and model 2 and model 3
                    lambda_model2 = eig( pd.DataFrame(z) , 'H1*' , p , r2 )
                    lambda_model3 = eig( pd.DataFrame(z) , 'H1' , p , r2 )
                    test = np.log(lambda_model2/lambda_model3) * (len(min_price)-p)
                
                    if test > 3.8414:   
                        local_select_model.append('model3')
                        local_weight.append( weigh( pd.DataFrame(z) , 'H1' , p , r3 ) )             
                        local_name.append([stock1_name,stock2_name])                       
                        local_pval.append( vecm_pvalue('model3', vecm( pd.DataFrame(z),'H1',p)[0][0] ) )
                   
                    else:
                        lambda_model1 = eig( pd.DataFrame(z) , 'H2' , p , r1 )
                        test = np.log(lambda_model1/lambda_model2) * (len(min_price)-p)
                                
                        if test > 3.8414:

                            local_select_model.append('model2')
                            local_weight.append( weigh( pd.DataFrame(z) , 'H1*' , p , r2 ) )                               
                            local_name.append([stock1_name,stock2_name])                            
                            local_pval.append( vecm_pvalue('model2',vecm(pd.DataFrame(z),'H1*',p)[0][1] ) ) 
                                
                        else:
                            local_select_model.append('model1')
                            local_weight.append( weigh( pd.DataFrame(z) , 'H2' , p , r1 ) )
                            local_name.append([stock1_name,stock2_name])
                            local_pval.append( vecm_pvalue('model1',vecm(pd.DataFrame(z),'H2',p)[0][0] ) )
                    
                else:                   # select model 2 and model 3

                    lambda_model2 = eig( pd.DataFrame(z) , 'H1*' , p , r2 )
                    lambda_model3 = eig( pd.DataFrame(z) , 'H1' , p , r2 )
                    test = np.log(lambda_model2/lambda_model3) * (len(min_price)-p)
                
                    if test > 3.8414:     
                        
                        local_select_model.append('model3')
                        local_weight.append( weigh( pd.DataFrame(z) , 'H1' , p , r3 ) )
                        local_name.append([stock1_name,stock2_name])
                        local_pval.append( vecm_pvalue('model3',vecm(pd.DataFrame(z),'H1',p)[0][0] ) )
                   
                    else:
 
                        local_select_model.append('model2')                    
                        local_weight.append( weigh( pd.DataFrame(z) , 'H1*' , p , r2 ) )                  
                        local_name.append([stock1_name,stock2_name])                       
                        local_pval.append( vecm_pvalue('model2',vecm(pd.DataFrame(z),'H1*',p)[0][1] ) )
                    
            else :                     # 只在 model 3 上有rank                       
                local_select_model.append('model3')
                local_weight.append( weigh( pd.DataFrame(z) , 'H1' , p , r3 ) )
                local_name.append([stock1_name,stock2_name])
                local_pval.append( vecm_pvalue('model3',vecm(pd.DataFrame(z),'H1',p)[0][0] ) )     
        else:       # 表示此配對無rank
            continue

    return local_weight, local_name, local_select_model , local_pval

def check_coint(stock,stock1,stock2):
    formate_time = 150   # 建模時間長度
    trade_time = 100         # 回測時間長度
    for j in range(1):    # 一天建模??次               
        day1_1 = stock.iloc[(trade_time * j) : (formate_time + (trade_time * j)),:]                
        day1_1.index  = np.arange(0,len(day1_1),1)
    #print(type(day1_1))
    unitroot_stock = ADF.adf.drop_stationary(ADF.adf(day1_1))
    print(unitroot_stock)
    x = find_pairs( stock1 , stock2 , unitroot_stock )
    print(x)
    x = list(x)
    if len(x[0])!= 0:
        total = abs(x[0][0][0]) + abs(x[0][0][1])
        weight1_std = x[0][0][0]/total 
        weight2_std = x[0][0][1]/total 
        spread = weight1_std * unitroot_stock.iloc[:,0] + weight2_std *unitroot_stock.iloc[:,1]
        ave = []
        std = []
        ske = []
        for i in range(1):                    
            y = spread                    
            # 有時間趨勢項的模型必須分開計算
            if x[2][0] == 'model4':                      
                x = np.arange(0,len(y))
                b1 , b0 = np.polyfit(x,y,1)                      
                trend_line = x*b1 + b0
                y = y - trend_line                                  
                # 將spread消除趨勢項後，計算mu與std
                ave.append( np.mean(y) )
                std.append( np.std(y) )
                ske.append( skew(y) )                                       
            else:        		
                ave.append( np.mean(y) )
                std.append( np.std(y) )
                ske.append( skew(y) )
        x.extend([ave,std,ske])
        c={"weight1" : weight1_std,
           "weight2" : weight2_std,
           "stock_number1" : x[1][0][0],
           "stock_number2" : x[1][0][1],
           "model":x[2],
           "p_val":x[3],
           "mean":x[4],
           "std":x[5],
           "skewness":x[6]}
        data=pd.DataFrame(c)#将字典转换成为数据框
        print(data)
        return data
   
    else: 
         print("無共整合關係")
     
if __name__ == "__main__": 
    datelist = [f.split('_')[0] for f in os.listdir(path_to_average)]
    #print(datelist)
    #stock_number1 = input("輸入股票代號1:")
    #stock_number2 = input("輸入股票代號2:")
    stock_number1 = "1301"
    stock_number2= "4904"
    for date in sorted(datelist):
        if date == "20160104":
            day1 = pd.read_csv(path_to_average+date+ext_of_average).drop([266,267,268,269,270])
            #print(day1)
            day1 = day1.drop(index=np.arange(0,16,1)) ; day1.index = np.arange(0,len(day1),1)  
                
            # formation period and trading period -----------------------------------------------------------------------------------------------------------           
            high_stock = np.array(np.where(day1.iloc[0,:] > 1000 )).T            
            name = day1.columns.values.tolist()            
            for stock in high_stock:            
                day1.drop(columns = str(name[int(stock)]) , inplace = True ) 
            
            stock = day1[[stock_number1,stock_number2]]
            #print(stock)
            data = check_coint(stock,stock_number1,stock_number2)
            #flag = os.path.isfile(path+path_compare+str(date)+'_table.csv')
        """
        if not flag :
            df.to_csv(path+path_compare+str(date)+'_ground truth.csv', mode='w',index=False)
        else :
            df.to_csv(path+path_compare+str(date)+'_ground truth.csv', mode='a', header=False,index=False)
        """
        print("ZZ")
