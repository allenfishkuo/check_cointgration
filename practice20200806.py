# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 13:58:04 2020

@author: user
"""

import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
from scipy.linalg import eigh
import math
import os

path = os.getcwd()
def VAR_model( raw_y , p ):
    
    #k = len(y.T)     # 幾檔股票
    n = len(raw_y)       # 資料長度
    # 以下，把資料疊好，準備做OLS估計
    xt = raw_y[:-p,:]
    for j in range(1,p):
        xt_1 = raw_y[j:-p+j,:]
        xt = np.hstack((xt_1,xt))
    
    int_one = np.ones((n-p, 1))
    int_trd = np.arange(1,n-p+1,1).reshape(n-p,1)
    insept_x = np.hstack( (  int_one ,  int_trd) )
    xt = np.hstack( (insept_x, xt) )
    
    yt = np.delete(raw_y,np.s_[0:p],axis=0)
    #資料疊好了，下面一行是轉成matrix，計算比較不會錯
    xt ,yt = np.mat(xt) , np.mat(yt)

    beta = ( xt.T * xt ).I * xt.T * yt                      # 計算VAR的參數
    E = yt - xt * beta                                      # 計算殘差
    Res_sigma = ( (E.T) * E ) / (n-p)                           # 計算殘差的共變異數矩陣
        
    return [  beta ,Res_sigma ]

# 配適 VAR(P) 模型 ，並利用BIC選擇落後期數，max_p意味著會檢查2~max_p--------------------
def order_select( raw_y , max_p ):
    k = len(raw_y.T)     # 幾檔股票
    n = len(raw_y)       # 資料長度
    lags = [i+1 for i in range(1, max_p)]  #產生一個2~max_p的list
    bic = np.zeros((len(lags),1))
    for p in range(len(lags)):
        sigma = VAR_model( raw_y , lags[p] )[1] 
        bic[p] = np.log( np.linalg.det(sigma) ) + np.log(n) * p * (k*k) / n
        
    bic_order = lags[np.argmin(bic)]
    
    return bic_order

def JCItestModel3(Xt, opt_q, alpha):
    # 防呆檢查
    if alpha == 0.05:
        testCi = 0
    elif alpha == 0.01:
        testCi = 1
    else:
        return 'alpha should be 0.05 or 0.01.'

    [NumObs, NumDim] = Xt.shape

    if NumObs < NumDim:
        return 'Xt must be a T*NumDim matrix and T>NumDim. '
    elif NumDim > 2:
        return 'Only 2 NumDim can work.'
    elif opt_q < 1:
        return 'Set x higher than 1 '

    # 最適落後期數VAR(p)應該要先用order_select 的BIC找最適p，且VEC(q)，q=p-1
    # Xt=NumObs*NumDim的矩陣，NumObs為觀察值（t=1,2,...,NumObs），NumDim為股票個數

    # 設定計算用參數
    
    p = opt_q + 1
    T = NumObs - p
    dY_ALL = Xt[1:, :] - Xt[0:-1, :]
    dY = dY_ALL[p - 1:, :]
    Ys = Xt[p - 1:-1, :]
    dX = np.zeros([T, NumDim * (p - 1)])
    dX = np.hstack( ( dX, np.ones((T,1)) ) )

    for xi in range(p - 1):
        dX[:, xi * NumDim:(xi + 1) * NumDim] = dY_ALL[p - xi - 2:NumObs - xi - 2, :]

    # 轉成matrix，計算比較直觀
    dX, dY, Ys = np.mat(dX), np.mat(dY), np.mat(Ys)

    # 先求dX'*dX 方便下面做inverse
    DX = dX.T * dX
    # I-dX * (dX'*dX)^-1 * dX'
    M = np.identity(T) - dX * DX.I * dX.T

    # matrix 跟numpy 都可以用下面這行，跟上面是同樣的計算
    # M = np.identity(T) - np.dot(np.dot(dX , np.linalg.inv(np.dot(dX.T , dX))) , dX.T )

    R0, R1 = dY.T * M, Ys.T * M
    S00 = R0 * R0.T / T
    S01 = R0 * R1.T / T
    S10 = R1 * R0.T / T
    S11 = R1 * R1.T / T

    eigVals, eigvecs = eigh(S10 * S00.I * S01, S11, eigvals_only=False)
    '''
    #如果要驗算特徵值與特徵矩陣，MustBeZero越接近0越好
    eig_d = np.zeros((NumDim,NumDim))
    for e in range(NumDim):
        eig_d[e,e] = eig_D[e]

    MustBeZero = np.dot(Sxx , eig_V)-np.dot(np.dot(S11 , eig_V), eig_d)
    '''
    # 排序特徵向量與特徵值
    
    sort_ind = np.argsort(-eigVals)
    eigVals = eigVals[sort_ind]
    eigVecs = eigvecs[:, sort_ind]
    eigVals = eigVals.reshape( NumDim , 1)

    # VECM各項參數
    JCIstat = [[0] * 7 for i in range(NumDim)]
    JCIstat[0][:] = ['', 'A', 'B', 'Bq', 'eigValue', 'eigVector', 'testStat']

    # 是否通過檢定，True=通過，False=沒通過
    H = [[0] * 2 for i in range(NumDim)]

    # CVTableRow 跟CVTable 都是算pValue的表格，但是因為CVTable表不精確，Pvalue等之後比較精確再來做
    # CVTableRow = [95, 99]  95=95% , 99=99%
    # CVTable = [ (k-r) , criterion percentage]
    CVTable = [[4.1475, 6.9701],
               [12.3329, 16.2917],
               [24.3168, 29.6712]]

    for rn in range(1, NumDim):
        B = np.mat(eigVecs[:, 0:rn])
        A = S01 * B
        W = dY - Ys * B * A.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T

        Bq = [[0] * 2 for i in range(opt_q)]
        for bi in range(1, opt_q + 1):
            Bq[bi - 1][0] = 'B' + str(bi)
            Bq[bi - 1][1] = P[:, ((bi - 1) * NumDim):bi * NumDim]

        JCIstat[rn][0] = ['r' + str(rn)]
        JCIstat[rn][1] = A[:, 0]
        JCIstat[rn][2] = B[:, 0]
        JCIstat[rn][3] = Bq
        JCIstat[rn][4] = eigVals[rn, :]
        JCIstat[rn][5] = eigVecs[:, rn]
        # 如果需要Residual = W - dX * P.T
        eig_lambda = np.cumprod(1 - eigVals[rn:NumDim, :])
        JCIstat[rn][6] = -2 * np.log(eig_lambda[-1] ** (T / 2))

        H[rn][0] = ['h' + str(rn)]
        H[rn][1] = CVTable[NumDim - rn - 1][testCi] < JCIstat[rn][6]
        '''
        #因為統計檢定量的分配數字還不夠精確，Pvalue的計算暫緩
        CVraw = CVTable[NumDim-rn-1] 
        pvalue = np.interp(JCIstat[rn][11], CVraw , CVTableRow)
        pvalue = 1-(pvalue/100)
        '''
        return [H, JCIstat]
#讀入2015 6月分鐘資料
if __name__ == "__main__":
    
    stock_table_201506 = pd.read_csv(path+'\PTwithTimeTrendMinData201506.csv',header=0)
#分割讀取2015 6/1 分鐘資料
    stock_table_20150601 = stock_table_201506.iloc[0:271,2:]

    #輸入股票代號
    stock_number1 = input("輸入股票代號1:")
    stock_number2 = input("輸入股票代號2:")
    stock= stock_table_20150601[[stock_number1,stock_number2]]
    stock_matrix = stock.values
    opt_p = order_select( stock_matrix , 5 )
    H, JCIstat = JCItestModel3(stock_matrix, opt_p-1, 0.05)

    if (H[1][1] == 1):
        print(JCIstat)
    else:
        print("無共整合關係")
