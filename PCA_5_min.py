# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:18:51 2020

@author: crystal
"""
import pandas as pd
import csv
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn import preprocessing
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
sc = MinMaxScaler()
def cluster_pairs(date):
    df = pd.read_csv('./return_series_5min_move/'+str(date)+'_5min_moving_stock_return_series.csv')#53*148
     #sklearn API 預期的輸入維度為 (n_samples, n_features)
     
    #print('shape:',df.shape) # 53*148
    pca_component = 2
    pca = PCA(n_components= pca_component)
    pca.fit(df.T)
    df1_pca = pca.transform(df.T)
    
   # print('explained_variance_ratio_',pca.explained_variance_ratio_)
    
    #plt.figure()
    """
    plt.plot(range(pca_component),pca.explained_variance_ratio_*100,'o')
    plt.xlabel('dimensions')
    plt.ylabel('(%)')
    plt.title('explained_variance_ratio_')
    plt.show()
    """
    
    df1_pca=df1_pca.T #5*148
    df2=pd.DataFrame(df1_pca)
    df2.columns = ['{}'.format(name) for name in df.columns]
    df2.to_csv('20160104_5min_stock_PCA.csv',index=False) 
    
    X=pd.DataFrame(df1_pca).to_numpy()
    
    #print('xshape',X.shape)#5*148
    X=X.T 
    X = sc.fit_transform(X)
    #print(X)
    
    
    #clustering
    
    clust = OPTICS(min_samples=5, xi=.05).fit(X)
    core_samples_mask = np.zeros_like(clust.labels_, dtype=bool)
    
    labels = clust.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    
    '''
    # #############################################################################
    # Plot result
    
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 0.0001]
    
        class_member_mask = (labels == k)
    
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()'''
    name = ['{}'.format(name) for name in df.columns]
    col = ['{}'.format(name) for name in df.columns]
    #print("name :",col)
    
    #print(labels)
    clust1=[]
    clust2=[]
    clust3=[]
    all_pair = []
    for i in range(n_clusters_):
        cluster =[]
        for j in range(len(labels)):
            if labels[j] == i:
                cluster =np.append(cluster,col[j])
        #print("number:{} cluster:{}".format(i+1,cluster))
        all_pair.extend(list(combinations(cluster,2)))
    #print(all_pair)
    return all_pair
    
    
    
    '''
    #clust 
    print(clust)
    y=clust.labels_
    print('label shape',y.shape)
    print(y)
    
    tsne = manifold.TSNE(n_components=2,perplexity=35, init='random',  verbose=1)
    #clustering = OPTICS(min_samples=2).fit(df2)
    X_tsne = tsne.fit_transform(x)
    df = pd.DataFrame(dict(Feature_1=X_tsne[:,0], Feature_2=X_tsne[:,1], label=y))
    
    df.plot(x="Feature_1", y="Feature_2", kind='scatter',c='label',colormap='viridis')
    '''
