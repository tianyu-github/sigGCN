#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:38:21 2020

@author: tianyu
"""
'''
path = '/Users/tianyu/Desktop/scRNAseq_Benchmark_datasets/Intra-dataset/Pancreatic_data/'
net='String'
from sklearn import preprocessing   
##############
xin = pd.read_csv(os.path.join(path + 'Xin') +'/Filtered_Xin_HumanPancreas_data.csv',index_col = 0, header = 0)
bh = pd.read_csv(os.path.join(path + 'BaronHuman') +'/Filtered_Baron_HumanPancreas_data.csv',index_col = 0, header = 0)
mu = pd.read_csv(os.path.join(path + 'Muraro') +'/Filtered_Muraro_HumanPancreas_data_renameCols.csv',index_col = 0, header = 0)
se = pd.read_csv(os.path.join(path + 'Segerstolpe') +'/Filtered_Segerstolpe_HumanPancreas_data.csv',index_col = 0, header = 0)

gene_set = list(set(xin.columns)&set(bh.columns)&set(mu.columns)&set(se.columns))
gene_set.sort()
gene_index_bh = [i for i, e in enumerate(bh.columns) if e in gene_set]
xin = xin[gene_set]
bh = bh[gene_set]
mu = mu[gene_set]
se = se[gene_set]

mu = np.log1p(mu)
se = np.log1p(se)
bh = np.log1p(bh)
xin = np.log1p(xin)
min_max_scaler = preprocessing.MinMaxScaler()
mu = min_max_scaler.fit_transform(np.asarray(mu))
se = min_max_scaler.fit_transform(np.asarray(se))
bh = min_max_scaler.fit_transform(np.asarray(bh))
xin = min_max_scaler.fit_transform(np.asarray(xin))
    
############### 
features = pd.read_csv(os.path.join(path + 'BaronHuman') +'/Filtered_Baron_HumanPancreas_data.csv',index_col = 0, header = 0, nrows=2)      
features = findDuplicated(features)
print(features.shape)
adj = sp.load_npz(os.path.join(path + 'BaronHuman') + '/adj'+ net + 'BaronHuman' + '_'+str(features.T.shape[0])+'.npz')
print(adj.shape)
adj = adj[gene_index_bh, :][:, gene_index_bh]

###############
datasets = ['Xin','BaronHuman','Muraro','Segerstolpe', 'BaronMouse']
l_xin = pd.read_csv(os.path.join(path + datasets[0]) +'/Labels.csv',index_col = None)
l_bh = pd.read_csv(os.path.join(path + datasets[1]) +'/Labels.csv',index_col = None) 
l_mu = pd.read_csv(os.path.join(path + datasets[2]) +'/Labels.csv',index_col = None) 
l_mu = l_mu.replace('duct','ductal')
l_mu = l_mu.replace('pp','gamma')
l_se = pd.read_csv(os.path.join(path + datasets[3]) +'/Labels.csv',index_col = None) 
labels_set = list(set(l_xin['x']) & set(l_bh['x']) & set(l_mu['x']))
#    labels = pd.concat((labels, temp), 0)

labels_set = set(['alpha','beta','delta','gamma'])
index = [i for i in range(len(l_mu)) if l_mu['x'][i] in labels_set]
mu = mu[index]
index = [i for i in range(len(l_se)) if l_se['x'][i] in labels_set]
se = se[index]
index = [i for i in range(len(l_bh)) if l_bh['x'][i] in labels_set]
bh = bh[index]
index = [i for i in range(len(l_xin)) if l_xin['x'][i] in labels_set]
xin = xin[index]
alldata = pd.concat((xin,bh,mu,se), 0)

labels = pd.concat((l_xin, l_bh, l_mu, l_se), 0)
labels.columns = ['V1']
class_mapping = {label: idx for idx, label in enumerate(np.unique(labels['V1']))}
labels['V1'] = labels['V1'].map(class_mapping)
del class_mapping
labels = np.asarray(labels).reshape(-1)  
###############

adj, np.asarray(alldata.T), labels


    adj, alldata,labels,shuffle_index = utilsdata.load_largesc(path = filepath, dataset=args.dataset, net='String')
    print('sample shape',shuffle_index.shape)
    features, geneind = utilsdata.high_var_npdata(alldata, num=2000, ind=1)
    adj = adj[geneind,:][:,geneind]   
    del geneind
    adj, features = utilsdata.removeZeroAdj(adj, np.asarray(features))
    print('load done.')
    
    adj = adj/np.max(adj)
#    adj = utilsdata.normalize(adj)
    adj = adj.astype('float32')
    
    features = np.log1p(features)
    
    features = features/np.max(features)
    #features = preprocessing.normalize(features, axis = 1, norm='l1')
    print('******************************',adj.shape, features.shape)    

sum_mu = mu.mean(axis = 1)
sum_se = se.mean(axis = 1)
sum_bh = bh.mean(axis = 1)
sum_xin = xin.mean(axis = 1)
mu = np.log1p(mu)
se = np.log1p(se)
bh = np.log1p(bh)
xin = np.log1p(xin)
    
mu = mu/mu.max().max()    
se = se/se.max().max() 
bh = bh/bh.max().max() 
xin = xin/xin.max().max() 
 
    
import seaborn as sns   
sns.distplot(np.asarray(mu), hist = False, kde = True,
                 kde_kws = {'linewidth': 2.5})        
sns.distplot(np.asarray(se), hist = False, kde = True,
                 kde_kws = {'linewidth': 2.5})  
sns.distplot(np.asarray(bh), hist = False, kde = True,
                 kde_kws = {'linewidth': 2.5})  
sns.distplot(np.asarray(xin), hist = False, kde = True,
                 kde_kws = {'linewidth': 2.5})      
    
'''   
   
#adjall, alldata,labels,shuffle_index = utilsdata.load_largesc(path = filepath, dataset=args.dataset, net='String')
#geneind = np.where(np.sum(alldata, 1) != 0)[0]
#alldata = alldata[geneind]
#adjall = adjall[geneind,:][:,geneind]   

features = np.log1p(alldata)
min_max_scaler = preprocessing.MinMaxScaler()
features = min_max_scaler.fit_transform(np.asarray(features.T))
features = features.T

tfIdf = np.zeros([features.shape[0], features.shape[1]]) 
tf_max = np.max(features, axis = 0)
tf = features/tf_max
idf = np.sum(features, axis = 1)
idf = np.log2(features.shape[1]/idf)
for i in range(features.shape[0]):
    tfIdf[i] = tf[i] * idf[i]



features, geneind = utilsdata.high_tfIdf_npdata(alldata,tfIdf, num=3000, ind=1)
geneind = np.random.choice(range(len(alldata)), size = 3000, replace = False)
features = alldata[geneind]

adj = adjall[geneind,:][:,geneind]   
adj, features = utilsdata.removeZeroAdj(adj, np.asarray(features))


#features = np.log1p(features)    
#features = features/np.max(features)
#features = preprocessing.normalize(features, axis = 1, norm='l1')

np.sum(features == 0)/features.shape[0]/features.shape[1]    

import seaborn as sns   
sns.distplot(np.asarray(features), hist = False, kde = True,
                 kde_kws = {'linewidth': 2.5})     


#sns.distplot(features, bins=10, kde=False, rug=True);    
 







    
    