#
# Template for Task 3: Kmeans Clustering
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Your Task --- #
# import libraries as needed 
# .......
# --- end of task --- #

# -------------------------------------
# load data 
# note we do not need label 
data = np.loadtxt('crimerate.csv', delimiter=',')
[n,p] = np.shape(data)
sample = data[:,0:-1]
# -------------------------------------

# --- Your Task --- #
# pick a proper number of clusters 
k = ...
# --- end of task --- #


# --- Your Task --- #
# implement the Kmeans clustering algorithm 
# you need to first randomly initialize k cluster centers 
# ......
# ......
# then start a loop 
# ......
# ......
# ......
# when clustering is done, 
# store the clustering label in `label_cluster' 
# cluster index starts from 0 e.g., 
# label_cluster[0] = 1 means the 1st point assigned to cluster 1
# label_cluster[1] = 0 means the 2nd point assigned to cluster 0
# label_cluster[2] = 2 means the 3rd point assigned to cluster 2
label_cluster = ...
# --- end of task --- #


# the following code plot your clustering result in a 2D space
pca = PCA(n_components=2)
pca.fit(sample)
sample_pca = pca.transform(sample)
idx = []
colors = ['blue','red','green','m']
for i in range(k):
     idx = np.where(label_cluster == i)
     plt.scatter(sample_pca[idx,0],sample_pca[idx,1],color=colors[i],facecolors='none')