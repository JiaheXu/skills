import torch
import numpy as np
import glob, os, sys, argparse
import rosbag
import rospy
import pdb
import sklearn.manifold as skl_manifold
from sklearn.decomposition import PCA
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Ellipse


import matplotlib
matplotlib.use("Agg")

import seaborn as sns
sns.set(style="white")
colors = sns.color_palette("Paired", n_colors=12).as_hex()

from math import sqrt

from scipy.spatial import KDTree

from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from sklearn.cluster import AgglomerativeClustering


file_path2 = "/home/mmpug/skills/TrainingLogs/MAGI_0105_test/MEval/Latent_Z/"
latent_z_set = np.load(os.path.join(file_path2, "latent_z_test.npy"), allow_pickle=True)

print("latent_z_set: ", latent_z_set.shape)

test_set = np.load(os.path.join(file_path2, "latent_z_for_validation.npy"), allow_pickle=True)
print("test_set: ", len(test_set))

for i in range(len(test_set)):
	print(test_set[i].shape)

def get_robot_embedding(latent_z_set, return_tsne_object=False, perplexity=None): #!!! here
	normed_z = latent_z_set #!!! here

	if perplexity is None:
		perplexity = 5
		
	print("Perplexity: ", perplexity)

	tsne = skl_manifold.TSNE(n_components=2,random_state=0,perplexity=perplexity)
	embedded_zs = tsne.fit_transform(normed_z)

	scale_factor = 1
	scaled_embedded_zs = scale_factor*embedded_zs

	if return_tsne_object:
		return scaled_embedded_zs, tsne
	else:
		return scaled_embedded_zs

def plotting(data, label, title):
	plt.scatter(data[:, 0], data[:, 1], c=labels, s=40, cmap='viridis')
	plt.savefig(title + ".png")



data = get_robot_embedding(latent_z_set, perplexity=50) #!!! here
cluster_num = 4

kmeans = KMeans(cluster_num, random_state=0)
labels = kmeans.fit(data).predict(data)
plotting(data, labels, "kmeans")

# gmm = mixture.GaussianMixture(cluster_num, covariance_type='full').fit(data)
# labels = gmm.predict(data)
# plotting(data, labels, "gmm")

# birch = Birch(n_clusters=cluster_num)
# birch.fit(data)
# labels = birch.predict(data)
# plotting(data, labels, "birch")

# model = AffinityPropagation(random_state=0)
# model.fit(data)
# labels = model.labels_
# plotting(data, labels, "AffinityPropagation")


# model = MeanShift(bandwidth=2)
# model.fit(data)
# labels = model.labels_
# plotting(data, labels, "MeanShift")


# model = OPTICS(min_samples=5)
# model.fit(data)
# labels = model.labels_
# plotting(data, labels, "OPTICS")

# model = AgglomerativeClustering()
# model.fit(data)
# labels = model.labels_
# plotting(data, labels, "AgglomerativeClustering")

# dbscan = DBSCAN(eps = 10, min_samples= 5).fit(data)
# labels = dbscan.labels_
# plotting(data, labels, "DBSCAN")


number_neighbors = 1

kdtree = KDTree(latent_z_set)
# z_neighbor_distances, z_neighbors_indices = kdtree.query( test_set ,p = 2, k=number_neighbors)

for i in range(len(test_set)):

	print("task: ", i//5," demo: ", i%5 +1)
	for j in range(0, test_set[i].shape[0], 14 ):
		data_point = test_set[i][j]
		z_neighbor_distances, z_neighbors_indices = kdtree.query( data_point ,p = 2, k=number_neighbors)
		# print(z_neighbors_indices)
		print("step ", j, ' to ',  j+13, " is : ", labels[z_neighbors_indices])
	print("\n")