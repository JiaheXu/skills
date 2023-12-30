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
from sklearn import mixture

import matplotlib
matplotlib.use("Agg")

import seaborn as sns
sns.set(style="white")
colors = sns.color_palette("Paired", n_colors=12).as_hex()

from math import sqrt

from scipy.spatial import KDTree

file_path = "/home/mmpug/CausalSkillLearning/TrainingLogs/MAGI_1224_test/MEval/m100000/NumpyZs/ALL_Latent_Z.npy"

test_latent_z_sets = np.load(file_path, allow_pickle=True)
print("test_latent_z_sets: ", len(test_latent_z_sets))

traj_length = []
pre_sum = [0]
N = 0
test_set = None
for i in range(len(test_latent_z_sets)):
	traj_length.append(test_latent_z_sets[i].shape[0])
	pre_sum.append( pre_sum[-1] + test_latent_z_sets[i].shape[0])
	N += test_latent_z_sets[i].shape[0]
	if(i == 0):
		test_set = test_latent_z_sets[i].copy()
	else:
		# print("test_set: ", test_set.shape)
		# print("latent_z_sets[i]: ", latent_z_sets[i].shape)
		test_set = np.concatenate( (test_set, test_latent_z_sets[i] ))
print("traj_length: ", traj_length)
print("test_set: ", test_set.shape)

file_path2 = "/home/mmpug/CausalSkillLearning/TrainingLogs/MAGI_1224/LatentSetDirectory/E100000_C100000"
latent_z_set = np.load(os.path.join(file_path2, "LatentSet.npy"))

print("latent_z_set: ", latent_z_set.shape)
# print("gt_trajectory_set: ", gt_trajectory_set.shape)
# print("embedded_zs: ", embedded_zs.shape)
# print("task_id_set: ", task_id_set.shape)

def get_robot_embedding(return_tsne_object=False, perplexity=None): #!!! here
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

def plot_embedding(embedded_zs, title, shared=False, trajectory=False): #!!! here
	
	fig = plt.figure()
	ax = fig.gca()
		
	if shared:
		colors = 0.2*np.ones((2*N))
		colors[N:] = 0.8
	else:
		colors = 0.2*np.ones((N))

	ax.scatter(embedded_zs[:,0],embedded_zs[:,1],c=colors,vmin=0,vmax=1,cmap='jet')
		
	# Title. 
	ax.set_title("{0}".format(title),fontdict={'fontsize':15})
	fig.canvas.draw()

	fig.savefig( title[ -3:-1] + ".png" )

	return None

def plot(data, y):
    n = y.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(1.61803398875*4, 4))
    ax.set_facecolor("#bbbbbb")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    # plot the locations of all data points ..
    for i, point in enumerate(data.data):
        if y[i] == 0:
            # .. as well as their predicted class
            ax.scatter(*point, zorder=i, color="#dbe9ff", alpha=.6, edgecolors=colors[0])
        elif y[i] == 1:
            ax.scatter(*point, zorder=i, color="#ffffff", alpha=.6, edgecolors=colors[1])			
        else:
            ax.scatter(*point, zorder=i, color="#ffdbdb", alpha=.6, edgecolors=colors[2])

    handles = [
		plt.Line2D([0], [0], color="r", lw=4, label=" skill 1 "),
        plt.Line2D([0], [0], color="g", lw=4, label=" skill 2 "),
        plt.Line2D([0], [0], color="b", lw=4, label=" skill 3 ")]

    legend = ax.legend(loc="best", handles=handles)

    plt.tight_layout()
    plt.savefig("valid.pdf")

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
	# ax = ax or plt.gca()
	fig, ax = plt.subplots(1, 1, figsize=(1.61803398875*4, 4))
	ax.set_facecolor("#bbbbbb")
	ax.set_xlabel("Dimension 1")
	ax.set_ylabel("Dimension 2")

	labels = gmm.fit(X).predict(X)
	if label:
	    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
	else:
		ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
	ax.axis('equal')
    
	w_factor = 0.2 / gmm.weights_.max()
	for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
		draw_ellipse(pos, covar, alpha=w * w_factor)
	legend = ax.legend(loc="best")

	plt.tight_layout()
	plt.savefig("valid.pdf")

embedded_z_dict = {}
embedded_z_dict['perp5'] = get_robot_embedding(perplexity=5) #!!! here
embedded_z_dict['perp10'] = get_robot_embedding(perplexity=10)
embedded_z_dict['perp30'] = get_robot_embedding(perplexity=30)

# print("embedded_z_dict['perp5']: ", embedded_z_dict['perp5'].size())

data = torch.from_numpy( embedded_z_dict['perp10'] )
n_components = 3
d = 2
model = mixture.GaussianMixture(n_components, covariance_type='full').fit(data)

labels = model.predict(data)
plot_gmm(model, data)

kdtree = KDTree(latent_z_set)
number_neighbors = 1
z_neighbor_distances, z_neighbors_indices = kdtree.query( test_set ,p = 2, k=number_neighbors)
print("z_neighbors_indices: ", len(z_neighbors_indices) )
z_neighbors_indices = np.array(z_neighbors_indices)

count = 0
for i in range( len(traj_length) ):
	if(i%10 == 0):
		print( "task: ", i//10 )
	for j in range(traj_length[i]):
		print(labels[ z_neighbors_indices[count] ], end = "")
		count += 1
	print("\n")
			