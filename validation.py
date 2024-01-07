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

task1_seg = [[94, 135],
[94, 146],
[87, 119],
[106, 133],
[120, 160],
[120, 163],
[130, 170],
[160, 190]]

task2_seg = [[160, 201],
[190, 230],
[140, 182],
[150, 190], 
[160, 200], 
[230, 270], 
[140, 170], 
[140, 170], 
[140, 170], 
[110, 135], 
[160, 180], 
[150, 180], 
[130, 150], 
[140, 175], 
[130, 160], 
[125, 155], 
[160, 190], 
[175, 205], 
[175, 205], 
[155, 185], 
[190, 220], 
[190, 220], 
[195, 225]]

task3_seg = [[150, 190],
[180, 220],
[150, 181],
[145, 170], 
[145, 170], 
[170, 200], 
[150, 170], 
[160, 190], 
[150, 175], 
[160, 180], 
[160, 190], 
[190, 210], 
[140, 170], 
[150, 170], 
[160, 190], 
[170, 200], 
[195, 225], 
[170, 190], 
[130, 160], 
[160, 190]] 

task1_seg = np.array(task1_seg)
task2_seg = np.array(task2_seg)
task3_seg = np.array(task3_seg)

tasks_seg = [task1_seg, task3_seg]

segment_length = 14


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

def remove_same_neighbor( demo_skills ):
	last_skill = -1
	skills = []
	print("demo_skills: ", demo_skills)
	for skill in demo_skills:
		if(skill!= last_skill):
			skills.append(skill)
		last_skill = skill
	print("cleaned skills: ", skills)
	return skills

def get_demo_score(task_id, demo_id, demo_skills):
	ground_truth = tasks_seg[task_id][demo_id]
	print("seg1: ")
	final_stack = []

	stack = []

	for i in range( len(demo_skills) ):
		start_time = i*segment_length
		end_time = start_time + segment_length - 1
		if(end_time <= ground_truth[0]):
			print(demo_skills[i])
			stack.append(demo_skills[i])

	stack = np.array( stack ) 
	final_stack.append(np.bincount(stack).argmax())
	
	stack = []
	print("seg2: ")
	for i in range( len(demo_skills) ):
		start_time = i*segment_length
		end_time = start_time + segment_length - 1
		if(ground_truth[0] < start_time  and end_time < ground_truth[1]):
			print(demo_skills[i])
			stack.append(demo_skills[i])

	stack = np.array( stack ) 
	final_stack.append(np.bincount(stack).argmax())

	stack = []
	print("seg3: ")
	for i in range( len(demo_skills) ):
		start_time = i*segment_length
		end_time = start_time + segment_length - 1
		if(ground_truth[1] <= start_time ):
			print(demo_skills[i])
			stack.append(demo_skills[i])
	
	stack = np.array( stack ) 
	final_stack.append(np.bincount(stack).argmax())
	unique_stack = unique(final_stack)

	score = 0
	if(len(final_stack) == len(unique_stack)):
		score = 1
	print(final_stack)

	return score

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


number_neighbors = 7

kdtree = KDTree(latent_z_set)
# z_neighbor_distances, z_neighbors_indices = kdtree.query( test_set ,p = 2, k=number_neighbors)

tasks_skill = []

task_length = [5,5]
demo_id = [1,2,3,4,5,1,2,3,4,5]
count = 0
for i in range( len(task_length) ):
	task_skill = []
	for j in range(task_length[i]):
		print("task: ", i," demo: ", count)
		demo_skill = []
		
		for start_point in range(0, test_set[count].shape[0], 14 ):
			data_point = test_set[count][start_point]
			z_neighbor_distances, z_neighbors_indices = kdtree.query( data_point ,p = 2, k=number_neighbors)
			skill = np.bincount(labels[z_neighbors_indices]).argmax()
			print("step ", start_point, ' to ',  start_point+13, " is : ", labels[z_neighbors_indices], skill)
			demo_skill.append( skill )
		demo_skill_result = remove_same_neighbor(demo_skill)
		print("ground_truth: ", tasks_seg[i][j])

		get_demo_score(i, j, demo_skill)

		task_skill.append(demo_skill_result)	
		count += 1
	# task_skill_result = get_state_machine(task_skill)
	tasks_skill.append(task_skill)
	# tasks_skill.append(task_skill_result)
	print("\n")