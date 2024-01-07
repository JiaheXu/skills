# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Utils.headers import *


def resample(original_trajectory, desired_number_timepoints):
	original_traj_len = len(original_trajectory)
	new_timepoints = np.linspace(0, original_traj_len-1, desired_number_timepoints, dtype=int)
	return original_trajectory[new_timepoints]

# def wrist_norm(relevant_joints_datapoint):
# 	relevant_joints_datapoint[:, 1:21] -= relevant_joints_datapoint[:, 0].reshape(relevant_joints_datapoint.shape[0], 1, 3)
# 	if len(relevant_joints_datapoint[0]) > 22:
# 		relevant_joints_datapoint[:, 22:] -= relevant_joints_datapoint[:, 21].reshape(relevant_joints_datapoint.shape[0], 1, 3)
# 	return relevant_joints_datapoint


class MAGI_PreDataset(Dataset):

	def __init__(self, args, split='train', short_traj=False, traj_length_threshold=500):

		# Some book-keeping first. 
		self.args = args
		if self.args.datadir is None:
			self.dataset_directory = '../data/MAGI/'
		else:
			self.dataset_directory = self.args.datadir
		   
		self.stat_dir_name = "MAGI"
	
		# Logging all the files we need. 
		self.file_path = os.path.join(self.dataset_directory, '*.npy')
		self.filelist = sorted(glob.glob(self.file_path))

		# print("self.filelist: ", self.filelist)
		# Get number of files. 
		self.total_length = len(self.filelist)

		# Set downsampling frequency.
		self.ds_freq = 1

		# Setup. 
		self.setup()

		self.compute_statistics()

	def set_relevant_joints(self):
		"""
	    Index       Description
        0 - 27	    right shadow hand dof position
        28 - 55	    right shadow hand dof velocity
        56 - 83	    right shadow hand dof force

        84 - 148	right shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
        149 - 178	right shadow hand fingertip force, torque (5 x 6)

        179 - 181	right shadow hand base position
        182 - 184	right shadow hand base rotation
        185 - 212	right shadow hand actions

        213 - 240	left shadow hand dof position
        241 - 268	left shadow hand dof velocity
        269 - 296	left shadow hand dof force
        297 - 361	left shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
        362 - 391	left shadow hand fingertip force, torque (5 x 6)
        392 - 394	left shadow hand base position
        395 - 397	left shadow hand base rotation
        398 - 425	left shadow hand actions

        426 - 432	door right handle position
        433 - 439	door left handle position

		"""
		self.right_hand_joint_max = 213
		self.left_hand_joint_max = 426
		self.object_joint_max = 440
		
		right_tips = []
		for i in range(5):
			right_tips += list(range(84 + i*13, 84 + i*13 + 7) )

		self.robot_index = list(range(0,28)) + list( range(179,185) )
		self.env_index = list( range(426,433) ) #+ list(range(149, 179)) 

		# self.robot_index = list( range(179,185) )
		# self.env_index = list( range(426,433) ) 

		self.joint_indices = self.robot_index + self.env_index 
		
		# 28 + 5*7 + 30 + 6 + 7 = 69 + 7 = 76
		print("self.joint_indices: ", self.joint_indices)
		self.robot_state_dim = len(self.robot_index)
		self.env_state_dim = len(self.env_index)
		
		print("observation dim: ", len(self.joint_indices) )
		print("observation dim: ", len(self.joint_indices) )
		print("observation dim: ", len(self.joint_indices) )

	def subsample_relevant_joints(self, datapoint, dataset_name):

		# Remember, the datapoint is going to be of the form.. 
		# Timesteps x Joints x 3 (dimensions). 
		# Index into it as: 

		# Figure out whether to use full hands, or just use the arm positions. 
		# Consider unsupervised translation to robots without articulated grippers. 
		# For now use arm joint indices. 
		# We can later consider adding other robots / hands.

		self.set_relevant_joints()
		return datapoint[:, self.joint_indices]
		
	def setup(self):

		# Load all files.. 
		self.files = []
		# self.object_files = []
		self.dataset_trajectory_lengths = np.zeros(self.total_length)
		
		# set joints
		self.set_relevant_joints()

		self.cumulative_num_demos = [0]

		# For all files. 
		for k, v in enumerate(self.filelist):
						
			print("Loading from file: ", v)

			# Now actually load file. 
			datapoints = np.load(v, allow_pickle=True).flat[0].get("observations")
			# print("number of demos: ", len(datapoints))
			
			for datapoint in datapoints:

				relevant_joints_datapoint = self.subsample_relevant_joints(datapoint, v)
				reshaped_normalized_datapoint = relevant_joints_datapoint.reshape(relevant_joints_datapoint.shape[0],-1)
				self.state_size = reshaped_normalized_datapoint.shape[1]
				# Subsample in time. 
				number_of_timesteps = datapoint.shape[0]//self.ds_freq
				subsampled_data = resample(reshaped_normalized_datapoint, number_of_timesteps)
				# Add subsampled datapoint to file. 
				self.files.append(subsampled_data)      
      
			self.cumulative_num_demos.append(len(self.files))
		print("Cumulative length:", len(self.files))      

		# Create array. 
		self.file_array = np.array(self.files)
		# self.object_file_array = np.array(self.object_files)
		self.dims = {}

		self.dims["state_size"] = len(self.joint_indices)
		self.dims["state_dim"] = len(self.joint_indices)
		self.dims["robot_state_dim"] = self.robot_state_dim
		self.dims["env_state_dim"] = self.env_state_dim

		# Now save these files.
		self.dataset_directory = self.dataset_directory
		np.save(os.path.join(self.dataset_directory, self.getname() + "_DataFile_BaseNormalize.npy"), self.file_array)
		np.save(os.path.join(self.dataset_directory, self.getname() + "_OrderedFileList.npy"), self.filelist)
		np.save(os.path.join(self.dataset_directory, self.getname() + "_Lengths.npy"), self.cumulative_num_demos)
		np.save(os.path.join(self.dataset_directory, self.getname() + "_Dims.npy"), self.dims)

	def normalize(self, relevant_joints_datapoint):
		return relevant_joints_datapoint

	def getname(self):
		return "MAGI"

	def __len__(self):
		return self.total_length

	def __getitem__(self, index):
		
		if isinstance(index, np.ndarray):
			return list(self.file_array[index])
		else:
			return self.file_array[index]

	def compute_statistics(self):

		self.total_length = self.__len__()
		mean = np.zeros((self.state_size))
		variance = np.zeros((self.state_size))
		mins = np.zeros((self.total_length, self.state_size))
		maxs = np.zeros((self.total_length, self.state_size))
		lens = np.zeros((self.total_length))

		# And velocity statistics. 
		vel_mean = np.zeros((self.state_size))
		vel_variance = np.zeros((self.state_size))
		vel_mins = np.zeros((self.total_length, self.state_size))
		vel_maxs = np.zeros((self.total_length, self.state_size))
		
		for i in range(self.total_length):

			print("Phase 1: DP: ",i)
			data_element = {}
			data_element['is_valid'] = True
			data_element['demo'] = self.file_array[i]
			
			# data_element['object'] = self.object_file_array[i]

			data_element['file'] = self.filelist[i]

			if data_element['is_valid']:
				demo = data_element['demo']
				vel = np.diff(demo,axis=0)

				mins[i] = demo.min(axis=0)
				maxs[i] = demo.max(axis=0)
				mean += demo.sum(axis=0)
				lens[i] = demo.shape[0]

				vel_mins[i] = abs(vel).min(axis=0)
				vel_maxs[i] = abs(vel).max(axis=0)
				vel_mean += vel.sum(axis=0)			

		mean /= lens.sum()
		vel_mean /= lens.sum()

		for i in range(self.total_length):

			print("Phase 2: DP: ",i)
			data_element = {}
			data_element['is_valid'] = True
			data_element['demo'] = self.file_array[i]
			# data_element['object'] = self.object_file_array[i]
			data_element['file'] = self.filelist[i]
			
			# Just need to normalize the demonstration. Not the rest. 
			if data_element['is_valid']:
				demo = data_element['demo']
				vel = np.diff(demo,axis=0)
				variance += ((demo-mean)**2).sum(axis=0)
				vel_variance += ((vel-vel_mean)**2).sum(axis=0)

		variance /= lens.sum()
		variance = np.sqrt(variance)

		vel_variance /= lens.sum()
		vel_variance = np.sqrt(vel_variance)

		max_value = maxs.max(axis=0)
		min_value = mins.min(axis=0)

		vel_max_value = vel_maxs.max(axis=0)
		vel_min_value = vel_mins.min(axis=0)

		statdir = "Statistics/" + self.getname()
		if not os.path.exists(statdir):
			os.makedirs(statdir)

		np.save(os.path.join(statdir, self.getname() + "_Mean.npy"), mean)
		np.save(os.path.join(statdir, self.getname() + "_Var.npy"), variance)
		np.save(os.path.join(statdir, self.getname() + "_Min.npy"), min_value)
		np.save(os.path.join(statdir, self.getname() + "_Max.npy"), max_value)
		np.save(os.path.join(statdir, self.getname() + "_Vel_Mean.npy"), vel_mean)
		np.save(os.path.join(statdir, self.getname() + "_Vel_Var.npy"), vel_variance)
		np.save(os.path.join(statdir, self.getname() + "_Vel_Min.npy"), vel_min_value)
		np.save(os.path.join(statdir, self.getname() + "_Vel_Max.npy"), vel_max_value)

class MAGI_Dataset(Dataset):

	def __init__(self, args):

		# Some book-keeping first. 
		self.args = args
		self.stat_dir_name = 'MAGI'
		
		self.ds_freq = args.ds_freq

		if self.args.datadir is None:
			self.dataset_directory = '../data/MAGI/'
		else:
			self.dataset_directory = self.args.datadir
		   
		# Load file.
		self.data_list = np.load(os.path.join(self.dataset_directory , self.getname() + "_DataFile_BaseNormalize.npy"), allow_pickle=True)
		self.original_data = np.load(os.path.join(self.dataset_directory , self.getname() + "_DataFile_BaseNormalize.npy"), allow_pickle=True)
		
		self.filelist = np.load(os.path.join(self.dataset_directory, self.getname() + "_OrderedFileList.npy"), allow_pickle=True)
		self.cumulative_num_demos = np.load(os.path.join(self.dataset_directory, self.getname() + "_Lengths.npy"), allow_pickle=True)
		self.dims = np.load(os.path.join(self.dataset_directory, self.getname() + "_Dims.npy"), allow_pickle=True)

		self.state_size = self.dims.flat[0]['state_size']
		self.state_dim = self.dims.flat[0]["state_dim"]
		self.robot_state_dim = self.dims.flat[0]["robot_state_dim"]
		self.env_state_dim = self.dims.flat[0]["env_state_dim"]

		self.dataset_length = len(self.data_list)
		self.original_demo_length = []
		

		# print("self.data_list", self.data_list)
		print("dataset_length", self.dataset_length)
		print("dataset_length", self.dataset_length)
		print("dataset_length", self.dataset_length)

		for i in range(self.dataset_length):
			self.original_demo_length.append( copy.deepcopy(self.data_list[i].shape[0] ))
			print("demo length: ", i, " ", self.data_list[i].shape[0])

			number_of_timesteps = self.data_list[i].shape[0]//self.ds_freq
			# print("before downsample: ", self.data_list[i].shape[0])
			self.data_list[i] = resample(self.data_list[i], int(number_of_timesteps)) 
			# print("after downsample: ", self.data_list[i].shape[0])

		if self.args.dataset_traj_length_limit>0:			
			self.short_data_list = []
			self.short_file_list = []
			self.dataset_trajectory_lengths = []
			for i in range(self.dataset_length):
				if self.data_list[i].shape[0]<self.args.dataset_traj_length_limit:
					self.short_data_list.append(self.data_list[i])
					self.dataset_trajectory_lengths.append(self.data_list[i].shape[0])

			for i in range(len(self.filelist)):
				self.short_file_list.append(self.filelist[i])


			self.data_list = self.short_data_list
			# self.object_data_list = self.short_data_list2
			self.filelist = self.short_file_list
			self.dataset_length = len(self.data_list)
			self.dataset_trajectory_lengths = np.array(self.dataset_trajectory_lengths)
				
		self.data_list_array = np.array(self.data_list)		

		self.environment_names = []
		# print("self.filelist: ", self.filelist)
		for i in range(len(self.filelist)):
			f = self.filelist[i][len(self.dataset_directory):-4] # remove path and .pkl
			# print("f: ", f)
			for j in range(self.cumulative_num_demos[i], self.cumulative_num_demos[i+1]):
				self.environment_names.append(f)
		print("Env names:\n", np.unique(self.environment_names))


	def getname(self):
		return "MAGI"

	def __len__(self):

		return self.dataset_length

	def __getitem__(self, index):

		data_element = {}
		data_element['is_valid'] = True
		data_element['demo'] = self.data_list[index]
		data_element['file'] = self.environment_names[index]
		data_element['task-id'] = index

		return data_element
	

# class DexMV_ObjectDataset(DexMV_Dataset):


# 	def getname(self):
# 		return "DexMVObject"

# 	def __getitem__(self, index):
# 		# Return n'th item of dataset.
# 		# This has already processed everything.

# 		# if isinstance(index,np.ndarray):			
# 		# 	return list(self.data_list_array[index])
# 		# else:
# 		# 	return self.data_list[index]

# 		data_element = {}
# 		data_element['is_valid'] = True
# 		data_element['demo'] = self.data_list[index][:, 30:43]
# 		# data_element['object-state'] = self.object_data_list[index]
# 		# data_element['demo'] = np.concatenate((self.data_list[index], self.object_data_list[index]), axis=1)
# 		# task_index = np.searchsorted(self.cumulative_num_demos, index, side='right')-1
# 		# data_element['file'] = self.filelist[task_index][81:-7]
# 		data_element['file'] = self.environment_names[index]
# 		data_element['task-id'] = index
# 		# print("Printing the index and the task ID from dataset:", index, data_element['file'])

# 		return data_element
	
# class DexMVHand_Dataset(DexMV_Dataset):

# 	def getname(self):
# 		return "DexMVHand"

# 	def __getitem__(self, index):
# 		# Return n'th item of dataset.
# 		# This has already processed everything.

# 		# if isinstance(index,np.ndarray):			
# 		# 	return list(self.data_list_array[index])
# 		# else:
# 		# 	return self.data_list[index]

# 		data_element = {}
# 		data_element['is_valid'] = True
# 		data_element['demo'] = self.data_list[index][:, 0:30]
# 		# data_element['object-state'] = self.object_data_list[index]
# 		# data_element['demo'] = np.concatenate((self.data_list[index], self.object_data_list[index]), axis=1)
# 		# task_index = np.searchsorted(self.cumulative_num_demos, index, side='right')-1
# 		# data_element['file'] = self.filelist[task_index][81:-7]
# 		data_element['file'] = self.environment_names[index]
# 		data_element['task-id'] = index
# 		# print("Printing the index and the task ID from dataset:", index, data_element['file'])

# 		return data_element