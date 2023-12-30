from PolicyManagers.Base import *

class PolicyManager_BatchPretrain(PolicyManager_BaseClass):

	def __init__(self, dataset=None, args=None):
		super(PolicyManager_BatchPretrain, self).__init__(dataset=dataset, args=args)

	def concat_state_action(self, sample_traj, sample_action_seq):
		# Add blank to start of action sequence and then concatenate. 
		sample_action_seq = np.concatenate([np.zeros((self.args.batch_size,1,self.output_size)),sample_action_seq],axis=1)

		# Currently returns: 
		# s0, s1, s2, s3, ..., sn-1, sn
		#  _, a0, a1, a2, ..., an_1, an
		return np.concatenate([sample_traj, sample_action_seq],axis=-1)

	def old_concat_state_action(self, sample_traj, sample_action_seq):
		sample_action_seq = np.concatenate([sample_action_seq, np.zeros((self.args.batch_size,1,self.output_size))],axis=1)
		return np.concatenate([sample_traj, sample_action_seq],axis=-1)
		
	def get_batch_element(self, i):

		# Make data_element a list of dictionaries. 
		data_element = []
						
		for b in range(self.args.batch_size):
			index = self.index_list[i+b]

			if self.args.train:
				self.coverage[index] += 1
			data_element.append(self.dataset[index])

		return data_element

	def get_trajectory(self, i, k):
		
		print("in PM_BatchPretrain get_trajectory func !!!")
		print("in PM_BatchPretrain get_trajectory func !!!")
		print("in PM_BatchPretrain get_trajectory func !!!")

		if self.args.data in global_dataset_list:

			data_element = self.dataset[self.index_list[i]]
			# print("index_list: ", self.index_list[i])
			self.current_traj_len = 14     

			# print("self.current_traj_len: ", self.current_traj_len)

			batch_trajectory = np.zeros((self.args.batch_size, self.current_traj_len, self.state_size))
			self.subsampled_relative_object_state = np.zeros((self.args.batch_size, self.current_traj_len, self.args.env_state_size))

			# POTENTIAL:
			# for x in range(min(self.args.batch_size, len(self.index_list) - 1)):

			# Changing this selection, assuming that the index_list is corrected to only have within dataset length indices.
			for x in range(self.args.batch_size):
			
				# Select the trajectory for each instance in the batch. 
				if self.args.ee_trajectories:
					traj = data_element[x]['endeffector_trajectory']
				else:
					traj = data_element['demo'][k: k+14]
           
				batch_trajectory[x] = data_element['demo'][k: k+14]

			# print("batch_trajectory: ", batch_trajectory.shape)
			# If normalization is set to some value.
			if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
				batch_trajectory = (batch_trajectory-self.norm_sub_value)/self.norm_denom_value

				if self.args.data not in ['NDAX','NDAXMotorAngles']:
					self.normalized_subsampled_relative_object_state = (self.subsampled_relative_object_state - self.norm_sub_value[-self.args.env_state_size:])/self.norm_denom_value[-self.args.env_state_size:]

			# Compute actions.
			action_sequence = np.diff(batch_trajectory,axis=1)
			if self.args.data not in ['NDAX','NDAXMotorAngles']:
				self.relative_object_state_actions = np.diff(self.normalized_subsampled_relative_object_state, axis=1)

			# Concatenate
			concatenated_traj = self.concat_state_action(batch_trajectory, action_sequence)

			# Scaling action sequence by some factor.             
			scaled_action_sequence = self.args.action_scale_factor*action_sequence

			# return concatenated_traj.transpose((1,0,2)), scaled_action_sequence.transpose((1,0,2)), batch_trajectory.transpose((1,0,2))
			return concatenated_traj.transpose((1,0,2)), scaled_action_sequence.transpose((1,0,2)), batch_trajectory.transpose((1,0,2)), data_element

	def get_trajectory_segment(self, i):
		
		print("in PM_BatchPretrain get_trajectory_segment func !!!")
		print("in PM_BatchPretrain get_trajectory_segment func !!!")
		print("in PM_BatchPretrain get_trajectory_segment func !!!")

		if self.args.data in global_dataset_list:

			if self.args.data=='Mocap':
				data_element = self.dataset[self.index_list[i:i+self.args.batch_size]]				
			else:
				data_element = self.get_batch_element(i)

			# If allowing variable skill length, set length for this sample.				
			if self.args.var_skill_length:
				# Choose length of 12-16 with certain probabilities. 
				self.current_traj_len = np.random.choice([12,13,14,15,16],p=[0.1,0.2,0.4,0.2,0.1])
			else:
				self.current_traj_len = self.traj_length            
			
			batch_trajectory = np.zeros((self.args.batch_size, self.current_traj_len, self.state_size))
			self.subsampled_relative_object_state = np.zeros((self.args.batch_size, self.current_traj_len, self.args.env_state_size))

			# POTENTIAL:
			# for x in range(min(self.args.batch_size, len(self.index_list) - 1)):

			# Changing this selection, assuming that the index_list is corrected to only have within dataset length indices.
			for x in range(self.args.batch_size):
			
				# Select the trajectory for each instance in the batch. 
				if self.args.ee_trajectories:
					traj = data_element[x]['endeffector_trajectory']
				else:
					traj = data_element[x]['demo']

				# Pick start and end.               

				# Sample random start point.
				if traj.shape[0]>self.current_traj_len:

					bias_length = int(self.args.pretrain_bias_sampling*traj.shape[0])

					# Probability with which to sample biased segment: 
					sample_biased_segment = np.random.binomial(1,p=self.args.pretrain_bias_sampling_prob)

					# If we want to bias sampling of trajectory segments towards the middle of the trajectory, to increase proportion of trajectory segments
					# that are performing motions apart from reaching and returning. 

					# Sample a biased segment if trajectory length is sufficient, and based on probability of sampling.
					if ((traj.shape[0]-2*bias_length)>self.current_traj_len) and sample_biased_segment:		
						start_timepoint = np.random.randint(bias_length, traj.shape[0] - self.current_traj_len - bias_length)
					else:
						start_timepoint = np.random.randint(0,traj.shape[0]-self.current_traj_len)

					end_timepoint = start_timepoint + self.current_traj_len


					# print("self.state_dim: ", self.state_dim) #self.state_dim = 13
					# print("self.rollout_timesteps: ", self.rollout_timesteps) #self.rollout_timesteps = self.traj_length

					if self.args.ee_trajectories:
						batch_trajectory[x] = data_element[x]['endeffector_trajectory'][start_timepoint:end_timepoint]
					else:
						batch_trajectory[x] = data_element[x]['demo'][start_timepoint:end_timepoint]
					
					if not(self.args.gripper):
						if self.args.ee_trajectories:
							batch_trajectory[x] = data_element['endeffector_trajectory'][start_timepoint:end_timepoint,:-1]
						else:
							batch_trajectory[x] = data_element['demo'][start_timepoint:end_timepoint,:-1]

					if self.args.data in ['RealWorldRigid', 'RealWorldRigidJEEF']:

						# Truncate the images to start and end timepoint. 
						data_element[x]['subsampled_images'] = data_element[x]['images'][start_timepoint:end_timepoint]

					if self.args.data in ['RealWorldRigidJEEF']:
						self.subsampled_relative_object_state[x] = data_element[x]['relative-object-state'][start_timepoint:end_timepoint]

			# If normalization is set to some value.
			if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
				batch_trajectory = (batch_trajectory-self.norm_sub_value)/self.norm_denom_value

				if self.args.data not in ['NDAX','NDAXMotorAngles']:
					self.normalized_subsampled_relative_object_state = (self.subsampled_relative_object_state - self.norm_sub_value[-self.args.env_state_size:])/self.norm_denom_value[-self.args.env_state_size:]

			# Compute actions.
			action_sequence = np.diff(batch_trajectory,axis=1)
			if self.args.data not in ['NDAX','NDAXMotorAngles']:
				self.relative_object_state_actions = np.diff(self.normalized_subsampled_relative_object_state, axis=1)

			# Concatenate
			concatenated_traj = self.concat_state_action(batch_trajectory, action_sequence)

			# Scaling action sequence by some factor.             
			scaled_action_sequence = self.args.action_scale_factor*action_sequence

			# return concatenated_traj.transpose((1,0,2)), scaled_action_sequence.transpose((1,0,2)), batch_trajectory.transpose((1,0,2))
			return concatenated_traj.transpose((1,0,2)), scaled_action_sequence.transpose((1,0,2)), batch_trajectory.transpose((1,0,2)), data_element

	def construct_dummy_latents(self, latent_z):

		if not(self.args.discrete_z):
			# This construction should work irrespective of reparam or not.
			latent_z_indices = torch.cat([latent_z for i in range(self.current_traj_len)],dim=0)

		# Setting latent_b's to 00001. 
		latent_b = torch.zeros((self.args.batch_size, self.current_traj_len)).to(device).float()
		latent_b[:,0] = 1.

		return latent_z_indices, latent_b	
		# return latent_z_indices



