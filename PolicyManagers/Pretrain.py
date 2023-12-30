from PolicyManagers.Base import *

class PolicyManager_Pretrain(PolicyManager_BaseClass):

	def __init__(self, number_policies=4, dataset=None, args=None):

		super(PolicyManager_Pretrain, self).__init__()


	def save_all_models(self, suffix):

		logdir = os.path.join(self.args.logdir, self.args.name)
		savedir = os.path.join(logdir,"saved_models")
		if not(os.path.isdir(savedir)):
			os.mkdir(savedir)
		save_object = {}
		save_object['Policy_Network'] = self.policy_network.state_dict()
		save_object['Encoder_Network'] = self.encoder_network.state_dict()
		torch.save(save_object,os.path.join(savedir,"Model_"+suffix))

	def load_all_models(self, path, only_policy=False, just_subpolicy=False):
		load_object = torch.load(path)

		if self.args.train_only_policy and self.args.train: 		
			self.encoder_network.load_state_dict(load_object['Encoder_Network'])
		else:
			self.policy_network.load_state_dict(load_object['Policy_Network'])
			if not(only_policy):
				self.encoder_network.load_state_dict(load_object['Encoder_Network'])

	def set_epoch(self, counter):
		if self.args.train:

			# Annealing epsilon and policy variance.
			if counter<self.decay_counter:
				self.epsilon = self.initial_epsilon-self.decay_rate*counter
				
				if self.args.variance_mode in ['Constant']:
					self.policy_variance_value = self.args.variance_value
				elif self.args.variance_mode in ['LinearAnnealed']:
					self.policy_variance_value = self.args.initial_policy_variance - self.linear_variance_decay_rate*counter
				elif self.args.variance_mode in ['QuadraticAnnealed']:
					self.policy_variance_value = self.args.final_policy_variance + self.quadratic_variance_decay_rate*((counter-self.variance_decay_counter)**2)				

			else:
				self.epsilon = self.final_epsilon
				if self.args.variance_mode in ['Constant']:
					self.policy_variance_value = self.args.variance_value
				elif self.args.variance_mode in ['LinearAnnealed', 'QuadraticAnnealed']:
					self.policy_variance_value = self.args.final_policy_variance
		else:
			self.epsilon = self.final_epsilon
			# self.policy_variance_value = self.args.final_policy_variance
			
			# Default variance value, but this shouldn't really matter... because it's in test / eval mode.
			self.policy_variance_value = self.args.variance_value
		
		# print("embed in set epoch")
		# embed()

		# Set KL weight. 
		self.set_kl_weight(counter)		

	def set_kl_weight(self, counter):
		
		# Monotonic KL increase.
		if self.args.kl_schedule=='Monotonic':
			if counter>self.kl_begin_increment_counter:
				if (counter-self.kl_begin_increment_counter)<self.kl_increment_counter:
					self.kl_weight = self.args.initial_kl_weight + self.kl_increment_rate*counter
				else:
					self.kl_weight = self.args.final_kl_weight
			else:
				self.kl_weight = self.args.initial_kl_weight

		# Cyclic KL.
		elif self.args.kl_schedule=='Cyclic':

			# Setup is before X epochs, don't decay / cycle. 
			# After X epochs, cycle. 
			
			if counter<self.kl_begin_increment_counter:
				self.kl_weight = self.args.initial_kl_weight				
			else: 			
				# While cycling, self.kl_phase_length_counter is the number of iterations over which we repeat. 
				# self.kl_increment_counter is the iterations (within a cycle) over which we increment KL to maximum.
				# Get where in a single cycle it is. 
				kl_counter = counter % self.kl_phase_length_counter

				# If we're done with incremenet, just set to final weight. 
				if kl_counter>self.kl_increment_counter:
					self.kl_weight = self.args.final_kl_weight
				# Otherwise, do the incremene.t 
				else:
					self.kl_weight = self.args.initial_kl_weight + self.kl_increment_rate*kl_counter		
		
		# No Schedule. 
		else:
			self.kl_weight = self.args.kl_weight

		# Adding branch for cyclic KL weight.		

	def update_plots(self, counter, loglikelihood, sample_traj, stat_dictionary):#!!! here
		
		# log_dict['Subpolicy Loglikelihood'] = loglikelihood.mean()
		log_dict = {'Subpolicy Loglikelihood': loglikelihood.mean(), 'Total Loss': self.total_loss.mean(), 'Encoder KL': self.encoder_KL.mean(), 'KL Weight': self.kl_weight}
		if self.args.relative_state_reconstruction_loss_weight>0.:
			log_dict['Unweighted Relative State Recon Loss'] = self.unweighted_relative_state_reconstruction_loss
			log_dict['Relative State Recon Loss'] = self.relative_state_reconstruction_loss
			log_dict['Auxillary Loss'] = self.aux_loss
		if self.args.task_based_aux_loss_weight>0.:
			log_dict['Unweighted Task Based Auxillary Loss'] = self.unweighted_task_based_aux_loss
			log_dict['Task Based Auxillary Loss'] = self.task_based_aux_loss
			log_dict['Auxillary Loss'] = self.aux_loss
		if self.args.relative_state_phase_aux_loss_weight>0.:
			log_dict['Unweighted Relative Phase Auxillary Loss'] = self.unweighted_relative_state_phase_aux_loss
			log_dict['Relative Phase Auxillary Loss'] = self.relative_state_phase_aux_loss
			log_dict['Auxillary Loss'] = self.aux_loss
		if self.args.cummulative_computed_state_reconstruction_loss_weight>0.:
			log_dict['Unweighted Cummmulative Computed State Reconstruction Loss'] = self.unweighted_cummulative_computed_state_reconstruction_loss
			log_dict['Cummulative Computed State Reconstruction Loss'] = self.cummulative_computed_state_reconstruction_loss
		if self.args.teacher_forced_state_reconstruction_loss_weight>0.:
			log_dict['Unweighted Teacher Forced State Reconstruction Loss'] = self.unweighted_teacher_forced_state_reconstruction_loss
			log_dict['Teacher Forced State Reconstruction Loss'] = self.teacher_forced_state_reconstruction_loss
		if self.args.cummulative_computed_state_reconstruction_loss_weight>0. or self.args.teacher_forced_state_reconstruction_loss_weight>0.:
			log_dict['State Reconstruction Loss'] = self.absolute_state_reconstruction_loss

		if counter%self.args.display_freq==0:
			
			if self.args.batch_size>1:
				# Just select one trajectory from batch.
				sample_traj = sample_traj[:,0]

			############
			# Plotting embedding in tensorboard. 
			############

			# Get latent_z set. 
			self.get_trajectory_and_latent_sets(get_visuals=True)

			log_dict['Average Reconstruction Error:'] = self.avg_reconstruction_error

			# Get embeddings for perplexity=5,10,30, and then plot these.
			# Once we have latent set, get embedding and plot it. 
			self.embedded_z_dict = {}
			self.embedded_z_dict['perp5'] = self.get_robot_embedding(perplexity=5) #!!! here
			self.embedded_z_dict['perp10'] = self.get_robot_embedding(perplexity=10)
			self.embedded_z_dict['perp30'] = self.get_robot_embedding(perplexity=30)

			# Save embedded z's and trajectory and latent sets.
			self.save_latent_sets(stat_dictionary)

			# Now plot the embedding.
			statistics_line = "Epoch: {0}, Count: {1}, I: {2}, Batch: {3}".format(stat_dictionary['epoch'], stat_dictionary['counter'], stat_dictionary['i'], stat_dictionary['batch_size'])
			image_perp5 = self.plot_embedding(self.embedded_z_dict['perp5'], title="Z Space {0} Perp 5".format(statistics_line)) #!!! here
			image_perp10 = self.plot_embedding(self.embedded_z_dict['perp10'], title="Z Space {0} Perp 10".format(statistics_line))
			image_perp30 = self.plot_embedding(self.embedded_z_dict['perp30'], title="Z Space {0} Perp 30".format(statistics_line))
			
			# Now adding image visuals to the wandb logs.
			# log_dict["GT Trajectory"] = self.return_wandb_image(self.visualize_trajectory(sample_traj))
			log_dict["Embedded Z Space Perplexity 5"] = self.return_wandb_image(image_perp5)
			log_dict["Embedded Z Space Perplexity 10"] =  self.return_wandb_image(image_perp10)
			log_dict["Embedded Z Space Perplexity 30"] =  self.return_wandb_image(image_perp30)

		wandb.log(log_dict, step=counter)

	def plot_embedding(self, embedded_zs, title, shared=False, trajectory=False): #!!! here
	
		fig = plt.figure()
		ax = fig.gca()
		
		if shared:
			colors = 0.2*np.ones((2*self.N))
			colors[self.N:] = 0.8
		else:
			colors = 0.2*np.ones((self.N))
		
		#not for pretrain_sub
		if trajectory:
			# Create a scatter plot of the embedding.

			self.source_manager.get_trajectory_and_latent_sets()
			self.target_manager.get_trajectory_and_latent_sets()

			ratio = 0.4
			color_scaling = 15

			# Assemble shared trajectory set. 
			traj_length = len(self.source_manager.trajectory_set[0,:,0])
			self.shared_trajectory_set = np.zeros((2*self.N, traj_length, 2))
			
			self.shared_trajectory_set[:self.N] = self.source_manager.trajectory_set
			self.shared_trajectory_set[self.N:] = self.target_manager.trajectory_set
			
			color_range_min = 0.2*color_scaling
			color_range_max = 0.8*color_scaling+traj_length-1

			for i in range(2*self.N):
				ax.scatter(embedded_zs[i,0]+ratio*self.shared_trajectory_set[i,:,0],embedded_zs[i,1]+ratio*self.shared_trajectory_set[i,:,1],c=colors[i]*color_scaling+range(traj_length),cmap='jet',vmin=color_range_min,vmax=color_range_max)

		else:
			# Create a scatter plot of the embedding.
			ax.scatter(embedded_zs[:,0],embedded_zs[:,1],c=colors,vmin=0,vmax=1,cmap='jet')
		
		# Title. 
		ax.set_title("{0}".format(title),fontdict={'fontsize':15})
		fig.canvas.draw()
		# Grab image.
		width, height = fig.get_size_inches() * fig.get_dpi()
		image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
		image = np.transpose(image, axes=[2,0,1])

		return image

	def save_latent_sets(self, stats):

		# Save latent sets, trajectory sets, and finally, the embedded z's for later visualization.

		# Create save directory:
		upper_dir_name = os.path.join(self.args.logdir,self.args.name,"LatentSetDirectory")

		if not(os.path.isdir(upper_dir_name)):
			os.mkdir(upper_dir_name)

		self.dir_name = os.path.join(self.args.logdir,self.args.name,"LatentSetDirectory","E{0}_C{1}".format(stats['epoch'],stats['counter']))
		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

		np.save(os.path.join(self.dir_name, "LatentSet.npy") , self.latent_z_set)
		np.save(os.path.join(self.dir_name, "GT_TrajSet.npy") , self.gt_trajectory_set)
		np.save(os.path.join(self.dir_name, "EmbeddedZSet.npy") , self.embedded_z_dict)
		np.save(os.path.join(self.dir_name, "TaskIDSet.npy"), self.task_id_set)

	def load_latent_sets(self, file_path):
		
		self.latent_z_set = np.load(os.path.join(file_path, "LatentSet.npy"))
		self.gt_trajectory_set = np.load(os.path.join(file_path, "GT_TrajSet.npy"), allow_pickle=True)
		self.embedded_zs = np.load(os.path.join(file_path, "EmbeddedZSet.npy"), allow_pickle=True)
		self.task_id_set = np.load(os.path.join(file_path, "TaskIDSet.npy"), allow_pickle=True)

	def assemble_inputs(self, input_trajectory, latent_z_indices, latent_b, sample_action_seq):

		if self.args.discrete_z:
			# Append latent z indices to sample_traj data to feed as input to BOTH the latent policy network and the subpolicy network. 
			assembled_inputs = torch.zeros((len(input_trajectory),self.input_size+self.number_policies+1)).to(device)
			assembled_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()
			assembled_inputs[range(1,len(input_trajectory)),self.input_size+latent_z_indices[:-1].long()] = 1.
			assembled_inputs[range(1,len(input_trajectory)),-1] = latent_b[:-1].float()

			# Now assemble inputs for subpolicy.
			subpolicy_inputs = torch.zeros((len(input_trajectory),self.input_size+self.number_policies)).to(device)
			subpolicy_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()
			subpolicy_inputs[range(len(input_trajectory)),self.input_size+latent_z_indices.long()] = 1.
			# subpolicy_inputs[range(len(input_trajectory)),-1] = latent_b.float()

			# # Concatenated action sqeuence for policy network. 
			padded_action_seq = np.concatenate([sample_action_seq,np.zeros((1,self.output_size))],axis=0)

			return assembled_inputs, subpolicy_inputs, padded_action_seq

		else:
			# Append latent z indices to sample_traj data to feed as input to BOTH the latent policy network and the subpolicy network. 
			assembled_inputs = torch.zeros((len(input_trajectory),self.input_size+self.latent_z_dimensionality+1)).to(device)

			# Mask input trajectory according to subpolicy dropout. 
			self.subpolicy_input_dropout_layer = torch.nn.Dropout(self.args.subpolicy_input_dropout)

			torch_input_trajectory = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()
			masked_input_trajectory = self.subpolicy_input_dropout_layer(torch_input_trajectory)
			assembled_inputs[:,:self.input_size] = masked_input_trajectory

			assembled_inputs[range(1,len(input_trajectory)),self.input_size:-1] = latent_z_indices[:-1]
			assembled_inputs[range(1,len(input_trajectory)),-1] = latent_b[:-1].float()

			# Now assemble inputs for subpolicy.
			subpolicy_inputs = torch.zeros((len(input_trajectory),self.input_size+self.latent_z_dimensionality)).to(device)
			subpolicy_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()
			subpolicy_inputs[range(len(input_trajectory)),self.input_size:] = latent_z_indices
			# subpolicy_inputs[range(len(input_trajectory)),-1] = latent_b.float()

			# # Concatenated action sequence for policy network's forward / logprobabilities function. 
			# padded_action_seq = np.concatenate([np.zeros((1,self.output_size)),sample_action_seq],axis=0)
			padded_action_seq = np.concatenate([sample_action_seq,np.zeros((1,self.output_size))],axis=0)

			return assembled_inputs, subpolicy_inputs, padded_action_seq

	def concat_state_action(self, sample_traj, sample_action_seq):
		# Add blank to start of action sequence and then concatenate. 
		sample_action_seq = np.concatenate([np.zeros((1,self.output_size)),sample_action_seq],axis=0)

		# Currently returns: 
		# s0, s1, s2, s3, ..., sn-1, sn
		#  _, a0, a1, a2, ..., an_1, an
		return np.concatenate([sample_traj, sample_action_seq],axis=-1)

	def old_concat_state_action(self, sample_traj, sample_action_seq):
		sample_action_seq = np.concatenate([sample_action_seq, np.zeros((1,self.output_size))],axis=0)
		return np.concatenate([sample_traj, sample_action_seq],axis=-1)

	def get_trajectory_segment(self, i):
		return None, None, None, None

	def construct_dummy_latents(self, latent_z):

		if self.args.discrete_z:
			latent_z_indices = latent_z.float()*torch.ones((self.traj_length)).to(device).float()			
		else:
			# This construction should work irrespective of reparam or not.
			latent_z_indices = torch.cat([latent_z.squeeze(0) for i in range(self.current_traj_len)],dim=0)

		# Setting latent_b's to 00001. 
		# This is just a dummy value.
		# latent_b = torch.ones((5)).to(device).float()
		latent_b = torch.zeros((self.current_traj_len)).to(device).float()
		latent_b[-1] = 1.

		return latent_z_indices, latent_b			

	def initialize_aux_losses(self):
		
		# Initialize losses.
		self.unweighted_relative_state_reconstruction_loss = 0.
		self.relative_state_reconstruction_loss = 0.
		# 
		self.unweighted_relative_state_phase_aux_loss = 0.
		self.relative_state_phase_aux_loss = 0.
		# 
		self.unweighted_task_based_aux_loss = 0.
		self.task_based_aux_loss = 0.

		# 
		self.unweighted_teacher_forced_state_reconstruction_loss = 0.
		self.teacher_forced_state_reconstruction_loss = 0.
		self.unweighted_cummmulative_computed_state_reconstruction_loss = 0.
		self.cummulative_computed_state_reconstruction_loss = 0.

	def compute_auxillary_losses(self, update_dict):

		self.initialize_aux_losses()

		# Set the relative state reconstruction loss.
		if self.args.relative_state_reconstruction_loss_weight>0.:
			self.compute_relative_state_reconstruction_loss()
		if self.args.task_based_aux_loss_weight>0. or self.args.relative_state_phase_aux_loss_weight>0.:
			self.compute_pairwise_z_distance(update_dict['latent_z'][0])
		# Task based aux loss weight. 
		if self.args.task_based_aux_loss_weight>0.:
			self.compute_task_based_aux_loss(update_dict)
		# Relative. 
		if self.args.relative_state_phase_aux_loss_weight>0.:
			self.compute_relative_state_phase_aux_loss(update_dict)
		if self.args.cummulative_computed_state_reconstruction_loss_weight>0. or self.args.teacher_forced_state_reconstruction_loss_weight>0.:
			self.compute_absolute_state_reconstruction_loss()

		# Weighting the auxillary loss...
		self.aux_loss = self.relative_state_reconstruction_loss + self.relative_state_phase_aux_loss + self.task_based_aux_loss + self.absolute_state_reconstruction_loss

	def compute_pairwise_z_distance(self, z_set):

		# Compute pairwise task based weights.
		self.pairwise_z_distance = torch.cdist(z_set, z_set)[0]

		# Clamped z distance loss. 
		# self.clamped_pairwise_z_distance = torch.clamp(self.pairwise_z_distance - self.args.pairwise_z_distance_threshold, min=0.)
		self.clamped_pairwise_z_distance = torch.clamp(self.args.pairwise_z_distance_threshold - self.pairwise_z_distance, min=0.)

	def compute_relative_state_class_vectors(self, update_dict):

		# Compute relative state vectors.

		# Get original states. 
		robot_traj = self.sample_traj_var[...,:3]
		env_traj = self.sample_traj_var[...,self.args.robot_state_size:self.args.robot_state_size+3]
		relative_state_traj = robot_traj - env_traj		

		# Compute relative state. 
		# relative_state_traj = torch.tensor(robot_traj - env_traj).to(device)
		
		# Compute diff. 
		robot_traj_diff = np.diff(robot_traj, axis=0)
		env_traj_diff = np.diff(env_traj, axis=0)
		relative_state_traj_diff = np.diff(relative_state_traj, axis=0)		

		# Compute norm. 
		robot_traj_norm = np.linalg.norm(robot_traj_diff, axis=-1)
		env_traj_norm = np.linalg.norm(env_traj_diff, axis=-1)
		relative_state_traj_norm = np.linalg.norm(relative_state_traj_diff, axis=-1)

		# Compute sum.
		beta_vector = np.stack([robot_traj_norm.sum(axis=0), env_traj_norm.sum(axis=0), relative_state_traj_norm.sum(axis=0)])

		# Threshold this vector. 
		self.beta_threshold_value = 0.5
		self.thresholded_beta_vector = np.swapaxes((beta_vector>self.beta_threshold_value).astype(float), 0, 1)		
		self.torch_thresholded_beta_vector = torch.tensor(self.thresholded_beta_vector).to(device)

	def compute_task_based_aux_loss(self, update_dict):

		# Task list. 
		task_list = []
		for k in range(self.args.batch_size):
			task_list.append(update_dict['data_element'][k]['task-id'])
		# task_array = np.array(task_list).reshape(self.args.batch_size,1)
		torch_task_array = torch.tensor(task_list, dtype=float).reshape(self.args.batch_size,1).to(device)
		
		# Compute pairwise task based weights. 
		# pairwise_task_matrix = (scipy.spatial.distance.cdist(task_array)==0).astype(int).astype(float)
		pairwise_task_matrix = (torch.cdist(torch_task_array, torch_task_array)==0).int().float()

		# Positive weighted task loss. 
		positive_weighted_task_loss = pairwise_task_matrix*self.pairwise_z_distance

		# Negative weighted task loss. 
		# MUST CHECK SIGNAGE OF THIS. 
		negative_weighted_task_loss = (1.-pairwise_task_matrix)*self.clamped_pairwise_z_distance

		# Total task_based_aux_loss.
		self.unweighted_task_based_aux_loss = (positive_weighted_task_loss + self.args.negative_task_based_component_weight*negative_weighted_task_loss).mean()
		self.task_based_aux_loss = self.args.task_based_aux_loss_weight*self.unweighted_task_based_aux_loss

	def compute_relative_state_phase_aux_loss(self, update_dict):

		# Compute vectors first for the batch.
		self.compute_relative_state_class_vectors(update_dict)

		# Compute similarity of rel state vector across batch.
		self.relative_state_vector_distance = torch.cdist(self.torch_thresholded_beta_vector, self.torch_thresholded_beta_vector)
		self.relative_state_vector_similarity_matrix = (self.relative_state_vector_distance==0).float()
	
		# Now set positive loss.
		positive_weighted_rel_state_phase_loss = self.relative_state_vector_similarity_matrix*self.pairwise_z_distance

		# Set negative component
		negative_weighted_rel_state_phase_loss = (1.-self.relative_state_vector_similarity_matrix)*self.clamped_pairwise_z_distance

		# Total rel state phase loss.
		self.unweighted_relative_state_phase_aux_loss = (positive_weighted_rel_state_phase_loss + self.args.negative_task_based_component_weight*negative_weighted_rel_state_phase_loss).mean()
		self.relative_state_phase_aux_loss = self.args.relative_state_phase_aux_loss_weight*self.unweighted_relative_state_phase_aux_loss

	def compute_relative_state_reconstruction_loss(self):
		
		# Get mean of actions from the policy networks.
		mean_policy_actions = self.policy_network.mean_outputs

		# Get translational states. 
		mean_policy_robot_actions = mean_policy_actions[...,:3]
		mean_policy_env_actions = mean_policy_actions[...,self.args.robot_state_size:self.args.robot_state_size+3]
		# Compute relative actions. 
		mean_policy_relative_state_actions = mean_policy_robot_actions - mean_policy_env_actions

		# Rollout states, then compute relative states - although this shouldn't matter because it's linear. 

		# # Compute relative initial state. 		
		# initial_state = self.sample_traj_var[0]
		# initial_robot_state = initial_state[:,:3]
		# initial_env_state = initial_state[:,self.args.robot_state_size:self.args.robot_state_size+3]
		# relative_initial_state = initial_robot_state - initial_env_state

		# Get relative states.
		robot_traj = self.sample_traj_var[...,:3]
		env_traj = self.sample_traj_var[...,self.args.robot_state_size:self.args.robot_state_size+3]
		relative_state_traj = torch.tensor(robot_traj - env_traj).to(device)
		initial_relative_state = relative_state_traj[0]
		# torch_initial_relative_state = torch.tensor(initial_relative_state).cuda()		

		# Differentiable rollouts. 
		policy_predicted_relative_state_traj = initial_relative_state + torch.cumsum(mean_policy_relative_state_actions, axis=0)

		# Set reconsturction loss.
		self.unweighted_relative_state_reconstruction_loss = (policy_predicted_relative_state_traj - relative_state_traj).norm(dim=2).mean()
		self.relative_state_reconstruction_loss = self.args.relative_state_reconstruction_loss_weight*self.unweighted_relative_state_reconstruction_loss

	def relabel_relative_object_state(self, torch_trajectory):

		# Copy over
		relabelled_state_sequence = torch_trajectory

		# Relabel the dims. 

		print("Debug in Relabel")
		embed()

		torchified_object_state = torch.from_numpy(self.normalized_subsampled_relative_object_state).to(device).view(-1, self.args.batch_size, self.args.env_state_size)		
		relabelled_state_sequence[..., -self.args.env_state_size:] = torchified_object_state

		return relabelled_state_sequence	

	def compute_absolute_state_reconstruction_loss(self):

		# Get the mean of the actions from the policy networks until the penultimate action.
		mean_policy_actions = self.policy_network.mean_outputs[:-1]

		# Initial state - remember, states are Time x Batch x State.
		torch_trajectory = torch.from_numpy(self.sample_traj_var).to(device)

		if self.args.data in ['RealWorldRigidJEEF']:
			torch_trajectory = self.relabel_relative_object_state(torch_trajectory)

		initial_state = torch_trajectory[0]

		# Compute reconstructed trajectory differentiably excluding the first timestep. 
		cummulative_computed_reconstructed_trajectory = initial_state + torch.cumsum(mean_policy_actions, axis=0)
		# Teacher forced state.
		teacher_forced_reconstructed_trajectory = torch_trajectory[:-1] + mean_policy_actions

		# Set both of the reconstruction losses of absolute state.
		self.unweighted_cummulative_computed_state_reconstruction_loss = (cummulative_computed_reconstructed_trajectory - torch_trajectory[1:]).norm(dim=2).mean()
		self.unweighted_teacher_forced_state_reconstruction_loss = (teacher_forced_reconstructed_trajectory - torch_trajectory[1:]).norm(dim=2).mean()
		
		# Weighted losses. 
		self.cummulative_computed_state_reconstruction_loss = self.args.cummulative_computed_state_reconstruction_loss_weight * self.unweighted_cummulative_computed_state_reconstruction_loss
		self.teacher_forced_state_reconstruction_loss = self.args.teacher_forced_state_reconstruction_loss_weight*self.unweighted_teacher_forced_state_reconstruction_loss

		# Merge. 
		self.absolute_state_reconstruction_loss = self.cummulative_computed_state_reconstruction_loss + self.teacher_forced_state_reconstruction_loss

	def update_policies_reparam(self, loglikelihood, encoder_KL, update_dict=None):
		
		self.optimizer.zero_grad()

		# Losses computed as sums.
		# self.likelihood_loss = -loglikelihood.sum()
		# self.encoder_KL = encoder_KL.sum()

		# Instead of summing losses, we should try taking the mean of the  losses, so we can avoid running into issues of variable timesteps and stuff like that. 
		# We should also consider training with randomly sampled number of timesteps.
		self.likelihood_loss = -loglikelihood.mean()
		self.encoder_KL = encoder_KL.mean()

		self.compute_auxillary_losses(update_dict)
		# Adding a penalty for link lengths. 
		# self.link_length_loss = ... 

		self.total_loss = (self.likelihood_loss + self.kl_weight*self.encoder_KL + self.aux_loss) 
		# + self.link_length_loss) 

		if self.args.debug:
			print("Embedding in Update subpolicies.")
			embed()

		self.total_loss.backward()
		self.optimizer.step()	

	def run_iteration(self, counter, i, k=0, return_z=False, and_train=True): # return_z=True, and_train=False) for evaluate
																		 # self.run_iteration(counter, i) for train
		print("in PM_Pretrain run_iteration func !!!")
		# print("in PM_Pretrain run_iteration func !!!")
		# print("in PM_Pretrain run_iteration func !!!")
		####################################
		####################################
		# Basic Training Algorithm: 
		# For E epochs:
		# 	# For all trajectories:
		#		# Sample trajectory segment from dataset. 
		# 		# Encode trajectory segment into latent z. 
		# 		# Feed latent z and trajectory segment into policy network and evaluate likelihood. 
		# 		# Update parameters. 
		####################################
		####################################

		self.set_epoch(counter)

		############# (0) ##################
		# Sample trajectory segment from dataset. 
		####################################

		# Sample trajectory segment from dataset.
		input_dict = {}

		if(self.args.data == "MAGI" and self.args.train == 0):
			input_dict['state_action_trajectory'], input_dict['sample_action_seq'], input_dict['sample_traj'], input_dict['data_element'] = self.get_trajectory(i ,k)

			# print("input_dict['state_action_trajectory']: ", input_dict['state_action_trajectory'].shape)
			# print("input_dict['sample_action_seq']: ", input_dict['sample_action_seq'].shape)
			# print("input_dict['sample_traj']: ", input_dict['sample_traj'].shape)
			#print("input_dict['data_element']: ", input_dict['data_element'].shape)
		else:
			input_dict['state_action_trajectory'], input_dict['sample_action_seq'], input_dict['sample_traj'], input_dict['data_element'] = self.get_trajectory_segment(i)


		self.sample_traj_var = input_dict['sample_traj']
		self.input_dict = input_dict
		####################################
		############# (0a) #############
		####################################

		####################################
		# for evaluation as well? Todo
		####################################
		# Corrupt the inputs according to how much input_corruption_noise is set to.
		state_action_trajectory = self.corrupt_inputs(input_dict['state_action_trajectory'])
		# print("state_action_trajectory: ", state_action_trajectory.shape)
		if state_action_trajectory is not None:
			
			####################################
			############# (1) #############
			####################################

			torch_traj_seg = torch.tensor(state_action_trajectory).to(device).float()
			# Encode trajectory segment into latent z. 		
						
			latent_z, encoder_loglikelihood, encoder_entropy, kl_divergence = self.encoder_network.forward(torch_traj_seg, self.epsilon) #!!!!!!!! here
			# here is how we get latent_z, need to check dim

			####################################
			########## (2) & (3) ##########
			####################################

			# Feed latent z and trajectory segment into policy network and evaluate likelihood. 
			latent_z_seq, latent_b = self.construct_dummy_latents(latent_z)

			############# (3a) #############
			_, subpolicy_inputs, sample_action_seq = self.assemble_inputs(state_action_trajectory, latent_z_seq, latent_b, input_dict['sample_action_seq'])
			
			############# (3b) #############
			# Policy net doesn't use the decay epislon. (Because we never sample from it in training, only rollouts.)

			loglikelihoods, _ = self.policy_network.forward(subpolicy_inputs, sample_action_seq, self.policy_variance_value)
			loglikelihood = loglikelihoods[:-1].mean()
			 
			if self.args.debug:
				print("Embedding in Train.")
				embed()

			####################################
			# (4) Update parameters. 
			####################################
			
			if self.args.train and and_train:

				####################################
				# (4a) Update parameters based on likelihood, subpolicy inputs, and kl divergence.
				####################################
				
				update_dict = input_dict
				update_dict['latent_z'] = latent_z				

				self.update_policies_reparam(loglikelihood, kl_divergence, update_dict=update_dict)

				####################################
				# (4b) Update Plots. 
				####################################
				
				stats = {}
				stats['counter'] = counter
				stats['i'] = i
				stats['epoch'] = self.current_epoch_running
				stats['batch_size'] = self.args.batch_size			
				self.update_plots(counter, loglikelihood, state_action_trajectory, stats)

				####################################
				# (5) Return.
				####################################

			if return_z:
				return latent_z, input_dict['sample_traj'], sample_action_seq, input_dict['data_element']
									
		else: 
			return None, None, None
