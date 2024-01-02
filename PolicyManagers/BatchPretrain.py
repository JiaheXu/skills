from PolicyManagers.Base import *

class PolicyManager_BatchPretrain(PolicyManager_BaseClass):

	def __init__(self, dataset=None, args=None):
		super(PolicyManager_BatchPretrain, self).__init__( dataset=dataset, args=args )
		self.blah = 0

		# Training parameters. 		
		self.baseline_value = 0.
		self.beta_decay = 0.9
		self.learning_rate = self.args.learning_rate
		
		self.initial_epsilon = self.args.epsilon_from
		self.final_epsilon = self.args.epsilon_to
		self.decay_epochs = self.args.epsilon_over
		self.decay_counter = self.decay_epochs*(len(self.dataset)//self.args.batch_size+1)
		self.variance_decay_counter = self.args.policy_variance_decay_over*(len(self.dataset)//self.args.batch_size+1)
		
		if self.args.kl_schedule:
			self.kl_increment_epochs = self.args.kl_increment_epochs
			self.kl_increment_counter = self.kl_increment_epochs*(len(self.dataset)//self.args.batch_size+1)
			self.kl_begin_increment_epochs = self.args.kl_begin_increment_epochs
			self.kl_begin_increment_counter = self.kl_begin_increment_epochs*(len(self.dataset)//self.args.batch_size+1)
			self.kl_increment_rate = (self.args.final_kl_weight-self.args.initial_kl_weight)/(self.kl_increment_counter)
			self.kl_phase_length_counter = self.args.kl_cyclic_phase_epochs*(len(self.dataset)//self.args.batch_size+1)
		# Log-likelihood penalty.
		self.lambda_likelihood_penalty = self.args.likelihood_penalty
		self.baseline = None

		# Per step decay. 
		self.decay_rate = (self.initial_epsilon-self.final_epsilon)/(self.decay_counter)	
		self.linear_variance_decay_rate = (self.args.initial_policy_variance - self.args.final_policy_variance)/(self.variance_decay_counter)
		self.quadratic_variance_decay_rate = (self.args.initial_policy_variance - self.args.final_policy_variance)/(self.variance_decay_counter**2)

		self.state_trajectory_test = None
		self.latent_z_test = None
		self.latent_z_task_id_test = []

	def create_networks(self):
		
		# Create K Policy Networks. 
		# This policy network automatically manages input size. 
		if self.args.discrete_z:
			self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.number_policies, self.number_layers).to(device)
		else:
			self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.args, self.number_layers).to(device)

		# Create encoder.
		if self.args.discrete_z: 
			# The latent space is just one of 4 z's. So make output of encoder a one hot vector.		
			self.encoder_network = EncoderNetwork(self.input_size, self.hidden_size, self.number_policies).to(device)
		else:
			if self.args.split_stream_encoder:
				self.encoder_network = ContinuousFactoredEncoderNetwork(self.input_size, self.args.var_hidden_size, int(self.latent_z_dimensionality/2), self.args).to(device)
			else:
				self.encoder_network = ContinuousEncoderNetwork(self.input_size, self.args.var_hidden_size, self.latent_z_dimensionality, self.args).to(device)
	
	def create_training_ops(self):

		self.negative_log_likelihood_loss_function = torch.nn.NLLLoss(reduction='none')
		self.KLDivergence_loss_function = torch.nn.KLDivLoss(reduction='none')
		# Only need one object of the NLL loss function for the latent net. 

		# These are loss functions. You still instantiate the loss function every time you evaluate it on some particular data. 
		# When you instantiate it, you call backward on that instantiation. That's why you know what loss to optimize when computing gradients. 		

		if self.args.train_only_policy:
			self.parameter_list = self.policy_network.parameters()
		else:
			self.parameter_list = list(self.policy_network.parameters()) + list(self.encoder_network.parameters())
		
		# Optimize with reguliarzation weight.
		self.optimizer = torch.optim.Adam(self.parameter_list,lr=self.learning_rate,weight_decay=self.args.regularization_weight)

	def run_iteration(self, counter, i, return_z=False, and_train=True): 
		# return_z=True, and_train=False) for evaluate
		# self.run_iteration(counter, i) for train
		if self.args.debug:
			print("in PM_BatchPretrain run_iteration func !!!")
			print("in PM_Pretrain run_iteration func !!!")
			print("in PM_Pretrain run_iteration func !!!")

		# Training Process: 
		# For E epochs:
		# 	# For all trajectories:
		#		# Sample trajectory segment from dataset. 
		# 		# Encode trajectory segment into latent z. 
		# 		# Feed latent z and trajectory segment into policy network and evaluate likelihood. 
		# 		# Update parameters. 

		self.set_epoch(counter)

		# Sample trajectory segment from dataset.
		input_dict = {}

		if( self.args.train ):
			input_dict['state_action_trajectory'], input_dict['sample_action_seq'], input_dict['sample_traj'], input_dict['data_element'] = self.get_trajectory_segment(i)
		
		self.sample_traj_var = input_dict['sample_traj']
		self.input_dict = input_dict

		state_action_trajectory = self.corrupt_inputs(input_dict['state_action_trajectory']) # add noise
		# print("state_action_trajectory: ", state_action_trajectory.shape)

		if state_action_trajectory is not None:

			torch_traj_seg = torch.tensor(state_action_trajectory).to(device).float()
			# Encode trajectory segment into latent z. 		
			latent_z, encoder_loglikelihood, encoder_entropy, kl_divergence = self.encoder_network.forward(torch_traj_seg, self.epsilon) #!!!!!!!! here

			update_dict = input_dict
			update_dict['latent_z'] = latent_z

			# Feed latent z and trajectory segment into policy network and evaluate likelihood.
			if( self.args.train ):
				latent_z_seq, latent_b = self.construct_dummy_latents(latent_z)
				_, subpolicy_inputs, sample_action_seq = self.assemble_inputs(state_action_trajectory, latent_z_seq, latent_b, input_dict['sample_action_seq'])
			
				# Policy net doesn't use the decay epislon. (Because we never sample from it in training, only rollouts.)
				loglikelihoods, _ = self.policy_network.forward(subpolicy_inputs, sample_action_seq, self.policy_variance_value)
				loglikelihood = loglikelihoods[:-1].mean()
			
				if and_train: #Update parameters based on likelihood, subpolicy inputs, and kl divergence.
					self.update_policies_reparam(loglikelihood, kl_divergence, update_dict=update_dict)
					# Update Plots.
					stats = {}
					stats['counter'] = counter
					stats['i'] = i
					stats['epoch'] = self.current_epoch_running
					stats['batch_size'] = self.args.batch_size			
					self.update_plots(counter, loglikelihood, state_action_trajectory, stats)

				if return_z:
					return latent_z, input_dict['sample_traj'], sample_action_seq, input_dict['data_element']
									
		return None, None, None, None

	def run_evaluate_iteration(self, i): 
		# return_z=True, and_train=False) for evaluate
		# self.run_iteration(counter, i) for train
		if self.args.debug_evaluate:
			print("in PM_BatchPretrain run_evaluate_iteration func !!!")
			print("in PM_BatchPretrain run_evaluate_iteration func !!!")
			print("in PM_BatchPretrain run_evaluate_iteration func !!!")

		batch_trajectory = self.state_trajectory_test[i*self.args.batch_size: (i+1)*self.args.batch_size]

		# If normalization is set to some value.
		if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
			batch_trajectory = (batch_trajectory-self.norm_sub_value)/self.norm_denom_value
			self.normalized_subsampled_relative_object_state = (self.subsampled_relative_object_state - self.norm_sub_value[-self.args.env_state_size:])/self.norm_denom_value[-self.args.env_state_size:]

			# Compute actions.
			action_sequence = np.diff(batch_trajectory,axis=1)
			self.relative_object_state_actions = np.diff(self.normalized_subsampled_relative_object_state, axis=1)

			# Concatenate
			state_action_trajectory_test = self.concat_state_action(batch_trajectory, action_sequence)

			# Scaling action sequence by some factor.             
			scaled_action_sequence = self.args.action_scale_factor*action_sequence
		
		state_action_trajectory_test = state_action_trajectory_test.transpose((1,0,2))

		torch_traj_seg = torch.tensor(state_action_trajectory_test).to(device).float()	
		latent_z, encoder_loglikelihood, encoder_entropy, kl_divergence = self.encoder_network.forward(torch_traj_seg, 0.0) #!!!!!!!! here
		latent_z = latent_z.detach().cpu().numpy().squeeze()
		
		if self.args.debug_evaluate:
			print("evaluate latent shape: ", latent_z.shape)
			print("evaluate latent shape: ", latent_z.shape)
			print("evaluate latent shape: ", latent_z.shape)
		
		return latent_z


	def get_all_segment(self, data_element, task_id):
		print("task_id: ", task_id)
		traj = []
		for start_time in range(0, data_element['demo'].shape[0] - self.args.test_length + 1, self.args.test_length ):
			end_time = start_time + self.args.test_length
			if(end_time > data_element['demo'].shape[0]):
				break
			traj.append( data_element['demo'][start_time: end_time] )
			self.latent_z_task_id_test.append(task_id)
		traj = np.array(traj)
		if(self.args.debug_evaluate):
			print("demo length: ", data_element['demo'].shape[0] // self.args.test_length)
			print("traj: ", traj.shape[0])
		return traj

	def get_evaluate_data(self):

		trajectory = None
		for i in range( len(self.dataset.filelist) ):
			for j in range( self.args.test_len_pertask ):
				idx = self.dataset.cumulative_num_demos[i] + j
				if (trajectory is None):
					trajectory = self.get_all_segment( self.dataset[idx] , idx)
				else:
					trajectory = np.concatenate( (trajectory, self.get_all_segment(self.dataset[idx], idx) )  )
				# print("trajectory: ", trajectory)
		
		end_idx = ( (trajectory.shape[0]+self.args.batch_size-1) // self.args.batch_size ) * self.args.batch_size
		trajectory = np.pad(trajectory, ((0, end_idx - trajectory.shape[0]), (0,0), (0,0)), "edge")

		self.latent_z_task_id_test = np.array(self.latent_z_task_id_test)
		self.latent_z_task_id_test = np.pad(self.latent_z_task_id_test, ((0, end_idx - trajectory.shape[0])), "edge")

		self.state_trajectory_test = trajectory[0: end_idx]
		self.latent_z_task_id_test = self.latent_z_task_id_test[0: end_idx] 
		return
	
	def get_evaluate_latent_z(self):
		self.latent_z_test = None # clear the stack
		for i in range( self.state_trajectory_test.shape[0]//self.args.batch_size ):
			latent_z = self.run_evaluate_iteration(i)
			if(self.latent_z_test is None ):
				self.latent_z_test = latent_z
			else:
				self.latent_z_test = np.concatenate( (self.latent_z_test, latent_z) )
		return
	
	def save_latent_z(self, latent_z = None, save_latent_z_title = None):

		if(latent_z is None):
			print(" latent_z is None !!!!")
			print(" latent_z is None !!!!")
			print(" latent_z is None !!!!")
			return
		self.z_dir_name = os.path.join(self.dir_name, "Latent_Z")
		if not(os.path.isdir(self.z_dir_name)):
			os.mkdir(self.z_dir_name)

		if(save_latent_z_title is not None):
			file_pth = os.path.join( self.z_dir_name, "{0}.npy".format(save_latent_z_title) )
			np.save( file_pth, latent_z)
			print("file_pth: ", file_pth)
			print("file_pth: ", file_pth)
			print("file_pth: ", file_pth)
			print("latent_z: ", latent_z.shape)
			print("latent_z: ", latent_z.shape)
			print("latent_z: ", latent_z.shape)
			print("task id: ", self.latent_z_task_id_test)

		return

	def evaluate(self, model=None, save_latent_z_title = None):
		if model:
			print("Loading model in evaluating.")
			self.load_all_models(model)	
		if self.state_trajectory_test is None:
			self.get_evaluate_data() #concatenated_traj = self.concat_state_action(batch_trajectory, action_sequence)
		if self.latent_z_test is None:
			self.get_evaluate_latent_z()
		if save_latent_z_title is not None:
			self.save_latent_z( self.latent_z_test, save_latent_z_title)
		else:
			self.save_latent_z( self.latent_z_test, "latent_z_test")
		return

	def get_trajectory_segment(self, i):
		if self.args.debug:
			print("in PM_BatchPretrain get_trajectory_segment func !!!")
			print("in PM_BatchPretrain get_trajectory_segment func !!!")
			print("in PM_BatchPretrain get_trajectory_segment func !!!")

		if self.args.data in global_dataset_list:

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

			return concatenated_traj.transpose((1,0,2)), scaled_action_sequence.transpose((1,0,2)), batch_trajectory.transpose((1,0,2)), data_element

	def relabel_relative_object_state_actions(self, padded_action_seq):

		# Here, remove the actions computed from the absolute object states; 
		# Instead relabel the actions in these dimensions into actions computed from the relative state to EEF.. 

		relabelled_action_sequence = padded_action_seq
		# Relabel the action size computes.. 
		# relabelled_action_sequence[..., self.args.robot_state_size:] = self.relative_object_state_actions
		relabelled_action_sequence[..., -self.args.env_state_size:] = self.relative_object_state_actions

		return relabelled_action_sequence

	def assemble_inputs(self, input_trajectory, latent_z_indices, latent_b, sample_action_seq):

		# Now assemble inputs for subpolicy.
		
		# Create subpolicy inputs tensor. 			
		subpolicy_inputs = torch.zeros((input_trajectory.shape[0], self.args.batch_size, self.input_size+self.latent_z_dimensionality)).to(device)

		# Mask input trajectory according to subpolicy dropout. 
		self.subpolicy_input_dropout_layer = torch.nn.Dropout(self.args.subpolicy_input_dropout)

		torch_input_trajectory = torch.tensor(input_trajectory).view(input_trajectory.shape[0],self.args.batch_size,self.input_size).to(device).float()
		masked_input_trajectory = self.subpolicy_input_dropout_layer(torch_input_trajectory)

		# Now copy over trajectory. 
		# subpolicy_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()         
		subpolicy_inputs[:,:,:self.input_size] = masked_input_trajectory

		# Now copy over latent z's. 
		subpolicy_inputs[range(input_trajectory.shape[0]),:,self.input_size:] = latent_z_indices

		# # Concatenated action sequence for policy network's forward / logprobabilities function. 
		# padded_action_seq = np.concatenate([np.zeros((1,self.output_size)),sample_action_seq],axis=0)
		# View time first and batch second for downstream LSTM.
		padded_action_seq = np.concatenate([sample_action_seq,np.zeros((1,self.args.batch_size,self.output_size))],axis=0)

		if self.args.data in ['RealRobotRigidJEEF']:
			padded_action_seq = self.relabel_relative_object_state_actions(padded_action_seq)

		return None, subpolicy_inputs, padded_action_seq

	def construct_dummy_latents(self, latent_z):

		if not(self.args.discrete_z):
			# This construction should work irrespective of reparam or not.
			latent_z_indices = torch.cat([latent_z for i in range(self.current_traj_len)],dim=0)

		# Setting latent_b's to 00001. 
		# This is just a dummy value.
		# latent_b = torch.ones((5)).to(device).float()
		latent_b = torch.zeros((self.args.batch_size, self.current_traj_len)).to(device).float()
		latent_b[:,0] = 1.

		return latent_z_indices, latent_b	
		# return latent_z_indices





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
						
		# for b in range(min(self.args.batch_size, len(self.index_list) - i)):
		# Changing this selection, assuming that the index_list is corrected to only have within dataset length indices.
		for b in range(self.args.batch_size):

			# print("Index that the get_batch_element is using: b:",b," i+b: ",i+b, self.index_list[i+b])
			# Because of the new creation of index_list in random shuffling, this should be safe to index dataset with.

			# print("Getting data element, b: ", b, "i+b ", i+b, "index_list[i+b]: ", self.index_list[i+b])
			index = self.index_list[ (i+b) % len(self.dataset) ]

			if self.args.train:
				self.coverage[index] += 1
			data_element.append(self.dataset[index])

		return data_element








	def rollout_visuals(self, i, latent_z=None, return_traj=False, rollout_length=None, traj_start=None):

		# Initialize states and latent_z, etc. 
		# For t in range(number timesteps):
		# 	# Retrieve action by feeding input to policy. 
		# 	# Step in environment with action.
		# 	# Update inputs with new state and previously executed action. 

		if self.args.data in ['ContinuousNonZero','DirContNonZero','ToyContext']:
			self.state_dim = 2
			self.rollout_timesteps = 5
		elif self.args.data in ['MIME','OldMIME']:
			self.state_dim = 16
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['Roboturk','OrigRoboturk','FullRoboturk','OrigRoboMimic','RoboMimic']:
			self.state_dim = 8
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['GRAB']:
			self.state_dim = 24
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['GRABArmHand']:
			if self.args.position_normalization == 'pelvis':
				self.state_dim = 144
				if self.args.single_hand in ['left', 'right']:
					self.state_dim //= 2
			else:
				self.state_dim = 147
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['GRABArmHandObject']:
			self.state_size = 96
			self.state_dim = 96
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['GRABObject']:
			self.state_dim = 6
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['GRABHand']:
			self.state_dim = 120
			if self.args.single_hand in ['left', 'right']:
				self.state_dim //= 2
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['DAPG']:
			self.state_dim = 51
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['DAPGHand']:
			self.state_dim = 30
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['DAPGObject']:
			self.state_dim = 21
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['DexMV']:
			self.state_dim = 43
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['DexMVHand']:
			self.state_dim = 30
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['DexMVObject']:
			self.state_dim = 13
			self.rollout_timesteps = self.traj_length

		if rollout_length is not None:
			self.rollout_timesteps = rollout_length
	
		if traj_start is None:
			start_state = torch.zeros((self.state_dim))
		else:
			start_state = torch.from_numpy(traj_start)
		

		if self.args.discrete_z:
			# Assuming 4 discrete subpolicies, just set subpolicy input to 1 at the latent_z index == i. 
			subpolicy_inputs = torch.zeros((1,self.input_size+self.number_policies)).to(device).float()
			subpolicy_inputs[0,self.input_size+i] = 1. 
		else:
			subpolicy_inputs = torch.zeros((1,self.input_size+self.latent_z_dimensionality)).to(device)
			subpolicy_inputs[0,self.input_size:] = latent_z

		subpolicy_inputs[0,:self.state_dim] = start_state
		# subpolicy_inputs[0,-1] = 1.		
		
		for t in range(self.rollout_timesteps-1):

			actions = self.policy_network.get_actions(subpolicy_inputs,greedy=True,batch_size=1)
			
			# Select last action to execute.
			action_to_execute = actions[-1].squeeze(1)

			# Downscale the actions by action_scale_factor.
			action_to_execute = action_to_execute/self.args.action_scale_factor

			# Compute next state. 
			new_state = subpolicy_inputs[t,:self.state_dim]+action_to_execute

			# New input row: 
			if self.args.discrete_z:
				input_row = torch.zeros((1,self.input_size+self.number_policies)).to(device).float()
				input_row[0,self.input_size+i] = 1. 
			else:
				input_row = torch.zeros((1,self.input_size+self.latent_z_dimensionality)).to(device).float()
				input_row[0,self.input_size:] = latent_z
			input_row[0,:self.state_dim] = new_state
			input_row[0,self.state_dim:2*self.state_dim] = action_to_execute	
			# input_row[0,-1] = 1.

			subpolicy_inputs = torch.cat([subpolicy_inputs,input_row],dim=0)
		# print("latent_z:",latent_z)
		trajectory_rollout = subpolicy_inputs[:,:self.state_dim].detach().cpu().numpy()
		# print("Trajectory:",trajectory_rollout)

		if return_traj:
			return trajectory_rollout

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