# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from locale import normalize
from os import environ
from Utils.headers import *
from PolicyManagers.PolicyNetworks import *

from Utils.Visualizers import ToyDataVisualizer

# Check if CUDA is available, set device to GPU if it is, otherwise use CPU.
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.set_printoptions(sci_mode=False, precision=2)

# Global data list
global global_dataset_list 
global_dataset_list = [ 'DAPG', 'DAPGHand', 'DAPGObject', 'DexMV', 'DexMVHand', 'DexMVObject', \
			'RealWorldRigid', 'RealWorldRigidRobot', 'RealWorldRigidJEEF', "MAGI"]

class PolicyManager_BaseClass():

	def __init__(self, number_policies=4, dataset=None, args=None):
		super(PolicyManager_BaseClass, self).__init__()
		self.args = args
		# Fixing seeds.
		print("Setting random seeds.")

		print("self.args.seed: ", self.args.seed)
		np.random.seed(seed=self.args.seed)
		torch.manual_seed(self.args.seed)	

		self.data = self.args.data
		self.dataset = dataset

		self.number_policies = number_policies # Not used if discrete_z is false.

		# Model size parameters
		self.state_size = self.dataset.state_size 
		self.state_dim = self.dataset.state_dim
		self.input_size = 2*self.state_size
		self.hidden_size = self.args.hidden_size
		# Number of actions
		self.output_size = self.state_size		
		self.latent_z_dimensionality = self.args.z_dimensions
		self.number_layers = self.args.number_layers
		self.traj_length = self.args.traj_length
		self.number_epochs = self.args.epochs
		self.test_set_size = 0
		self.conditional_info_size = 0

		stat_dir_name = self.dataset.stat_dir_name
		if self.args.normalization=='meanvar':
			self.norm_sub_value = np.load("./data/Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
			self.norm_denom_value = np.load("./data/Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
		elif self.args.normalization=='minmax':
			self.norm_sub_value = np.load("./data/Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
			self.norm_denom_value = np.load("./data/Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
			self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1
			self.norm_sub_value = self.norm_sub_value[:self.state_dim]
			self.norm_denom_value = self.norm_denom_value[:self.state_dim]

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

		self.blah = 0

	def setup(self):

		print("RUNNING SETUP OF: ", self)

		# Fixing seeds.
		np.random.seed(seed=self.args.seed)
		torch.manual_seed(self.args.seed)
		np.set_printoptions(suppress=True,precision=2)

		self.create_networks()
		self.create_training_ops()

		extent = len(self.dataset) - self.test_set_size

		self.index_list = np.arange(0,extent)
		self.initialize_plots()

	def create_networks(self):

		# Create K Policy Networks. 
		# This policy network automatically manages input size. 
		if self.args.discrete_z:
			self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.number_policies, self.number_layers).to(device)
		else:
			# self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.latent_z_dimensionality, self.number_layers).to(device)
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
				# self.encoder_network = OldContinuousEncoderNetwork(self.input_size, self.args.var_hidden_size, self.latent_z_dimensionality, self.args).to(device)

	def create_training_ops(self):
		# self.negative_log_likelihood_loss_function = torch.nn.NLLLoss()
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

	def initialize_plots(self):
		
		self.visualizer = ToyDataVisualizer()
		self.rollout_gif_list = []
		self.gt_gif_list = []

		self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval")
		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

	def write_and_close(self):
		self.writer.export_scalars_to_json("./all_scalars.json")
		self.writer.close()

	def set_extents(self):

		##########################
		# Set extent.
		##########################

		# Modifying to make training functions handle batches. 
		# For every item in the epoch:

		if self.args.debugging_datapoints>-1:				
			extent = self.args.debugging_datapoints
		else:
			extent = len(self.dataset)-self.test_set_size

		if self.args.task_discriminability or self.args.task_based_supervision:
			extent = self.extent	

		return extent
	
	def train(self, model=None):

		print("Running Main Train Function. !!!")
		print("Running Main Train Function. !!!")
		print("Running Main Train Function. !!!")

		########################################
		# (1) Load Model If Necessary
		########################################
		if model:
			print("Loading model in training.")
			self.load_all_models(model)			
		
		########################################
		# (2) Set initial values.
		########################################

		counter = self.args.initial_counter_value
		epoch_time = 0.
		cum_epoch_time = 0.		
		self.epoch_coverage = np.zeros(len(self.dataset))

		########################################
		# (3) Outer loop over epochs. 
		########################################
		
		# For number of training epochs. 
		for e in range(self.number_epochs+1): 
					
			########################################
			# (4a) Bookkeeping
			########################################

			if e%self.args.save_freq==0:
				self.save_all_models("epoch{0}".format(e))

			if self.args.debug:
				print("Embedding in Outer Train Function.")
				embed()

			self.current_epoch_running = e
			print("Starting Epoch: ",e)

			########################################
			# (4b) Set extent of dataset. 
			########################################

			# Modifying to make training functions handle batches. 
			extent = self.set_extents()

			########################################
			# (4c) Shuffle based on extent of dataset. 
			########################################						

			# np.random.shuffle(self.index_list)
			self.shuffle(extent)
			self.batch_indices_sizes = []

			########################################
			# (4d) Inner training loop
			########################################

			t1 = time.time()
			self.coverage = np.zeros(len(self.dataset))

			# For all data points in the dataset. 
			for i in range(0,self.training_extent,self.args.batch_size):				
			# for i in range(0,extent-self.args.batch_size,self.args.batch_size):
				# print("RUN TRAIN", i)
				# Probably need to make run iteration handle batch of current index plus batch size.				
				# with torch.autograd.set_detect_anomaly(True):
				t2 = time.time()

				##############################################
				# (5) Run Iteration
				##############################################
				self.run_iteration(counter, i)
				t3 = time.time()
				print("Epoch:",e,"Trajectory:",str(i).zfill(5), "Datapoints:",str(i).zfill(5), "Iter Time:",format(t3-t2,".4f"),"PerET:",format(cum_epoch_time/max(e,1),".4f"),"CumET:",format(cum_epoch_time,".4f"),"Extent:",extent)
				counter = counter+1
				
			t4 = time.time()
			epoch_time = t4-t1
			cum_epoch_time += epoch_time

			##############################################
			# (7) Automatic evaluation if we need it. 
			##############################################
				
			# if e%self.args.eval_freq==0:
			# 	self.automatic_evaluation(e)

			self.epoch_coverage += self.coverage

	def preprocess_action(self, action=None):

		########################################
		# Numpy-fy and subsample. (It's 1x|S|, that's why we need to index into first dimension.)

		# It is now 1x|S| or Bx|S|? So squeezing should still be okay... 
		########################################
		
		# action_np = action.detach().cpu().numpy()[0,:8]			
		action_np = action.detach().cpu().squeeze(0).numpy()[...,:8]

		########################################
		# Unnormalize action.
		########################################

		if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
			# Remember. actions are normalized just by multiplying denominator, no addition of mean.
			unnormalized_action = (action_np*self.norm_denom_value)
		else:
			unnormalized_action = action_np
		
		########################################
		# Scale action.
		########################################

		scaled_action = unnormalized_action*self.args.sim_viz_action_scale_factor

		########################################
		# Second unnormalization to undo the visualizer environment normalization.... 
		########################################

		ctrl_range = self.visualizer.environment.sim.model.actuator_ctrlrange
		bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
		weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
		# Modify gripper normalization, so that the env normalization actually happens.
		bias = bias[:-1]
		bias[-1] = 0.
		weight = weight[:-1]
		weight[-1] = 1.
		
		# Unnormalized_scaled_action_for_env_step
		if self.visualizer.new_robosuite:
			unnormalized_scaled_action_for_env_step = scaled_action
		else:
			unnormalized_scaled_action_for_env_step = (scaled_action - bias)/weight

		# print("#####################")
		# print("Vanilla A:", action_np)
		# print("Stat Unnorm A: ", unnormalized_action)
		# print("Scaled A: ", scaled_action)
		# print("Env Unnorm A: ", unnormalized_scaled_action_for_env_step)

		return unnormalized_scaled_action_for_env_step

	def compute_next_state(self, current_state=None, action=None):

		####################################
		# If we're stepping in the environment:
		####################################
		
		if self.args.viz_sim_rollout:
			
			####################################
			# Take environment step.
			####################################

			action_to_execute = self.preprocess_action(action)

			########################################
			# Repeat steps for K times.
			########################################
			
			for k in range(self.args.sim_viz_step_repetition):
				# Use environment to take step.
				env_next_state_dict, _, _, _ = self.visualizer.environment.step(action_to_execute)
				gripper_state = env_next_state_dict[self.visualizer.gripper_key]
				if self.visualizer.new_robosuite:
					joint_state = self.visualizer.environment.sim.get_state()[1][:7]
				else:
					joint_state = env_next_state_dict['joint_pos']

			####################################
			# Assemble robot state.
			####################################
			
			gripper_open = np.array([0.0115, -0.0115])
			gripper_closed = np.array([-0.020833, 0.020833])

			# The state that we want is ... joint state?
			gripper_finger_values = gripper_state
			gripper_values = (gripper_finger_values - gripper_open)/(gripper_closed - gripper_open)			

			finger_diff = gripper_values[1]-gripper_values[0]
			gripper_value = 2*finger_diff-1

			########################################
			# Concatenate joint and gripper state. 	
			########################################

			robot_state_np = np.concatenate([joint_state, np.array(gripper_value).reshape((1,))])

			########################################
			# Assemble object state.
			########################################

			# Get just the object pose, object quaternion.
			object_state_np = env_next_state_dict['object-state'][:7]

			########################################
			# Assemble next state.
			########################################

			# Parse next state from dictionary, depending on what dataset we're using.

			# If we're using a dataset with both objects and the robot. 
			if self.args.data in ['RoboturkRobotObjects','RoboMimicRobotObjects']:
				next_state_np = np.concatenate([robot_state_np,object_state_np],axis=0)

			# REMEMBER, We're never actually using an only object dataset here, because we can't actually actuate the objects..
			# # If we're using an object only dataset. 
			# elif self.args.data in ['RoboturkObjects']: 
			# 	next_state_np = object_state_np			

			# If we're using a robot only dataset.
			else:
				next_state_np = robot_state_np

			if self.args.normalization in ['meanvar','minmax']:
				next_state_np = (next_state_np - self.norm_sub_value)/self.norm_denom_value

			# Return torchified version of next_state
			next_state = torch.from_numpy(next_state_np).to(device)	


			# print("embedding at gazoo")
			# embed()
			return next_state, env_next_state_dict[self.visualizer.image_key]

		####################################
		# If not using environment to rollout trajectories.
		####################################

		else:			
			# Simply create next state as addition of current state and action.		
			next_state = current_state+action
			# Return - remember this is already a torch tensor now.
			return next_state, None

	def get_robot_embedding(self, return_tsne_object=False, perplexity=None): #!!! here

		# # Mean and variance normalize z.
		# mean = self.latent_z_set.mean(axis=0)
		# std = self.latent_z_set.std(axis=0)
		# normed_z = (self.latent_z_set-mean)/std
		normed_z = self.latent_z_set #!!! here

		if perplexity is None:
			perplexity = self.args.perplexity
		
		print("Perplexity: ", perplexity)

		tsne = skl_manifold.TSNE(n_components=2,random_state=0,perplexity=perplexity)
		embedded_zs = tsne.fit_transform(normed_z)

		scale_factor = 1
		scaled_embedded_zs = scale_factor*embedded_zs

		if return_tsne_object:
			return scaled_embedded_zs, tsne
		else:
			return scaled_embedded_zs

	def return_wandb_image(self, image):
		return [wandb.Image(image.transpose(1,2,0))]		

	def return_wandb_gif(self, gif):
		return wandb.Video(gif.transpose((0,3,1,2)), fps=4, format='gif')

	def corrupt_inputs(self, input):
		# 0.1 seems like a good value for the input corruption noise value, that's basically the standard deviation of the Gaussian distribution form which we sample additive noise.
		if isinstance(input, np.ndarray):
			corrupted_input = np.random.normal(loc=0.,scale=self.args.input_corruption_noise,size=input.shape) + input
		else:			
			corrupted_input = torch.randn_like(input)*self.args.input_corruption_noise + input
		return corrupted_input	


	def trajectory_length_based_shuffling(self, extent, shuffle=True):
		
		# If we're using full trajectories, do trajectory length based shuffling.
		self.sorted_indices = np.argsort(self.dataset.dataset_trajectory_lengths)[::-1]

		# # Bias towards using shorter trajectories if we're debugging.
		# Use dataset_trajectory_length_bias arg isntaed.
		# if self.args.debugging_datapoints > -1: 
		# 	# BIAS SORTED INDICES AWAY FROM SUPER LONG TRAJECTORIES... 
		# 	self.traj_len_bias = 3000
		# 	self.sorted_indices = self.sorted_indices[self.traj_len_bias:]
		
		# Actually just uses sorted_indices...		
		blocks = [self.sorted_indices[i:i+self.args.batch_size] for i in range(0, extent, self.args.batch_size)]
		
		if shuffle:
			np.random.shuffle(blocks)
		# Shuffled index list is just a flattening of blocks.
		self.index_list = [b for bs in blocks for b in bs]

	def randomized_trajectory_length_based_shuffling(self, extent, shuffle=True):
		
		# Pipline.
		# 0) Set block size, and set extents. 
		# 1) Create sample index list. 
		# 2) Fluff indices upto training_extent size. 
		# 3) Sort based on dataset trajectory length. 
		# 4) Set block size. 
		# 5) Block up. 
		# 6) Shuffle blocks. 
		# 7) Divide blocks. 

		# 0) Set block size, and extents. 
		# The higher the batches per block parameter, more randomness, but more suboptimality in terms of runtime. 
		# With dataset trajectory limit, should not be too bad.  
		batches_per_block = 2

		# Now that extent is set, create rounded down extent.
		self.rounded_down_extent = extent//self.args.batch_size*self.args.batch_size

		# Training extent:
		if self.rounded_down_extent==extent:
			self.training_extent = self.rounded_down_extent
		else:
			# This needs to be done such that we have %3==0 batches. 
			batches_to_add = batches_per_block-(self.rounded_down_extent//self.args.batch_size)%batches_per_block
			self.training_extent = self.rounded_down_extent+self.args.batch_size*batches_to_add

		# 1) Create sample index list. 
		original_index_list = np.arange(0,extent)
		
		# 2) Fluff indices upto training_extent size. 		
		if self.rounded_down_extent==extent:
			index_list = original_index_list
		else:
			# additional_index_list = np.random.choice(original_index_list, size=extent-self.rounded_down_extent, replace=False)			
			additional_index_list = np.random.choice(original_index_list, size=self.training_extent - extent, replace=self.args.replace_samples)			
			index_list = np.concatenate([original_index_list, additional_index_list])		
			
		# 3) Sort based on dataset trajectory length. 
		lengths = self.dataset.dataset_trajectory_lengths[index_list]
		sorted_resampled_indices = np.argsort(lengths)[::-1]

		block_size = batches_per_block * self.args.batch_size

		# 5) Block up, now up till training extent.
		blocks = [index_list[sorted_resampled_indices[i:i+block_size]] for i in range(0, self.training_extent, block_size)]
		# blocks = [sorted_resampled_indices[i:i+block_size] for i in range(0, self.training_extent, block_size)]	
		
		# 6) Shuffle blocks. 
		if shuffle:
			for blk in blocks:			
				np.random.shuffle(blk)

		# 7) Divide blocks. 
		# self.index_list = np.concatenate(blocks)
		self.sorted_indices = np.concatenate(blocks)	

	def random_shuffle(self, extent):
		##########################
		# Set training extents.
		##########################

		# Now that extent is set, create rounded down extent.
		self.rounded_down_extent = extent//self.args.batch_size*self.args.batch_size

		# Training extent:
		if self.rounded_down_extent==extent:
			self.training_extent = self.rounded_down_extent
		else:
			self.training_extent = self.rounded_down_extent+self.args.batch_size

		##########################
		# Now shuffle
		##########################

		original_index_list = np.arange(0,extent)
		if self.rounded_down_extent==extent:
			index_list = original_index_list
		else:
			additional_index_list = np.random.choice(original_index_list, size=self.training_extent-extent, replace=self.args.replace_samples)
			index_list = np.concatenate([original_index_list, additional_index_list])
		np.random.shuffle(index_list)
		self.index_list = index_list

	def shuffle(self, extent, shuffle=True):
	
		if( self.args.data in global_dataset_list ):
			# Random shuffling.
			################################
			# Single element based shuffling because datasets are ordered
			################################
			self.random_shuffle(extent)

		# Task based shuffling.
		# if self.args.task_discriminability or self.args.task_based_supervision or self.args.task_based_shuffling:						
			# self.task_based_shuffling(extent=extent,shuffle=shuffle)							
						