from Utils.headers import *
from Datasets import DataLoaders, DAPG_DataLoader, DexMV_DataLoader, RealWorldRigid_DataLoader, MAGI_DataLoader

from PolicyManagers.BatchPretrain import PolicyManager_BatchPretrain
# import TestClass
# import faulthandler



def return_dataset(args, data=None, create_dataset_variation=False):
	
	# The data parameter overrides the data in args.data. 
	# This is so that we can call return_dataset with source and target data for transfer setting.
	# print("args.data: ", args.data)
	if data is not None:
		args.data = data
	
	elif args.data in ["MAGIPreproc"]:
		dataset = MAGI_DataLoader.MAGI_PreDataset( args )
	elif args.data in ["MAGI"]:
		dataset = MAGI_DataLoader.MAGI_Dataset( args )

	return dataset

class Master():

	def __init__(self, arguments):
		self.args = arguments 

		print("Creating Datasets")			
		self.dataset = return_dataset(self.args, create_dataset_variation=self.args.dataset_variation)

		if self.args.setting=='pretrain_sub':
			self.policy_manager = PolicyManager_BatchPretrain(self.dataset, self.args)

		if self.args.debug:
			print("Embedding in Master.")
			embed()
			
		# Create networks and training operations. 
		self.policy_manager.setup()

	def run(self):
		if self.args.setting in ['pretrain_sub']:
			if self.args.train:
				if self.args.model:
					self.policy_manager.train(self.args.model)
				else:
					self.policy_manager.train()
			else:			
				self.policy_manager.evaluate(model=self.args.model)		

	def test(self):
		if self.args.test_code:
			loader = TestClass.TestLoaderWithKwargs()
			suite = loader.loadTestsFromTestCase(TestClass.MetaTestClass, policy_manager=self.policy_manager)
			unittest.TextTestRunner().run(suite)

def parse_arguments():
	parser = argparse.ArgumentParser(description='Learning Skills from Demonstrations')

	# Setup training. 
	# parser.add_argument('--datadir', dest='datadir',type=str,default='../../Data/Datasets/ContData/')
	parser.add_argument('--datadir', dest='datadir',type=str,default=None)
	parser.add_argument('--train',dest='train',type=int,default=0)
	parser.add_argument('--debug',dest='debug',type=int,default=0)
	parser.add_argument('--debug_RL',dest='debug_RL',type=int,default=0)
	parser.add_argument('--notes',dest='notes',type=str)
	parser.add_argument('--name',dest='name',type=str,default=None)
	parser.add_argument('--fake_batch_size',dest='fake_batch_size',type=int,default=1)
	parser.add_argument('--batch_size',dest='batch_size',type=int,default=32)
	parser.add_argument('--training_phase_size',dest='training_phase_size',type=int,default=500000)
	parser.add_argument('--initial_counter_value',dest='initial_counter_value',type=int,default=0)
	parser.add_argument('--data',dest='data',type=str,default='Continuous')
	parser.add_argument('--setting',dest='setting',type=str,default='gtsub')
	parser.add_argument('--test_code',dest='test_code',type=int,default=0)
	parser.add_argument('--model',dest='model',type=str)
	parser.add_argument('--RL_model_path',dest='RL_model_path',type=str,help='High level policy model.')
	# parser.add_argument('--logdir',dest='logdir',type=str,default='Experiment_Logs/')
	parser.add_argument('--logdir',dest='logdir',type=str,default='ExpWandbLogs/')
	parser.add_argument('--epochs',dest='epochs',type=int,default=500) # Number of epochs to train for. Reduce for Mocap.
	parser.add_argument('--debugging_datapoints',dest='debugging_datapoints',type=int,default=-1,help='How many data points to run training on. If greater than 0, only select that many datapoints for debugging.')
	parser.add_argument('--seed',dest='seed',type=int,default=0,help='Seed value to initialize random processes.')
	parser.add_argument('--replace_samples', dest='replace_samples',type=bool,default=False,help='Use replace=true when sampling trajectories for training.')

	# Training setting. 
	parser.add_argument('--discrete_z',dest='discrete_z',type=int,default=0)
	# parser.add_argument('--transformer',dest='transformer',type=int,default=0)	
	parser.add_argument('--z_dimensions',dest='z_dimensions',type=int,default=64)
	parser.add_argument('--number_layers',dest='number_layers',type=int,default=5)
	parser.add_argument('--hidden_size',dest='hidden_size',type=int,default=64)
	parser.add_argument('--var_number_layers',dest='var_number_layers',type=int,default=5)
	parser.add_argument('--var_hidden_size',dest='var_hidden_size',type=int,default=64)
	parser.add_argument('--dropout',dest='dropout',type=float,default=0.,help='Whether to set dropout.') 
	parser.add_argument('--mlp_dropout',dest='mlp_dropout',type=float,default=0.,help='Whether to set dropout.') 
	parser.add_argument('--batch_norm',dest='batch_norm',type=int,default=0,help='Whether to use batch norm.')
	parser.add_argument('--leaky_relu',dest='leaky_relu',type=int,default=0,help='Whether to use leaky relu (or just vanilla relu).')
	parser.add_argument('--environment',dest='environment',type=str,default='SawyerLift') # Defines robosuite environment for RL.
	parser.add_argument('--target_environment',dest='target_environment',type=str,default='SawyerLift') # Defines robosuite environment for RL.
	
	# Variance parameters. 
	parser.add_argument('--variance_factor',dest='variance_factor',type=float,default=0.01,help='Factor by which to multiple variance value predicted by network.')
	# parser.add_argument('--constant_variance',dest='constant_variance',type=int,default=0,help='Whether to use constant variance')
	parser.add_argument('--variance_mode',dest='variance_mode',type=str,default='Learned',choices=['Learned','Constant','LinearAnnealed','QuadraticAnnealed'],help='What mode to set variance of policy network')	
	parser.add_argument('--variance_value',dest='variance_value',type=float,default=0.1,help='Variance value for network distributions.')
	parser.add_argument('--epsilon_scale_factor',dest='epsilon_scale_factor',type=float,default=100.,help='Factor by which to scale variance down for variance.')
	# Setting parameters for linear annealing of variance.
	parser.add_argument('--initial_policy_variance',dest='initial_policy_variance',type=float,default=0.1)
	parser.add_argument('--final_policy_variance',dest='final_policy_variance',type=float,default=0.0001)
	parser.add_argument('--policy_variance_decay_over',dest='policy_variance_decay_over',type=int,default=200)


	# Data parameters. 
	parser.add_argument('--traj_segments',dest='traj_segments',type=int,default=1) # Defines whether to use trajectory segments for pretraining or entire trajectories. Useful for baseline implementation.
	parser.add_argument('--gripper',dest='gripper',type=int,default=1) # Whether to use gripper training in roboturk.
	parser.add_argument('--ee_trajectories',dest='ee_trajectories',type=int,default=0,help='Whether to learn a skill space from end effector trajectories rather than joint space.')
	parser.add_argument('--single_hand',dest='single_hand',type=str,default=None,help='Whether to use a single hand, if so, which hand. Only for MIME dataset.')
	parser.add_argument('--ds_freq',dest='ds_freq',type=float,default=1.) # Additional downsample frequency.
	parser.add_argument('--condition_size',dest='condition_size',type=int,default=4)
	parser.add_argument('--smoothen', dest='smoothen',type=int,default=0) # Whether to smoothen the original dataset. 
	parser.add_argument('--smoothing_kernel_bandwidth', dest='smoothing_kernel_bandwidth',type=float,default=3.5) # The smoothing bandwidth that is applied to data loader trajectories. 
	parser.add_argument('--human_pos_normalization', dest='position_normalization', type=str, default='none') # The position normalization for GRAB dataloader
	parser.add_argument('--human_angular_data', dest='angular_data', type=bool, default=False) # Angular transformations for the GRABHand Dataset
	parser.add_argument('--skip_wrist',dest='skip_wrist',type=int,default=0,help='Whether or not to skip the wrist joint.')

	# Training paradigm parameters. 
	parser.add_argument('--new_gradient',dest='new_gradient',type=int,default=1)
	parser.add_argument('--b_prior',dest='b_prior',type=int,default=1)
	parser.add_argument('--constrained_b_prior',dest='constrained_b_prior',type=int,default=1) # Whether to use constrained b prior var network or just normal b prior one.
	parser.add_argument('--reparam',dest='reparam',type=int,default=1)	
	parser.add_argument('--number_policies',dest='number_policies',type=int,default=4)
	parser.add_argument('--fix_subpolicy',dest='fix_subpolicy',type=int,default=1)
	parser.add_argument('--task_based_shuffling',dest='task_based_shuffling',type=int,default=0,help='Whether to use task based shuffling.')
	parser.add_argument('--reset_training',dest='reset_training',type=int,default=0,help='Whether to reset subpolicy training for joint training, used mostly to learn contextual representations.')
	parser.add_argument('--train_only_policy',dest='train_only_policy',type=int,default=0) # Train only the policy network and use a pretrained encoder. This is weird but whatever. 
	parser.add_argument('--load_latent',dest='load_latent',type=int,default=1) # Whether to load latent policy from model or not.
	parser.add_argument('--subpolicy_model',dest='subpolicy_model',type=str)
	parser.add_argument('--traj_length',dest='traj_length',type=int,default=10)
	parser.add_argument('--short_trajectories',dest='short_trajectories',type=int,default=0,help='Whether to restrict training to short trajectories, to massively save GPU memory.')
	parser.add_argument('--dataset_traj_length_limit',dest='dataset_traj_length_limit',type=int,default=500,help='Value to restrict dataset size to.')
	parser.add_argument('--skill_length',dest='skill_length',type=int,default=5)
	parser.add_argument('--var_skill_length',dest='var_skill_length',type=int,default=1)

	# Parameters for evaluation. 
	parser.add_argument('--display_freq',dest='display_freq',type=int,default=10000)
	parser.add_argument('--save_freq',dest='save_freq',type=int,default=5)	
	parser.add_argument('--eval_freq',dest='eval_freq',type=int,default=20)	
	parser.add_argument('--metric_eval_freq',dest='metric_eval_freq',type=int,default=10000)	
	parser.add_argument('--N_trajectories_to_visualize',dest='N_trajectories_to_visualize',type=int,default=100,help='How many trajectories to visualize during embedding visualization.')
	parser.add_argument('--perplexity',dest='perplexity',type=float,default=30,help='Value of perplexity fed to TSNE.')
	parser.add_argument('--latent_set_file_path',dest='latent_set_file_path',type=str,help='File path to pre-computed latent sets to visualize.')
	parser.add_argument('--viz_latent_rollout',dest='viz_latent_rollout',type=int,default=0,help='Whether to visualize latent rollout or not.')
	parser.add_argument('--viz_sim_rollout',dest='viz_sim_rollout',type=int,default=0,help='Whether to visualize rollout by magically setting state, or stepping in the environment.')
	parser.add_argument('--viz_gt_sim_rollout',dest='viz_gt_sim_rollout',type=int,default=0,help='Whether or not to visualize the GT trajectory by replaying through the simulator.')
	parser.add_argument('--sim_viz_action_scale_factor',dest='sim_viz_action_scale_factor',type=float,default=0.3,help='Factor by which to scale actions when visualizing in simulation env.')
	parser.add_argument('--sim_viz_step_repetition',dest='sim_viz_step_repetition',type=int,default=20,help='Number of times to repeat simulation step of visualization of traj.')

	parser.add_argument('--entropy',dest='entropy',type=int,default=0)
	parser.add_argument('--var_entropy',dest='var_entropy',type=int,default=0)
	parser.add_argument('--ent_weight',dest='ent_weight',type=float,default=0.)
	parser.add_argument('--var_ent_weight',dest='var_ent_weight',type=float,default=2.)
	
	parser.add_argument('--pretrain_bias_sampling',type=float,default=0.) # Defines percentage of trajectory within which to sample trajectory segments for pretraining.
	parser.add_argument('--pretrain_bias_sampling_prob',type=float,default=0.)
	parser.add_argument('--action_scale_factor',type=float,default=1.)	
	parser.add_argument('--state_scale_factor',dest='state_scale_factor',type=float,default=1.)	

	parser.add_argument('--z_exploration_bias',dest='z_exploration_bias',type=float,default=0.)
	parser.add_argument('--b_exploration_bias',dest='b_exploration_bias',type=float,default=0.)
	parser.add_argument('--lat_z_wt',dest='lat_z_wt',type=float,default=0.1)
	parser.add_argument('--lat_b_wt',dest='lat_b_wt',type=float,default=1.)
	parser.add_argument('--z_probability_factor',dest='z_probability_factor',type=float,default=0.1)
	parser.add_argument('--b_probability_factor',dest='b_probability_factor',type=float,default=0.01)
	parser.add_argument('--subpolicy_clamp_value',dest='subpolicy_clamp_value',type=float,default=-5)
	parser.add_argument('--latent_clamp_value',dest='latent_clamp_value',type=float,default=-5)
	parser.add_argument('--min_variance_bias',dest='min_variance_bias',type=float,default=0.01)
	parser.add_argument('--normalization',dest='normalization',type=str,default=None)
	parser.add_argument('--regularization_weight',dest='regularization_weight',type=float,default=0.,help='Value of regularization weight to be added to the model.')

	parser.add_argument('--likelihood_penalty',dest='likelihood_penalty',type=int,default=10)
	parser.add_argument('--subpolicy_ratio',dest='subpolicy_ratio',type=float,default=0.01)
	parser.add_argument('--latentpolicy_ratio',dest='latentpolicy_ratio',type=float,default=0.1)
	parser.add_argument('--temporal_latentpolicy_ratio',dest='temporal_latentpolicy_ratio',type=float,default=0.)
	parser.add_argument('--latent_loss_weight',dest='latent_loss_weight',type=float,default=0.1)	
	parser.add_argument('--var_loss_weight',dest='var_loss_weight',type=float,default=1.)
	parser.add_argument('--prior_weight',dest='prior_weight',type=float,default=0.00001)
	# parser.add_argument('--context_loss_weight',dest='context_loss_weight',type=float,default=1.,help='Weight of context loss.')
	parser.add_argument('--kl_weight',dest='kl_weight',type=float,default=0.01,help='KL weight when constant.')
	parser.add_argument('--kl_schedule',dest='kl_schedule',type=str,default=None,choices=[None, 'Monotonic', 'Cyclic'],help='Whether to schedule KL weight.')
	parser.add_argument('--initial_kl_weight',dest='initial_kl_weight',type=float,default=0.0,help='Initial KL weight.')
	parser.add_argument('--final_kl_weight',dest='final_kl_weight',type=float,default=1.0,help='Initial KL weight.')
	parser.add_argument('--kl_increment_epochs',dest='kl_increment_epochs',type=int,default=100,help='Number of epochs to increment KL over.')
	parser.add_argument('--kl_begin_increment_epochs',dest='kl_begin_increment_epochs',type=int,default=100,help='Number of epochs after which to increment KL.')
	parser.add_argument('--kl_cyclic_phase_epochs',dest='kl_cyclic_phase_epochs',type=int,default=100,help='Number of epochs to cycle KL weight over.')	

	# architecture
	parser.add_argument('--split_stream_encoder',dest='split_stream_encoder',type=int,default=0,help='Whether to use split stream encoder or not.')
	parser.add_argument('--embedding_visualization_stream',dest='embedding_visualization_stream',type=str,default=None,help='Which stream to use to embed and visualize Z space.')
	parser.add_argument('--robot_state_size',dest='robot_state_size',type=int,default=8,help='Default robot state size.')
	parser.add_argument('--env_state_size',dest='env_state_size',type=int,default=7,help='Default environment state size.')
	parser.add_argument('--object_pure_relative_state',dest='object_pure_relative_state',type=int,default=0,help='Whether or not to use pure relative state for env abstraction input. ')
	parser.add_argument('--soft_object', dest='soft_object', type=int, default=0, help='Whether or not we are learning with deformable objects.')
	parser.add_argument('--images_in_real_world_dataset', dest='images_in_real_world_dataset', type=int, default=0, help='Whether to dela with images in the realworld datasset.')

	# Relative state reconstruction loss.
	parser.add_argument('--relative_state_reconstruction_loss_weight', dest='relative_state_reconstruction_loss_weight', type=float, default=0., help='What weight to place on the relative state reconstruction loss in the robot-object setting..')	
	parser.add_argument('--relative_state_phase_aux_loss_weight', dest='relative_state_phase_aux_loss_weight', type=float, default=0., help='Weight to place on the relative state phase aux loss.')
	parser.add_argument('--task_based_aux_loss_weight', dest='task_based_aux_loss_weight', type=float, default=0., help='Weight to place on task based auxillary loss.')
	parser.add_argument('--negative_task_based_component_weight', dest='negative_task_based_component_weight', type=float, default=1., help='Weight to place on the negative component of the task based aux loss.')
	parser.add_argument('--pairwise_z_distance_threshold', dest='pairwise_z_distance_threshold', type=float, default=2., help='Minimum distance to push apart different parts of latent space that are semantically different.')

	# absolute state reconstruction
	parser.add_argument('--cummulative_computed_state_reconstruction_loss_weight', dest='cummulative_computed_state_reconstruction_loss_weight', type=float, default=0., \
					 help='Weight to place on the cummulative_computed_state_reconstruction_loss.')
	parser.add_argument('--teacher_forced_state_reconstruction_loss_weight', dest='teacher_forced_state_reconstruction_loss_weight', type=float, default=0.,\
					 help='Weight to set on teacher_forced_state_reconstruction_loss_weight. ')

	# Cross Domain Skill Transfer parameters. 
	parser.add_argument('--discriminability_weight',dest='discriminability_weight',type=float,default=1.,help='Weight of discriminability loss in cross domain skill transfer.') 
	parser.add_argument('--discriminator_weight',dest='discriminator_weight',type=float,default=1.,help='Weight of z discriminator loss.')
	parser.add_argument('--vae_loss_weight',dest='vae_loss_weight',type=float,default=1.,help='Weight of VAE loss in cross domain skill transfer.') 	
	parser.add_argument('--alternating_phase_size',dest='alternating_phase_size',type=int,default=2000, help='Size of alternating training phases.')
	parser.add_argument('--discriminator_phase_size',dest='discriminator_phase_size',type=int,default=2,help='Factor by which to train discriminator more than generator.')
	parser.add_argument('--generator_phase_size',dest='generator_phase_size',type=int,default=1,help='Factor by which to train generaotr more than discriminator.')
	parser.add_argument('--cycle_reconstruction_loss_weight',dest='cycle_reconstruction_loss_weight',type=float,default=1.,help='Weight of the cycle-consistency reconstruction loss term.')
	parser.add_argument('--real_translated_discriminator',dest='real_translated_discriminator',type=int,default=0,help='Whether to include real-translated discriminator based losses.')
	parser.add_argument('--real_trans_loss_weight',dest='real_trans_loss_weight',type=float,default=1.,help='Weight of discriminability loss between real and (cycle) translated trajectories.')
	parser.add_argument('--z_transform_discriminator',dest='z_transform_discriminator',type=int,default=0,help='Whether to use z transform discriminators.')
	parser.add_argument('--z_trajectory_discriminator',dest='z_trajectory_discriminator',type=int,default=0,help='Whether to use z trajectory discriminators.')
	parser.add_argument('--z_trajectory_discriminability_weight',dest='z_trajectory_discriminability_weight',type=float,default=1.,help='Weight of z trajectory discriminability loss.')
	parser.add_argument('--z_trajectory_discriminator_weight',dest='z_trajectory_discriminator_weight',type=float,default=1.,help='Weight of z trajectory discriminator loss.')	
	parser.add_argument('--max_viz_trajs',dest='max_viz_trajs',type=int,default=5,help='How many trajectories to visualize.')
	parser.add_argument('--z_transform_or_tuple',dest='z_transform_or_tuple',type=int,default=0,help='Whether to use the z transform or z tuples.')	
	parser.add_argument('--ignore_last_z_transform',dest='ignore_last_z_transform',type=int,default=0,help='Whether to ignore or last z transform.')
	parser.add_argument('--number_of_visualized_translations',dest='number_of_visualized_translations',type=int,default=2, help='How many pairs of translations to visualize.')

	# Exploration and learning rate parameters. 
	parser.add_argument('--epsilon_from',dest='epsilon_from',type=float,default=0.3)
	parser.add_argument('--epsilon_to',dest='epsilon_to',type=float,default=0.05)
	parser.add_argument('--epsilon_over',dest='epsilon_over',type=int,default=200)
	parser.add_argument('--learning_rate',dest='learning_rate',type=float,default=1e-4)
	parser.add_argument('--transfer_learning_rate',dest='transfer_learning_rate',type=float,default=1e-4)

	# Baseline parameters. 
	parser.add_argument('--baseline_kernels',dest='baseline_kernels',type=int,default=15)
	parser.add_argument('--baseline_window',dest='baseline_window',type=int,default=15)
	parser.add_argument('--baseline_kernel_bandwidth',dest='baseline_kernel_bandwidth',type=float,default=3.5)

	# Reinforcement Learning parameters. 
	parser.add_argument('--TD',dest='TD',type=int,default=0) # Whether or not to use Temporal difference while training the critic network.
	parser.add_argument('--OU',dest='OU',type=int,default=1) # Whether or not to use the Ornstein Uhlenbeck noise process while training.
	parser.add_argument('--OU_max_sigma',dest='OU_max_sigma',type=float,default=0.2) # Max Sigma value of the Ornstein Uhlenbeck noise process.
	parser.add_argument('--OU_min_sigma',dest='OU_min_sigma',type=float,default=0.2) # Min Sigma value of the Ornstein Uhlenbeck noise process.
	parser.add_argument('--MLP_policy',dest='MLP_policy',type=int,default=0) # Whether or not to use MLP policy.
	parser.add_argument('--mean_nonlinearity',dest='mean_nonlinearity',type=int,default=0) # Whether or not to use Tanh activation.
	parser.add_argument('--burn_in_eps',dest='burn_in_eps',type=int,default=500) # How many epsiodes to burn in.
	parser.add_argument('--random_memory_burn_in',dest='random_memory_burn_in',type=int,default=1) # Whether to burn in episodes into memory randomly or not.
	parser.add_argument('--shaped_reward',dest='shaped_reward',type=int,default=0) # Whether or not to use shaped rewards.
	parser.add_argument('--memory_size',dest='memory_size',type=int,default=2000) # Size of replay memory. 2000 is okay, but is still kind of short sighted. 
	parser.add_argument('--no_mujoco',dest='no_mujoco',type=int,default=0,help='Whether we have mujoco installation or not.')

	# Transfer learning domains, etc. 
	parser.add_argument('--source_domain',dest='source_domain',type=str,help='What the source domain is in transfer.')
	parser.add_argument('--target_domain',dest='target_domain',type=str,help='What the target domain is in transfer.')
	parser.add_argument('--source_datadir',dest='source_datadir',type=str,default=None,help='What the directory of the source dataset is. Defaults to args.datadir.')
	parser.add_argument('--target_datadir',dest='target_datadir',type=str,default=None,help='What the directory of the target dataset is. Defaults to args.datadir.')
	parser.add_argument('--source_single_hand',dest='source_single_hand',type=str,default=None,help='Whether to use a single hand for each domain, if so, which hand. Only for MIME dataset.')
	parser.add_argument('--target_single_hand',dest='target_single_hand',type=str,default=None,help='Whether to use a single hand for each domain, if so, which hand. Only for MIME dataset.')
	parser.add_argument('--source_ee_trajs',dest='source_ee_trajs',type=int,default=0,help='Whether to use EE trajectories in source domain or not.')
	parser.add_argument('--target_ee_trajs',dest='target_ee_trajs',type=int,default=0,help='Whether to use EE trajectories in target domain or not.')
	parser.add_argument('--source_model',dest='source_model',type=str,help='What model to use for the source domain.',default=None)
	parser.add_argument('--target_model',dest='target_model',type=str,help='What model to use for the target domain.',default=None)
	parser.add_argument('--source_subpolicy_model',dest='source_subpolicy_model',type=str,help='What subpolicy model to use for the source domain.',default=None)
	parser.add_argument('--target_subpolicy_model',dest='target_subpolicy_model',type=str,help='What subpolicy model to use for the target domain.',default=None)
	parser.add_argument('--fix_source',dest='fix_source',type=int,default=0,help='Whether to fix source domain representation.')
	parser.add_argument('--fix_target',dest='fix_target',type=int,default=0,help='Whether to fix target domain representation.')
	parser.add_argument('--load_from_transfer',dest='load_from_transfer',type=int,default=0,help='Whether we are loading joint model from transfer training.')
	parser.add_argument('--dataset_variation',dest='dataset_variation',type=int,default=0,help='Whether to use flipped or original version of the dataset.')
	# parser.add_argument('--reset_subpolicy_training',dest='reset_subpolicy_training',type=int,default=1,help='Whether to reset subpolicy training.')
	parser.add_argument('--residual_translation',dest='residual_translation',type=int,default=0,help='Whether to use a residual model for translation or just a regular one.')
	parser.add_argument('--small_translation_model',dest='small_translation_model',type=int,default=0,help='Whether to use a small model for translation or not. Restricts network capacity.')
	parser.add_argument('--recurrent_translation',dest='recurrent_translation',type=int,default=0,help='Whether to implement a recurrent translation model.')
	parser.add_argument('--input_corruption_noise',dest='input_corruption_noise',type=float,default=0.,help='How much noise to add to the input to corrupt it. Default no corruption.')
	parser.add_argument('--subpolicy_input_dropout',dest='subpolicy_input_dropout',type=float,default=0.,help='What fraction to set subpolicy dropout to.') 

	parser.add_argument('--equivariance',dest='equivariance',type=int,default=0,help='Whether to implement equivariance objective in (Joint) Fix Embed setting.')
	parser.add_argument('--equivariance_loss_weight',dest='equivariance_loss_weight',type=float,default=1.,help='Weight associated with the equivariance loss.')
	parser.add_argument('--cross_domain_supervision',dest='cross_domain_supervision',type=int,default=0,help='Whether to use cross domain supervision when operating in pair of same domains.')
	parser.add_argument('--cross_domain_supervision_loss_weight',dest='cross_domain_supervision_loss_weight',type=float,default=0.,help='Weight associated with the cross domain supervision loss.')
	parser.add_argument('--new_supervision',dest='new_supervision',type=int,default=1,help='Whether to use new form of supervision or not.')
	
	parser.add_argument('--number_of_supervised_datapoints',dest='number_of_supervised_datapoints',type=int,default=0,help='Number of supervised datapoints to use in training.')
	parser.add_argument('--cycle_cross_domain_supervision_loss_weight',dest='cycle_cross_domain_supervision_loss_weight',type=float,default=0.,help='Weight associated with the cycle cross domain supervision loss.')
	parser.add_argument('--z_normalization',dest='z_normalization',type=str,default=None,choices=[None, 'global','ind'],help='What normalization to use for zs.')

	# Density loss terms
	parser.add_argument('--supervised_set_based_density_loss',dest='supervised_set_based_density_loss',type=int,default=0,help='Whether to use the supervised_set_based_density_loss to train.')
	parser.add_argument('--supervised_set_based_density_loss_weight',dest='supervised_set_based_density_loss_weight',type=float,default=0.,help='Weight associated with the supervised_set_based_density_loss.')
	parser.add_argument('--cross_domain_density_loss_weight',dest='cross_domain_density_loss_weight',type=float,default=0.,help='Weight associated with the cross domain density loss.')
	parser.add_argument('--cross_domain_z_tuple_density_loss_weight',dest='cross_domain_z_tuple_density_loss_weight',type=float,default=0.,help='Weight associated with the z tuple cross domain density loss.')	
	parser.add_argument('--forward_density_loss_weight',dest='forward_density_loss_weight',type=float,default=0.,help='Weight associated with forward density loss.')
	parser.add_argument('--backward_density_loss_weight',dest='backward_density_loss_weight',type=float,default=0.,help='Weight associated with backward density loss.')
	parser.add_argument('--forward_tuple_density_loss_weight',dest='forward_tuple_density_loss_weight',type=float,default=0.,help='Weight associated with forward tuple density loss.')
	parser.add_argument('--backward_tuple_density_loss_weight',dest='backward_tuple_density_loss_weight',type=float,default=0.,help='Weight associated with backward tuple density loss.')	
	parser.add_argument('--gmm_variance_value', dest='gmm_variance_value', type=float, default=0.5,help='Variance value for GMM components.')
	parser.add_argument('--gmm_tuple_variance_value', dest='gmm_tuple_variance_value', type=float, default=0.5,help='Variance value for Z Tuple GMM components.')
	parser.add_argument('--z_tuple_gmm',dest='z_tuple_gmm',type=int,default=0,help='Whether to use a Z Tuple GMM or not.')
	parser.add_argument('--z_gmm',dest='z_gmm',type=int,default=0,help='Whether to use a Z GMM or not.')

	# Task ID based discriminability
	parser.add_argument('--task_discriminability',dest='task_discriminability',type=int,default=0,help='Whether or not to implement task based discriminability.')
	parser.add_argument('--number_of_tasks',dest='number_of_tasks',type=int,default=0,help='Number of tasks to be considered in task based discriminability.')
	parser.add_argument('--task_discriminability_loss_weight',dest='task_discriminability_loss_weight',type=float,default=0.,help='Loss weight associated with task based discriminability.')
	parser.add_argument('--task_discriminator_weight',dest='task_discriminator_weight',type=float,default=0.,help='Loss weight associated with task discriminator(s)')
	parser.add_argument('--task_based_supervision',dest='task_based_supervision',type=int,default=0,help='Whether or not we are using task based supervision.')

	parser.add_argument('--use_wandb',dest='use_wandb',type=int,default=1,help='Whether or not we are using wandb')


	parser.add_argument('--test_set_size',dest="test_set_size",type=int,default=0,help='test_set_size')
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	master = Master(args)

	if( args.use_wandb ):
		wandb.init(project=args.setting, dir=args.logdir, name=args.name)
		wandb.config.update(args)
	master.run()
	print("done !!!")
	print("done !!!")
	print("done !!!")
	
if __name__=='__main__':
	main(sys.argv)





