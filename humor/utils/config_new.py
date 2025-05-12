import os
import sys
import yaml
import json
from types import SimpleNamespace
from copy import deepcopy

class ConfigParser:
    """
    Configuration parser that loads settings from YAML or JSON files.
    Maintains hierarchy of base/model/dataset/loss configs similar to the original system.
    Handles default values for missing parameters.
    """
    def __init__(self, config_path):
        """
        Initialize the config parser with a path to a JSON or YAML file.
        
        Args:
            config_path: Path to the config file (.yaml, .yml, or .json)
        """
        self.config_path = config_path
        self.config_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else "."
        
        # Load the config file
        self.config_data = {}
        if config_path:
            if config_path.endswith(('.yaml', '.yml')):
                with open(config_path, 'r') as f:
                    self.config_data = yaml.safe_load(f) or {}
            elif config_path.endswith('.json'):
                with open(config_path, 'r') as f:
                    self.config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}. Use .yaml, .yml, or .json")

       
       
    def _add_defaults(self, stage: str, model: str, dataset: str, loss: str):
        # Define default configurations
        
        if stage == "train":
            self.default_base = {
                'dataset': None,  # Required
                'model': None,    # Required
                'loss': None,     # Optional
                'out': './output',
                'ckpt': None,
                'gpu': 0,
                'batch_size': 8,
                'print_every': 1,
                'epochs': 5,
                'val_every': 1,
                'save_every': 1,
                'lr': 1e-3,
                'beta1': 0.9,
                'beta2': 0.999,
                'eps': 1e-8,
                'sched_milestones': [1],
                'sched_decay': 1.0,
                'decay': 0.0,
                'load_optim': True,
                'use_adam': False,
                'sched_samp_start': None,
                'sched_samp_end': None
            }
        elif stage == "test":
            self.default_base = {
                # Test data configuration
                'shuffle_test': False,         # Shuffles test data
                'test_on_train': False,        # Runs evaluation on TRAINING data
                'test_on_val': False,          # Runs evaluation on VALIDATION data
                
                # Evaluation modes
                'eval_sampling': False,        # Visualizing random sample rollouts
                'eval_sampling_len': 10.0,     # Number of seconds to sample for
                'eval_sampling_debug': False,  # Visualizes random samples in interactive visualization
                'eval_full_test': False,       # Evaluate on the full test set with same metrics as during training
                'eval_num_samples': 1,         # Number of times to sample the model for the same initial state
                'eval_recon': False,           # Visualizes reconstructions of random AMASS sequences
                'eval_recon_debug': False,     # Interactively visualizes reconstructions of random AMASS sequences
                
                # Visualization options
                'viz_contacts': False,         # Body mesh is translucent and contacts are shown on SMPL joint skeleton
                'viz_pred_joints': False,      # HuMoR output joints are visualized
                'viz_smpl_joints': False,      # SMPL joints are visualized
            }
            
        elif stage == "fit":
            self.default_base = {
                # General fitting options
                'data_path': None,
                'data_type': 'AMASS',
                'data_fps': 30,
                'batch_size': 1,
                'shuffle': False,
                'op_keypts': None,

                # AMASS
                'amass_split_by': 'sequence',
                'amass_custom_split': None,
                'amass_batch_size': -1,
                'amass_seq_len': 60,
                'amass_use_joints': False,
                'amass_root_joint_only': False,
                'amass_use_verts': False,
                'amass_use_points': False,
                'amass_noise_std': 0.0,
                'amass_make_partial': False,
                'amass_partial_height': 0.9,
                'amass_drop_middle': False,

                # PROX
                'prox_batch_size': -1,
                'prox_seq_len': 60,
                'prox_recording': None,
                'prox_recording_subseq_idx': -1,

                # iMapper
                'imapper_seq_len': 60,
                'imapper_scene': None,
                'imapper_scene_subseq_idx': -1,

                # RGB
                'rgb_seq_len': None,
                'rgb_overlap_len': None,
                'rgb_intrinsics': None,
                'rgb_planercnn_res': None,
                'rgb_overlap_consist_weight': [0.0, 0.0, 0.0],

                # Common options
                'mask_joints2d': False,

                # Loss weights
                'joint3d_weight': [0.0, 0.0, 0.0],
                'joint3d_rollout_weight': [0.0, 0.0, 0.0],
                'joint3d_smooth_weight': [0.0, 0.0, 0.0],
                'vert3d_weight': [0.0, 0.0, 0.0],
                'point3d_weight': [0.0, 0.0, 0.0],
                'joint2d_weight': [0.0, 0.0, 0.0],
                'pose_prior_weight': [0.0, 0.0, 0.0],
                'shape_prior_weight': [0.0, 0.0, 0.0],
                'motion_prior_weight': [0.0, 0.0, 0.0],
                'init_motion_prior_weight': [0.0, 0.0, 0.0],
                'joint_consistency_weight': [0.0, 0.0, 0.0],
                'bone_length_weight': [0.0, 0.0, 0.0],
                'contact_vel_weight': [0.0, 0.0, 0.0],
                'contact_height_weight': [0.0, 0.0, 0.0],
                'floor_reg_weight': [0.0, 0.0, 0.0],
                'robust_loss': 'bisquare',
                'robust_tuning_const': 4.6851,
                'joint2d_sigma': 100.0,

                # Stage 3 control
                'stage3_tune_init_state': True,
                'stage3_tune_init_num_frames': 15,
                'stage3_tune_init_freeze_start': 30,
                'stage3_tune_init_freeze_end': 55,
                'stage3_contact_refine_only': True,

                # Model & optimization
                'smpl': './body_models/smplh/neutral/model.npz',
                'gt_body_type': 'smplh',
                'vposer': './body_models/vposer_v1_0',
                'openpose': './external/openpose',
                'humor': None,
                'humor_out_rot_rep': 'aa',
                'humor_in_rot_rep': 'mat',
                'humor_latent_size': 48,
                'humor_model_data_config': 'smpl+joints+contacts',
                'humor_steps_in': 1,
                'init_motion_prior': './checkpoints/init_state_prior_gmm',

                'lr': 1.0,
                'num_iters': [30, 80, 70],
                'lbfgs_max_iter': 20,

                # Save options
                'out': None,
                'save_results': False,
                'save_stages_results': False
            }
            
        if model == "HumorModel":
            self.default_model = {
                'out_rot_rep': 'aa',
                'in_rot_rep': 'mat',
                'latent_size': 48,
                'steps_in': 1,
                'conditional_prior': True,
                'output_delta': True,
                'posterior_arch': 'mlp',
                'decoder_arch': 'mlp',
                'prior_arch': 'mlp',
                'model_data_config': 'smpl+joints+contacts',
                'detach_sched_samp': True,
                'model_use_smpl_joint_inputs': False
            }
        elif model == "HumorDiffusion":
            self.default_model = {
                'out_rot_rep': 'aa',
                'in_rot_rep': 'mat',
                'latent_size': 48,
                'steps_in': 1,
                'output_delta': True,
                'model_data_config': 'smpl+joints+contacts',
                'detach_sched_samp': True,
                'model_use_smpl_joint_inputs': False,
                'diffusion_base_channels': 64,
                'diffusion_embed_dim': 256,
                'diffusion_num_layers': 4,
                'encoder_hidden_size': 1024,
                'encoder_num_layers': 4,
                'decoder_hidden_size':1024,
                'decoder_num_layers':4,
            }
        elif model == "MotionVAE":
            self.default_model = {
                'out_rot_rep': 'aa',
                'in_rot_rep': 'mat',
                'latent_size': 48,
                'steps_in': 1,
                'output_delta': True,
                'model_data_config': 'smpl+joints+contacts',
                'detach_sched_samp': True,
                'model_use_smpl_joint_inputs': False
            }
        elif model == "HumorDiffusionTransformer":
            self.default_model = {
                'out_rot_rep': 'aa',
                'in_rot_rep': 'mat',
                'latent_size': 128,
                'steps_in': 1,
                'output_delta': True,
                'model_data_config': 'smpl+joints+contacts',
                'detach_sched_samp': True,
                'model_use_smpl_joint_inputs': False,
                'pose_token_dim': 256,
                'diffusion_base_dim': 256,
                'nhead' : 4,
                'num_layers' : 6,
                'dim_feedforward' : 1024,
                'dropout' : 0.1,
                'cfg_scale' : 4.0,
                'cond_drop_prob': 0.1,
                'use_mean_sample': True,
                'ddim_steps': 100
            }
        else:
            self.default_model = {
                'out_rot_rep': 'aa',
                'in_rot_rep': 'mat',
                'latent_size': 48,
                'steps_in': 1,
                'conditional_prior': True,
                'output_delta': True,
                'model_data_config': 'smpl+joints+contacts',
                'detach_sched_samp': True,
                'model_use_smpl_joint_inputs': False
            }
            
        if dataset == "AmassDiscreteDataset":
            self.default_dataset = {
                'data_paths': None,  # Required
                'split_by': 'dataset',
                'splits_path': None,
                'sample_num_frames': 10,
                'data_rot_rep': 'mat',
                'step_frames_in': 1,
                'step_frames_out': 1,
                'frames_out_step_size': 1,
                'data_return_config': 'smpl+joints+contacts',
                'data_noise_std': 0.0
            }
        
        if loss == "HumorLoss":
            self.default_loss = {
                'kl_loss': 0.0004,
                'kl_loss_anneal_start': 0,
                'kl_loss_anneal_end': 50,
                'kl_loss_cycle_len': -1,
                'regr_trans_loss': 1.0,
                'regr_trans_vel_loss': 1.0,
                'regr_root_orient_loss': 1.0,
                'regr_root_orient_vel_loss': 1.0,
                'regr_pose_loss': 1.0,
                'regr_pose_vel_loss': 1.0,
                'regr_joint_loss': 1.0,
                'regr_joint_vel_loss': 1.0,
                'regr_joint_orient_vel_loss': 1.0,
                'regr_vert_loss': 1.0,
                'regr_vert_vel_loss': 1.0,
                'contacts_loss': 0.01,
                'contacts_vel_loss': 0.01,
                'smpl_joint_loss': 1.0,
                'smpl_mesh_loss': 1.0,
                'smpl_joint_consistency_loss': 1.0,
                'smpl_vert_consistency_loss': 0.0
            }
        elif loss == "DiffusionLoss":
            self.default_loss = {
                'ddpm': 1.0
            }
        
            
    def _merge_with_defaults(self, config, defaults):
        """
        Merge a configuration dictionary with default values.
        
        Args:
            config: Configuration dictionary to merge
            defaults: Default values dictionary
            
        Returns:
            Merged dictionary
        """
        if config is None:
            return deepcopy(defaults)
            
        result = deepcopy(defaults)
        
        defaults_keys = set(defaults.keys())
        config_keys = set(config.keys())
        
        defaults_keys -= config_keys  # Remove keys that are already in config
        print(f"Using default: {defaults_keys}")
        
        for key, value in config.items():
            result[key] = value
        return result
            
    def _load_subconfig(self, config_name):
        """
        Load a sub-configuration file if referenced in the main config.
        
        Args:
            config_name: Name of the sub-config to load
            
        Returns:
            Dict containing the sub-config data or empty dict if not found
        """
        # Check if there's a reference to another file
        if config_name in self.config_data and isinstance(self.config_data[config_name], str):
            subconfig_path = self.config_data[config_name]
            # Handle relative paths
            if not os.path.isabs(subconfig_path):
                subconfig_path = os.path.join(self.config_dir, subconfig_path)
                
            # Load the sub-config file
            if subconfig_path.endswith(('.yaml', '.yml')):
                with open(subconfig_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            elif subconfig_path.endswith('.json'):
                with open(subconfig_path, 'r') as f:
                    return json.load(f)
        # Check if it's directly embedded in the main config
        elif config_name in self.config_data and isinstance(self.config_data[config_name], dict):
            return self.config_data[config_name]
            
        return {}
        
    def parse(self, stage="train"):
        """
        Parse the configuration file and return an Args object with the same
        structure as the original command-line parser.
        
        Returns:
            Args object containing base/model/dataset/loss configurations
        """
        # Get base config and merge with defaults
        base_config = self.config_data.get('base', {})
        model_name = base_config.get('model', '')
        dataset_name = base_config.get('dataset', '')
        loss_name = base_config.get('loss', '')
        self._add_defaults(stage=stage, model=model_name, dataset=dataset_name, loss=loss_name)
        
        base_config = self._merge_with_defaults(base_config, self.default_base)
        base_args = SimpleNamespace(**base_config)
        
        model_config, dataset_config, loss_config = None, None, None
        if stage == "train":
            # Require some parameters
            if base_args.dataset is None:
                raise ValueError("Required parameter 'dataset' not provided in config")
            if base_args.model is None:
                raise ValueError("Required parameter 'model' not provided in config")
            
            # Load model config
            if base_args.model is not None:
                model_data = self._load_subconfig('model')
                model_data = self._merge_with_defaults(model_data, self.default_model)
                model_config = SimpleNamespace(**model_data)
            
            # Load dataset config
            if base_args.dataset is not None:
                dataset_data = self._load_subconfig('dataset')
                dataset_data = self._merge_with_defaults(dataset_data, self.default_dataset)
            
                # Require data_paths for dataset
                if dataset_data['data_paths'] is None:
                    raise ValueError("Required parameter 'data_paths' not provided in dataset config")
                
                dataset_config = SimpleNamespace(**dataset_data)
                    
            # Load loss config if specified
            loss_config = None
            if base_args.loss is not None:
                loss_data = self._load_subconfig('loss')
                loss_data = self._merge_with_defaults(loss_data, self.default_loss)
                loss_config = SimpleNamespace(**loss_data)
        
        # Create Args object
        args = Args(base_args, model=model_config, dataset=dataset_config, loss=loss_config)
        return args, []  # Return empty list for unknown args to match original API

class Args:
    """
    Container class identical to the original Args class to maintain compatibility.
    """
    def __init__(self, base, model=None, dataset=None, loss=None):
        self.base = base
        self.model = model
        self.dataset = dataset
        self.loss = loss

        # Dictionary versions of args that can be used to pass as constructor arguments
        self.base_dict = vars(self.base) if self.base is not None else None
        self.model_dict = vars(self.model) if self.model is not None else None
        self.dataset_dict = vars(self.dataset) if self.dataset is not None else None
        self.loss_dict = vars(self.loss) if self.loss is not None else None