# Databricks notebook source
import yaml
from typing import Dict, Any
import os
from pathlib import Path

def validate_feature_config(features: Dict[str, Any]) -> None:
    """Validate feature configuration."""
    if 'enabled_features' not in features:
        raise ValueError("Missing 'enabled_features' in features configuration")
    
    if 'feature_configs' not in features:
        raise ValueError("Missing 'feature_configs' in features configuration")
    
    required_fields = {
        'pretrained_embedding': ['type', 'dim'],
        'sequence': ['type', 'input_dim', 'dim'],
        'text': ['type', 'dim', 'model_name'],
        'numeric': ['type', 'dim'],
        'categorical': ['type', 'dim', 'num_categories'],
        'binary_embedding': ['type', 'dim']
    }
    
    for feature_name, config in features['feature_configs'].items():
        if not isinstance(config, dict):
            raise ValueError(f"Invalid configuration for feature '{feature_name}'")
            
        if 'enabled' not in config:
            raise ValueError(f"Missing 'enabled' field for feature '{feature_name}'")
            
        if not config['enabled']:
            continue
            
        feature_type = config.get('type')
        if feature_type not in required_fields:
            raise ValueError(f"Invalid feature type '{feature_type}' for feature '{feature_name}'")
        
        for field in required_fields[feature_type]:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' for feature '{feature_name}'")

def validate_task_config(task_config: Dict[str, Dict[str, Any]]) -> None:
    """Validate task configuration."""
    required_fields = ['weight', 'enabled', 'loss', 'metrics']
    enabled_tasks = [task for task, config in task_config.items() if config.get('enabled', True)]
    
    if not enabled_tasks:
        raise ValueError("At least one task must be enabled")
    
    total_weight = 0
    for task_name, config in task_config.items():
        if not config.get('enabled', True):
            continue
            
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' for task '{task_name}'")
        
        if not 0 <= config['weight'] <= 1:
            raise ValueError(f"Task weight must be between 0 and 1 for task '{task_name}'")
            
        if config['loss'] not in ['bce']:
            raise ValueError(f"Unsupported loss type '{config['loss']}' for task '{task_name}'")
            
        if not isinstance(config['metrics'], list):
            raise ValueError(f"Metrics must be a list for task '{task_name}'")
            
        total_weight += config['weight']
    
    if not 0.99 <= total_weight <= 1.01:  # Allow for small floating point errors
        raise ValueError(f"Weights of enabled tasks must sum to 1, got {total_weight}")

def validate_model_config(model_config: Dict[str, Any]) -> None:
    """Validate model configuration."""
    if 'architecture' not in model_config:
        raise ValueError("Missing 'architecture' in model configuration")
        
    if model_config['architecture'] not in ['dcn_v2', 'dcn_v2_moe']:
        raise ValueError(f"Unsupported model architecture: {model_config['architecture']}")
        
    if 'dcn_config' not in model_config:
        raise ValueError("Missing 'dcn_config' in model configuration")
        
    dcn_config = model_config['dcn_config']
    required_fields = [
        'cross_layers',
        'hidden_layers',
        'task_hidden_layers',
        'dropout_rate',
        'activation',
        'use_batch_norm',
        'cross_dropout',
        'stochastic_depth_rate'
    ]
    
    for field in required_fields:
        if field not in dcn_config:
            raise ValueError(f"Missing required field '{field}' in DCN configuration")
    
    if dcn_config['dropout_rate'] < 0 or dcn_config['dropout_rate'] > 1:
        raise ValueError("Dropout rate must be between 0 and 1")
        
    if dcn_config['cross_dropout'] < 0 or dcn_config['cross_dropout'] > 1:
        raise ValueError("Cross dropout rate must be between 0 and 1")
        
    if dcn_config['stochastic_depth_rate'] < 0 or dcn_config['stochastic_depth_rate'] > 1:
        raise ValueError("Stochastic depth rate must be between 0 and 1")

    # Validate MoE configuration if enabled
    if model_config['architecture'] == 'dcn_v2_moe':
        if 'moe_config' not in model_config:
            raise ValueError("Missing 'moe_config' for MoE architecture")
            
        moe_config = model_config['moe_config']
        required_moe_fields = [
            'enabled',
            'num_experts',
            'expert_hidden_size',
            'gating_hidden_size',
            'load_balancing_loss_weight',
            'expert_dropout',
            'use_region_specific_experts',
            'knowledge_distillation_weight',
            'training_stages'
        ]
        
        for field in required_moe_fields:
            if field not in moe_config:
                raise ValueError(f"Missing required field '{field}' in MoE configuration")
        
        if not isinstance(moe_config['num_experts'], int) or moe_config['num_experts'] < 1:
            raise ValueError("num_experts must be a positive integer")
            
        if moe_config['expert_dropout'] < 0 or moe_config['expert_dropout'] > 1:
            raise ValueError("expert_dropout must be between 0 and 1")
            
        if moe_config['knowledge_distillation_weight'] < 0:
            raise ValueError("knowledge_distillation_weight must be non-negative")
            
        # Validate training stages configuration
        stages_config = moe_config['training_stages']
        required_stage_fields = ['global_expert_epochs', 'regional_expert_epochs']
        
        for field in required_stage_fields:
            if field not in stages_config:
                raise ValueError(f"Missing required field '{field}' in training stages configuration")
            if not isinstance(stages_config[field], int) or stages_config[field] < 0:
                raise ValueError(f"{field} must be a non-negative integer")

def validate_training_config(training_config: Dict[str, Any]) -> None:
    """Validate training configuration."""
    required_sections = ['optimizer', 'scheduler', 'training_config']
    
    for section in required_sections:
        if section not in training_config:
            raise ValueError(f"Missing required section '{section}' in training configuration")
    
    # Validate optimizer config
    optimizer_config = training_config['optimizer']
    required_optimizer_fields = ['name', 'learning_rate', 'weight_decay', 'beta1', 'beta2']
    
    for field in required_optimizer_fields:
        if field not in optimizer_config:
            raise ValueError(f"Missing required field '{field}' in optimizer configuration")
            
    if optimizer_config['name'] not in ['adam', 'adamw']:
        raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")
    
    # Validate scheduler config
    scheduler_config = training_config['scheduler']
    if scheduler_config['enabled']:
        required_scheduler_fields = ['type', 'warmup_steps', 'min_lr']
        
        for field in required_scheduler_fields:
            if field not in scheduler_config:
                raise ValueError(f"Missing required field '{field}' in scheduler configuration")
                
        if scheduler_config['type'] not in ['cosine']:
            raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}")
    
    # Validate training config
    train_config = training_config['training_config']
    required_train_fields = [
        'batch_size',
        'num_epochs',
        'train_split',
        'validation_split',
        'test_split',
        'early_stopping',
        'gradient_clipping',
        'mixed_precision'
    ]
    
    for field in required_train_fields:
        if field not in train_config:
            raise ValueError(f"Missing required field '{field}' in training configuration")
    
    splits_sum = train_config['train_split'] + train_config['validation_split'] + train_config['test_split']
    if not 0.99 <= splits_sum <= 1.01:  # Allow for small floating point errors
        raise ValueError(f"Data splits must sum to 1, got {splits_sum}")

def validate_paths_config(paths_config: Dict[str, str]) -> None:
    """Validate paths configuration."""
    required_paths = ['model_save_dir', 'best_model_path', 'log_dir']
    
    for path_name in required_paths:
        if path_name not in paths_config:
            raise ValueError(f"Missing required path '{path_name}' in paths configuration")

def validate_logging_config(logging_config: Dict[str, Any]) -> None:
    """Validate logging configuration."""
    required_sections = ['wandb', 'tensorboard', 'checkpointing']
    
    for section in required_sections:
        if section not in logging_config:
            raise ValueError(f"Missing required section '{section}' in logging configuration")
    
    # Validate wandb config
    wandb_config = logging_config['wandb']
    if wandb_config['enabled']:
        required_wandb_fields = ['project', 'entity']
        for field in required_wandb_fields:
            if field not in wandb_config:
                raise ValueError(f"Missing required field '{field}' in wandb configuration")
    
    # Validate tensorboard config
    tensorboard_config = logging_config['tensorboard']
    if tensorboard_config['enabled']:
        if 'log_dir' not in tensorboard_config:
            raise ValueError("Missing 'log_dir' in tensorboard configuration")
    
    # Validate checkpointing config
    checkpoint_config = logging_config['checkpointing']
    required_checkpoint_fields = ['save_best', 'save_frequency', 'max_checkpoints']
    
    for field in required_checkpoint_fields:
        if field not in checkpoint_config:
            raise ValueError(f"Missing required field '{field}' in checkpointing configuration")

def validate_data_config(data_config: Dict[str, Any]) -> None:
    """Validate data configuration."""
    required_sections = ['input', 'augmentation', 'sampling']
    
    for section in required_sections:
        if section not in data_config:
            raise ValueError(f"Missing required section '{section}' in data configuration")
    
    # Validate input config
    input_config = data_config['input']
    required_input_fields = ['max_sequence_length', 'padding_value', 'truncation']
    
    for field in required_input_fields:
        if field not in input_config:
            raise ValueError(f"Missing required field '{field}' in input configuration")
            
    if input_config['truncation'] not in ['left', 'right']:
        raise ValueError(f"Invalid truncation type: {input_config['truncation']}")
    
    # Validate augmentation config
    augmentation_config = data_config['augmentation']
    required_augmentation_fields = ['enabled', 'sequence_masking_prob', 'feature_dropout']
    
    for field in required_augmentation_fields:
        if field not in augmentation_config:
            raise ValueError(f"Missing required field '{field}' in augmentation configuration")
    
    # Validate sampling config
    sampling_config = data_config['sampling']
    required_sampling_fields = ['negative_sampling_ratio', 'max_samples_per_user']
    
    for field in required_sampling_fields:
        if field not in sampling_config:
            raise ValueError(f"Missing required field '{field}' in sampling configuration")

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration."""
    required_sections = ['features', 'model', 'tasks', 'training', 'logging', 'data', 'paths', 'device']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in configuration")
    
    # Validate each section
    validate_feature_config(config['features'])
    validate_task_config(config['tasks'])
    validate_model_config(config['model'])
    validate_training_config(config['training'])
    validate_paths_config(config['paths'])
    validate_logging_config(config['logging'])
    validate_data_config(config['data'])
    
    if config['device'] not in ['cuda', 'cpu']:
        raise ValueError("Device must be either 'cuda' or 'cpu'")

def create_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories from configuration."""
    # Create paths directories
    paths = config['paths']
    for path in paths.values():
        directory = os.path.dirname(path)
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Create logging directories
    if config['logging']['tensorboard']['enabled']:
        Path(config['logging']['tensorboard']['log_dir']).mkdir(parents=True, exist_ok=True)

def load_config(config_path: str = 'config.yml') -> Dict[str, Any]:
    """Load and validate configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing validated configuration settings
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate configuration
    validate_config(config)
    
    # Create necessary directories
    create_directories(config)
    
    return config 