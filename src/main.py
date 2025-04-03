import torch
from torch.utils.data import Dataset, DataLoader
from models.dcn import DCNv2
from utils.trainer import ReRankerTrainer
from utils.config import load_config
import numpy as np
from typing import Dict, Any, List, Union, Tuple
from transformers import AutoTokenizer
import argparse
import os
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim.lr_scheduler import SequentialLR

def pad_sequences(sequences: List[List[int]], max_length: int, padding_value: int = 0, truncation: str = "right") -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad sequences to max_length and create padding mask."""
    padded_sequences = []
    padding_mask = []
    
    for seq in sequences:
        # Truncate if longer than max_length
        if truncation == "right":
            seq = seq[:max_length]
        else:  # truncation == "left"
            seq = seq[-max_length:] if len(seq) > max_length else seq
            
        # Calculate padding
        pad_length = max_length - len(seq)
        # Pad sequence
        padded_seq = seq + [padding_value] * pad_length
        # Create mask (False for real tokens, True for padding)
        mask = [False] * len(seq) + [True] * pad_length
        
        padded_sequences.append(padded_seq)
        padding_mask.append(mask)
    
    return (
        torch.tensor(padded_sequences),
        torch.tensor(padding_mask)
    )

class ReRankerDataset(Dataset):
    def __init__(self, 
                 features: Dict[str, Union[np.ndarray, List[str], List[List[int]]]],
                 labels: Dict[str, np.ndarray],
                 config: Dict[str, Any]):
        """
        Args:
            features: Dictionary of features (arrays, text lists, or sequence lists)
            labels: Dictionary of label arrays for each task
            config: Configuration dictionary
        """
        self.feature_config = config['features']['feature_configs']
        self.enabled_features = config['features']['enabled_features']
        self.data_config = config['data']
        
        # Process features based on their type
        self.features = {}
        for name in self.enabled_features:
            if name not in features:
                continue
                
            feature = features[name]
            feature_config = self.feature_config[name]
            
            if not feature_config['enabled']:
                continue
                
            if feature_config['type'] == 'pretrained_embedding':
                self.features[name] = torch.FloatTensor(feature)
            elif feature_config['type'] in ['text', 'sequence']:
                self.features[name] = feature
            else:
                self.features[name] = torch.FloatTensor(feature)
        
        # Process labels
        self.labels = {}
        for name, label in labels.items():
            self.labels[name] = torch.FloatTensor(label)
        
    def __len__(self):
        return len(next(iter(self.labels.values())))
    
    def __getitem__(self, idx):
        # Return features based on their type
        batch_features = {}
        for name, feature in self.features.items():
            if self.feature_config[name]['type'] in ['text', 'sequence']:
                batch_features[name] = feature[idx]
            else:
                batch_features[name] = feature[idx]
        
        return (
            batch_features,
            {name: label[idx] for name, label in self.labels.items()}
        )

def collate_fn(batch, config):
    """Custom collate function to handle different feature types."""
    feature_config = config['features']['feature_configs']
    enabled_features = config['features']['enabled_features']
    data_config = config['data']['input']
    
    features = {name: [] for name in enabled_features if feature_config[name]['enabled']}
    labels = {name: [] for name in batch[0][1]}
    
    # Collect features and labels from batch
    for sample_features, sample_labels in batch:
        for name, value in sample_features.items():
            features[name].append(value)
        for name, value in sample_labels.items():
            labels[name].append(value)
    
    # Process features based on their type
    processed_features = {}
    for name, values in features.items():
        config = feature_config[name]
        if not config['enabled']:
            continue
            
        if config['type'] == 'sequence':
            # Pad sequences and create padding mask
            sequences, padding_mask = pad_sequences(
                values,
                max_length=config.get('max_seq_length', data_config['max_sequence_length']),
                padding_value=data_config['padding_value'],
                truncation=data_config['truncation']
            )
            processed_features[name] = (sequences, padding_mask)
        elif config['type'] == 'text':
            # Keep text as list
            processed_features[name] = values
        else:
            # Stack tensors
            processed_features[name] = torch.stack(values)
    
    # Stack labels
    processed_labels = {
        name: torch.stack(values)
        for name, values in labels.items()
    }
    
    return processed_features, processed_labels

def generate_dummy_data(config: Dict[str, Any], num_samples: int) -> Dict[str, Union[np.ndarray, List[str], List[List[int]]]]:
    """Generate dummy data based on feature configuration."""
    feature_config = config['features']['feature_configs']
    enabled_features = config['features']['enabled_features']
    features = {}
    
    for name in enabled_features:
        config = feature_config[name]
        if not config['enabled']:
            continue
            
        if config['type'] == 'pretrained_embedding':
            features[name] = np.random.randn(num_samples, config['dim'])
        elif config['type'] == 'text':
            vocab = ['city', 'town', 'village', 'country', 'state', 'region', 'area', 'zone']
            features[name] = [
                f"{np.random.choice(vocab)} {np.random.choice(vocab)}"
                for _ in range(num_samples)
            ]
        elif config['type'] == 'sequence':
            max_seq_length = config.get('max_seq_length', 50)
            num_products = config.get('num_products', 1000)
            features[name] = [
                [np.random.randint(0, num_products) for _ in range(np.random.randint(1, max_seq_length))]
                for _ in range(num_samples)
            ]
        else:
            features[name] = np.random.randn(num_samples, config['dim'])
    
    return features

def setup_optimizer_and_scheduler(model, config):
    """Setup optimizer and learning rate scheduler."""
    optimizer_config = config['training']['optimizer']
    scheduler_config = config['training']['scheduler']
    
    # Setup optimizer
    optimizer_name = optimizer_config['name'].lower()
    if optimizer_name == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr=optimizer_config['learning_rate'],
            weight_decay=optimizer_config['weight_decay'],
            betas=(optimizer_config['beta1'], optimizer_config['beta2'])
        )
    elif optimizer_name == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=optimizer_config['learning_rate'],
            weight_decay=optimizer_config['weight_decay'],
            betas=(optimizer_config['beta1'], optimizer_config['beta2'])
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Setup scheduler
    if scheduler_config['enabled']:
        scheduler_type = scheduler_config['type'].lower()
        if scheduler_type == 'cosine':
            # Warmup scheduler
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=scheduler_config['warmup_steps']
            )
            
            # Main scheduler
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config['training']['training_config']['num_epochs'] - scheduler_config['warmup_steps'],
                eta_min=scheduler_config['min_lr']
            )
            
            # Combine schedulers
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[scheduler_config['warmup_steps']]
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    else:
        scheduler = None
    
    return optimizer, scheduler

def main(args):
    # Load and validate configuration
    config = load_config(args.config)
    print("Loaded configuration:")
    print("Features:", config.get('features', {}))
    print("Tasks:", config.get('tasks', {}))
    print("Model:", config.get('model', {}))
    
    # Set device
    device = torch.device(config['device'])
    
    # Generate or load data
    if args.use_dummy_data:
        print("Using dummy data for training...")
        features = generate_dummy_data(config, args.num_samples)
        labels = {}
        for task_name, task_config in config['tasks'].items():
            if task_config['enabled']:
                labels[task_name] = np.random.randint(0, 2, size=(args.num_samples, 1)).astype(np.float32)
                print(f"Generated {args.num_samples} labels for task: {task_name}")
    else:
        # TODO: Implement real data loading
        raise NotImplementedError("Real data loading not implemented yet")
    
    # Create dataset and dataloader
    dataset = ReRankerDataset(features, labels, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['training_config']['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, config)
    )
    
    # Initialize model
    model = DCNv2(config).to(device)
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, config)
    
    # Initialize trainer
    trainer = ReRankerTrainer(
        model=model,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    
    # Train model
    trainer.train(dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples for dummy data")
    parser.add_argument("--use_dummy_data", action="store_true", help="Use dummy data for training")
    args = parser.parse_args()
    main(args) 