import torch
from torch.utils.data import Dataset, DataLoader
from models.dcn import DCNv2
from utils.trainer import ReRankerTrainer
from utils.config import load_config
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Tuple
from transformers import AutoTokenizer
import argparse
import os
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim.lr_scheduler import SequentialLR
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

def load_data(config: Dict[str, Any], data_path: str) -> Tuple[Dict[str, Union[np.ndarray, List[str], List[List[int]]]], Dict[str, np.ndarray]]:
    """Load and preprocess data from CSV files using pandas."""
    # Load the main DataFrame
    df = pd.read_csv(data_path)
    
    # Initialize feature dictionary
    features = {}
    feature_config = config['features']['feature_configs']
    enabled_features = config['features']['enabled_features']
    
    # Initialize preprocessors
    label_encoders = {}
    scalers = {}
    
    # Process each feature based on its type
    for name in enabled_features:
        if not feature_config[name]['enabled']:
            continue
            
        feature_type = feature_config[name]['type']
        
        if feature_type == 'pretrained_embedding':
            # Load pre-computed embeddings from numpy file
            embed_path = os.path.join(os.path.dirname(data_path), f"{name}_embeddings.npy")
            if os.path.exists(embed_path):
                features[name] = np.load(embed_path)
            else:
                print(f"Warning: Embedding file not found for {name}")
                continue
                
        elif feature_type == 'text':
            # Keep text as is for later processing
            if name in df.columns:
                features[name] = df[name].fillna('').tolist()
                
        elif feature_type == 'sequence':
            # Convert sequence string to list of integers
            if name in df.columns:
                features[name] = df[name].apply(lambda x: 
                    [int(i) for i in str(x).split(',')] if pd.notna(x) else []
                ).tolist()
                
        elif feature_type == 'categorical':
            # Use label encoding for categorical features
            if name in df.columns:
                label_encoders[name] = LabelEncoder()
                features[name] = label_encoders[name].fit_transform(df[name].fillna('UNKNOWN'))
                
        elif feature_type == 'numeric':
            # Standardize numeric features
            if name in df.columns:
                scalers[name] = StandardScaler()
                features[name] = scalers[name].fit_transform(df[name].fillna(0).values.reshape(-1, 1))
    
    # Process labels
    labels = {}
    for task_name, task_config in config['tasks'].items():
        if task_config['enabled'] and f"{task_name}_label" in df.columns:
            labels[task_name] = df[f"{task_name}_label"].values.reshape(-1, 1)
    
    return features, labels

def generate_dummy_data(config: Dict[str, Any], num_samples: int) -> Tuple[Dict[str, Union[np.ndarray, List[str], List[List[int]]]], Dict[str, np.ndarray]]:
    """Generate dummy data based on feature configuration."""
    feature_config = config['features']['feature_configs']
    enabled_features = config['features']['enabled_features']
    features = {}
    
    # Generate dummy data for each feature type
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
            
        elif config['type'] == 'categorical':
            num_categories = config.get('num_categories', 10)
            features[name] = np.random.randint(0, num_categories, size=(num_samples, 1))
            
        elif config['type'] == 'numeric':
            # Generate random prices between 1 and 1000 with some skew
            if name == 'price':
                features[name] = np.exp(np.random.normal(4, 1, size=(num_samples, 1)))
            else:
                features[name] = np.random.randn(num_samples, config['dim'])
    
    # Generate labels for each task
    labels = {}
    for task_name, task_config in config['tasks'].items():
        if task_config['enabled']:
            # Generate binary labels with some correlation to features
            base_prob = 0.3 + 0.4 * (features.get('price', np.random.randn(num_samples, 1)) > 0)
            labels[task_name] = (np.random.random(size=(num_samples, 1)) < base_prob).astype(np.float32)
    
    return features, labels

def save_test_data(features: Dict[str, Union[np.ndarray, List[str], List[List[int]]]], 
                  labels: Dict[str, np.ndarray],
                  config: Dict[str, Any],
                  output_dir: str):
    """Save generated test data to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame to store non-embedding features
    df_data = {}
    
    for name, feature in features.items():
        feature_config = config['features']['feature_configs'][name]
        
        if feature_config['type'] == 'pretrained_embedding':
            # Save embeddings as numpy files
            np.save(os.path.join(output_dir, f"{name}_embeddings.npy"), feature)
        elif feature_config['type'] == 'sequence':
            # Convert sequences to comma-separated strings
            df_data[name] = [','.join(map(str, seq)) for seq in feature]
        else:
            # Add other features directly to DataFrame
            df_data[name] = feature
    
    # Add labels to DataFrame
    for task_name, label in labels.items():
        df_data[f"{task_name}_label"] = label
    
    # Save DataFrame to CSV
    pd.DataFrame(df_data).to_csv(os.path.join(output_dir, "features.csv"), index=False)

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
        features, labels = generate_dummy_data(config, args.num_samples)
        
        # Save dummy data for testing
        if args.save_dummy_data:
            print("Saving dummy data...")
            save_test_data(features, labels, config, "data/test")
            
    else:
        print("Loading data from:", args.data_path)
        features, labels = load_data(config, args.data_path)
    
    # Create dataset and dataloader
    dataset = ReRankerDataset(features, labels, config)
    
    # Split data into train/val/test
    train_size = int(len(dataset) * config['training']['training_config']['train_split'])
    val_size = int(len(dataset) * config['training']['training_config']['validation_split'])
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['training_config']['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, config)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['training_config']['batch_size'],
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, config)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['training_config']['batch_size'],
        shuffle=False,
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
    trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader)
    print("\nTest Set Metrics:")
    for task_name, metrics in test_metrics.items():
        print(f"\n{task_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--data_path", type=str, help="Path to input data CSV file")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples for dummy data")
    parser.add_argument("--use_dummy_data", action="store_true", help="Use dummy data for training")
    parser.add_argument("--save_dummy_data", action="store_true", help="Save generated dummy data")
    args = parser.parse_args()
    main(args) 