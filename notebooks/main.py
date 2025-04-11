# Databricks notebook source
# Databricks notebook source

import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.preprocessing import LabelEncoder

from commons.data.dataset import ReRankerDataset
from commons.data.collate import collate_fn
from commons.models import DCNv2
from commons.training import ReRankerTrainer
from commons.training.tuner import HyperparameterTuner
from commons.utils.config import load_config
from commons.training.utils import setup_optimizer_and_scheduler
from commons.data.loader import load_and_process_data

# COMMAND ----------

def setup_optimizer_and_scheduler(model, config):
    """Setup optimizer and learning rate scheduler."""
    optimizer_config = config['training']['optimizer']
    scheduler_config = config['training']['scheduler']
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=optimizer_config['learning_rate'],
        weight_decay=optimizer_config['weight_decay'],
        betas=(optimizer_config['beta1'], optimizer_config['beta2'])
    )
    
    # Setup scheduler
    if scheduler_config['enabled']:
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
        scheduler = None
    
    return optimizer, scheduler

# COMMAND ----------

def load_and_process_data(config):
    """Load and process the data."""
    features_df = pd.read_csv('data/test/features.csv')
    labels_df = pd.read_csv('data/test/labels.csv')
    embeddings_dict = np.load('data/test/embeddings.npy', allow_pickle=True).item()
    
    # Process features
    features = {}
    for feature_name in config['features']['enabled_features']:
        feature_config = config['features']['feature_configs'][feature_name]
        if not feature_config['enabled']:
            continue
            
        if feature_name in embeddings_dict:
            features[feature_name] = embeddings_dict[feature_name]
        elif feature_name.endswith('_embeddings') and feature_name in embeddings_dict:
            base_name = feature_name.replace('_embeddings', '')
            features[base_name] = embeddings_dict[feature_name]
        elif feature_name in features_df.columns:
            if feature_config['type'] == 'categorical':
                encoder = LabelEncoder()
                features[feature_name] = encoder.fit_transform(features_df[feature_name])
                feature_config['num_categories'] = len(encoder.classes_)
            else:
                features[feature_name] = features_df[feature_name].values
        elif feature_name + '_0' in features_df.columns:
            feature_col = feature_name + '_0'
            if feature_config['type'] == 'categorical':
                encoder = LabelEncoder()
                features[feature_name] = encoder.fit_transform(features_df[feature_col])
                feature_config['num_categories'] = len(encoder.classes_)
            else:
                features[feature_name] = features_df[feature_col].values
    
    # Prepare labels
    labels = {
        'click': labels_df['click'].values.reshape(-1, 1),
        'purchase': labels_df['purchase'].values.reshape(-1, 1),
        'add_to_cart': labels_df['add_to_cart'].values.reshape(-1, 1)
    }
    
    return features, labels

# COMMAND ----------

def train_model(config_path: str, is_tuning: bool = False):
    """Train a model using the specified configuration."""
    # Load configuration
    config = load_config(config_path)
    
    # Load and process data
    features, labels = load_and_process_data(config)
    
    # Create dataset
    dataset = ReRankerDataset(features, labels, config)
    
    # Create train/val/test split
    train_size = int(len(dataset) * config['training']['training_config']['train_split'])
    val_size = int(len(dataset) * config['training']['training_config']['validation_split'])
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['training_config']['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, config)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['training_config']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, config)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['training_config']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, config)
    )
    
    if is_tuning:
        # Run hyperparameter tuning
        print(f"\nStarting hyperparameter tuning for {config_path}...")
        tuner = HyperparameterTuner(
            base_config=config,
            dataset=dataset,
            tuning_config=config['hyperparameter_tuning']
        )
        best_trial = tuner.tune()
        print(f"\nBest trial saved to {config['hyperparameter_tuning']['checkpointing']['checkpoint_dir']}/best_trial.json")
        
        # Use best trial config for final training
        config = best_trial['config']
    
    # Initialize model
    model = DCNv2(config)
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, config)
    
    # Initialize trainer
    trainer = ReRankerTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config
    )
    
    # Train model
    print(f"\nTraining model with {config_path}...")
    trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    test_metrics = trainer.evaluate(test_loader)
    print("\nTest metrics:")
    for task, metrics in test_metrics.items():
        if task == 'total_loss':
            print(f"\nTotal Loss: {metrics:.4f}")
        else:
            print(f"\n{task.upper()}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
    
    # Save final model
    model_name = Path(config_path).stem
    final_model_path = Path(config['paths']['model_save_dir']) / f'final_model_{model_name}.pt'
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'test_metrics': test_metrics
    }, final_model_path)
    print(f"\nFinal model saved to {final_model_path}")

# COMMAND ----------

def main():
    """Run training for all configurations."""
    config_files = [
        'configs/base_config.yml',
        'configs/moe_config.yml',
        'configs/multihead_sequence_config.yml',
        'configs/tuning_config.yml'
    ]
    
    for config_file in config_files:
        print(f"\n{'='*80}")
        print(f"Processing {config_file}")
        print(f"{'='*80}")
        
        # Run training with tuning for tuning config
        is_tuning = 'tuning' in config_file
        train_model(config_file, is_tuning)

# COMMAND ----------

if __name__ == "__main__":
    main() 