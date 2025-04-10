import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional
from torch.serialization import safe_globals, add_safe_globals
from sklearn.preprocessing import LabelEncoder

from ..models import DCNv2
from ..data.dataset import ReRankerDataset
from ..data.collate import collate_fn
from torch.utils.data import DataLoader

# Add numpy scalar to safe globals
add_safe_globals(['numpy.core.multiarray.scalar'])

class Predictor:
    def __init__(self, model_path: str):
        """Initialize predictor with a trained model.
        
        Args:
            model_path: Path to the saved model checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.config = checkpoint['config']
        state_dict = checkpoint['model_state_dict']
        
        # Initialize model with configuration
        self.model = DCNv2(self.config)
        
        # Initialize networks first
        if self.config['model']['architecture'] == 'dcn_v2_moe':
            # Get input dimension from checkpoint
            input_dim = state_dict['deep_network.layers.0.weight'].size(1)
            
            # Initialize networks with the same dimension as checkpoint
            self.model._initialize_networks(input_dim)
            
            # Set MoE training stage
            self.model.set_moe_training_stage('regional')
        else:
            # Initialize networks with input dimension from checkpoint
            input_dim = state_dict['deep_network.layers.0.weight'].size(1)
            self.model._initialize_networks(input_dim)
        
        # Initialize cross network parameters
        for i in range(len(self.model.cross_net)):
            self.model.cross_net[i].initialize_parameters(input_dim)
        
        # Clean up state dict keys if needed
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            cleaned_state_dict[k] = v
        
        # Debug: Print state dict keys
        print("Model state dict keys:")
        for k in self.model.state_dict().keys():
            print(f"  {k}")
        print("\nCheckpoint state dict keys:")
        for k in cleaned_state_dict.keys():
            print(f"  {k}")
        
        # Load state dict
        self.model.load_state_dict(cleaned_state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize label encoders for categorical features
        self.label_encoders = {}
    
    def prepare_data(self, features_df: pd.DataFrame, embeddings_dict: Dict) -> ReRankerDataset:
        """Prepare data for inference.
        
        Args:
            features_df: DataFrame containing features
            embeddings_dict: Dictionary containing embeddings
            
        Returns:
            ReRankerDataset: Dataset ready for inference
        """
        features = {}
        for feature_name in self.config['features']['enabled_features']:
            feature_config = self.config['features']['feature_configs'][feature_name]
            if not feature_config['enabled']:
                continue
                
            if feature_name in embeddings_dict:
                features[feature_name] = embeddings_dict[feature_name]
            elif feature_name.endswith('_embeddings') and feature_name in embeddings_dict:
                base_name = feature_name.replace('_embeddings', '')
                features[base_name] = embeddings_dict[feature_name]
            elif feature_name in features_df.columns:
                if feature_config['type'] == 'categorical':
                    # Create and fit label encoder if not exists
                    if feature_name not in self.label_encoders:
                        self.label_encoders[feature_name] = LabelEncoder()
                        self.label_encoders[feature_name].fit(features_df[feature_name])
                    # Transform categorical values
                    features[feature_name] = self.label_encoders[feature_name].transform(features_df[feature_name])
                else:
                    features[feature_name] = features_df[feature_name].values
            elif feature_name + '_0' in features_df.columns:
                feature_col = feature_name + '_0'
                if feature_config['type'] == 'categorical':
                    # Create and fit label encoder if not exists
                    if feature_name not in self.label_encoders:
                        self.label_encoders[feature_name] = LabelEncoder()
                        self.label_encoders[feature_name].fit(features_df[feature_col])
                    # Transform categorical values
                    features[feature_name] = self.label_encoders[feature_name].transform(features_df[feature_col])
                else:
                    features[feature_name] = features_df[feature_col].values
        
        # Create dummy labels for inference
        dummy_labels = {
            'click': np.zeros((len(features_df), 1)),
            'purchase': np.zeros((len(features_df), 1)),
            'add_to_cart': np.zeros((len(features_df), 1))
        }
        
        return ReRankerDataset(features, dummy_labels, self.config)
    
    def predict(self, 
                features_df: pd.DataFrame, 
                embeddings_dict: Dict,
                batch_size: int = 32,
                tasks: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Generate predictions for the input data.
        
        Args:
            features_df: DataFrame containing features
            embeddings_dict: Dictionary containing embeddings
            batch_size: Batch size for inference
            tasks: List of tasks to predict. If None, predicts all tasks.
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing predictions for each task
        """
        dataset = self.prepare_data(features_df, embeddings_dict)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: collate_fn(batch, self.config)
        )
        
        if tasks is None:
            tasks = ['click', 'purchase', 'add_to_cart']
        
        predictions = {task: [] for task in tasks}
        
        with torch.no_grad():
            for batch_features, _ in dataloader:
                # Move batch to device
                batch_features = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch_features.items()}
                
                # Get model predictions
                outputs = self.model(batch_features)
                
                # Store predictions for each task
                for task in tasks:
                    task_preds = torch.sigmoid(outputs[task]).cpu().numpy()
                    predictions[task].append(task_preds)
        
        # Concatenate predictions
        predictions = {task: np.concatenate(preds) for task, preds in predictions.items()}
        
        return predictions 