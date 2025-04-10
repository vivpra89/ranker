import torch
import numpy as np
from typing import Dict, List, Any, Optional
from ..models.dcn import DCNv2
from pathlib import Path

class ReRankerInference:
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize inference class.
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get config
        self.config = checkpoint['config']
        
        # Initialize model
        self.model = DCNv2(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def preprocess_features(self, features: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Preprocess features for model input.
        
        Args:
            features: Dictionary of raw features
            
        Returns:
            Dictionary of preprocessed features as tensors
        """
        processed = {}
        feature_config = self.config['features']['feature_configs']
        
        for name, value in features.items():
            if name not in feature_config or not feature_config[name]['enabled']:
                continue
                
            feature_type = feature_config[name]['type']
            
            if feature_type == 'numeric':
                # Convert numeric features to tensor
                if isinstance(value, (int, float)):
                    value = np.array([value])
                processed[name] = torch.FloatTensor(value).to(self.device)
            elif feature_type == 'categorical':
                # Convert categorical features to tensor
                if isinstance(value, (int, str)):
                    value = np.array([value])
                processed[name] = torch.LongTensor(value).to(self.device)
            elif feature_type == 'sequence':
                # Convert sequence features to tensor
                if isinstance(value, list):
                    value = np.array(value)
                processed[name] = torch.FloatTensor(value).to(self.device)
            elif feature_type in ['pretrained_embedding', 'binary_embedding']:
                # Convert embedding features to tensor
                if isinstance(value, list):
                    value = np.array(value)
                processed[name] = torch.FloatTensor(value).to(self.device)
            elif feature_type == 'text':
                # Text features will be handled by the model's text embedder
                processed[name] = value
        
        return processed
    
    def predict(self,
               user_features: Dict[str, Any],
               product_features: List[Dict[str, Any]],
               batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Predict scores for a list of products based on user features.
        
        Args:
            user_features: Dictionary of user features
            product_features: List of product feature dictionaries
            batch_size: Batch size for inference
            
        Returns:
            Dictionary of predicted scores for each task
        """
        # Preprocess user features
        user_features = self.preprocess_features(user_features)
        
        # Initialize predictions
        all_predictions = {
            task: [] for task in self.config['tasks']
            if self.config['tasks'][task]['enabled']
        }
        
        # Process products in batches
        for i in range(0, len(product_features), batch_size):
            batch_products = product_features[i:i + batch_size]
            
            # Preprocess batch features
            batch_features = {}
            
            # Add user features (repeated for each product)
            for name, value in user_features.items():
                if isinstance(value, torch.Tensor):
                    batch_features[name] = value.repeat(len(batch_products), 1)
                else:
                    batch_features[name] = value
            
            # Add product features
            for name in self.config['features']['enabled_features']:
                if name in batch_products[0] and name not in batch_features:
                    product_values = [p[name] for p in batch_products]
                    batch_features[name] = self.preprocess_features({name: product_values})[name]
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(batch_features)
                
                # Collect predictions
                for task, pred in outputs.items():
                    all_predictions[task].append(pred.cpu().numpy())
        
        # Concatenate predictions
        return {
            task: np.concatenate(preds)
            for task, preds in all_predictions.items()
        }
    
    def rank_products(self,
                     user_features: Dict[str, Any],
                     product_features: List[Dict[str, Any]],
                     task: str = None,
                     batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Rank products based on predicted scores.
        
        Args:
            user_features: Dictionary of user features
            product_features: List of product feature dictionaries
            task: Task to use for ranking (if None, uses first enabled task)
            batch_size: Batch size for inference
            
        Returns:
            List of products sorted by predicted score
        """
        # Get predictions
        predictions = self.predict(user_features, product_features, batch_size)
        
        # Get task to use for ranking
        if task is None:
            task = next(t for t in self.config['tasks'] if self.config['tasks'][t]['enabled'])
        elif task not in predictions:
            raise ValueError(f"Task '{task}' not found in predictions")
        
        # Sort products by score
        scores = predictions[task].flatten()
        sorted_indices = np.argsort(scores)[::-1]
        
        # Return sorted products with scores
        return [
            {**product_features[idx], 'score': float(scores[idx])}
            for idx in sorted_indices
        ] 