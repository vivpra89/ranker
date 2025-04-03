import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from models.dcn import DCNv2
from utils.config import load_config
import heapq
from transformers import AutoTokenizer
import logging
import argparse
from pathlib import Path
import torch.serialization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add safe globals for numpy scalar types
torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])

class ReRankerInference:
    def __init__(self, model_path: str, config_path: str, device: str = None):
        """
        Initialize the reranker inference.
        
        Args:
            model_path: Path to the trained model checkpoint
            config_path: Path to the configuration file
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.config = load_config(config_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = DCNv2(self.config).to(self.device)
        
        # Load model weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            logger.warning(f"Failed to load with full checkpoint, trying weights only: {e}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # Initialize tokenizers for text features
        self.tokenizers = {}
        for name, feature_config in self.config['features']['feature_configs'].items():
            if feature_config['enabled'] and feature_config['type'] == 'text':
                self.tokenizers[name] = AutoTokenizer.from_pretrained(feature_config['model_name'])
    
    def preprocess_features(
        self,
        features: Dict[str, Any],
        feature_config: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess features for inference.
        
        Args:
            features: Dictionary of raw features
            feature_config: Feature configuration
            
        Returns:
            Dictionary of preprocessed features
        """
        processed = {}
        for name, value in features.items():
            if not feature_config[name]['enabled']:
                continue
                
            feature_type = feature_config[name]['type']
            if feature_type == 'pretrained_embedding':
                if isinstance(value, np.ndarray):
                    processed[name] = torch.FloatTensor(value)
                elif isinstance(value, list):
                    processed[name] = torch.FloatTensor(np.array(value))
                else:
                    continue
            elif feature_type == 'text':
                # Ensure text input is a list of strings
                if isinstance(value, str):
                    text_input = [value]
                elif isinstance(value, list) and all(isinstance(x, str) for x in value):
                    text_input = value
                else:
                    continue
                # Tokenize text
                inputs = self.tokenizers[name](
                    text_input,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                # Get embeddings from the model's text embedder
                with torch.no_grad():
                    embeddings = self.model.text_embedders[name](text_input)
                processed[name] = embeddings
            elif feature_type == 'sequence':
                # Convert sequence to tensor and create mask
                if isinstance(value, np.ndarray):
                    seq = torch.LongTensor(value)
                elif isinstance(value, list):
                    seq = torch.LongTensor(np.array(value))
                else:
                    continue
                mask = torch.zeros_like(seq, dtype=torch.bool)
                processed[name] = (seq, mask)
            else:
                # Numeric or categorical features
                if isinstance(value, np.ndarray):
                    processed[name] = torch.FloatTensor(value)
                elif isinstance(value, list):
                    processed[name] = torch.FloatTensor(np.array(value))
                else:
                    continue
        
        return processed
    
    def predict_scores(
        self,
        user_features: Dict[str, Any],
        product_features: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """
        Predict scores for a list of products for a given user.
        
        Args:
            user_features: Dictionary of user features
            product_features: List of product feature dictionaries
            
        Returns:
            Dictionary of predicted scores for each task
        """
        feature_config = self.config['features']['feature_configs']
        batch_size = len(product_features)
        
        # Prepare batch features
        batch_features = {}
        
        # Convert numpy array format to actual numpy arrays
        def to_numpy(value):
            if isinstance(value, dict) and value.get('__type__') == 'ndarray':
                return np.array(value['data']).reshape(value['shape'])
            return value
        
        # Add user features (repeated for each product)
        for name, value in user_features.items():
            if feature_config[name]['enabled']:
                value = to_numpy(value)
                if isinstance(value, np.ndarray):
                    batch_features[name] = np.repeat(value[np.newaxis, :], batch_size, axis=0)
                elif feature_config[name]['type'] == 'text':
                    batch_features[name] = [value] * batch_size
                else:
                    batch_features[name] = [value] * batch_size
        
        # Add product features
        for name in self.config['features']['enabled_features']:
            if name in product_features[0] and feature_config[name]['enabled']:
                values = [to_numpy(p[name]) for p in product_features]
                if isinstance(values[0], np.ndarray):
                    batch_features[name] = np.stack(values)
                elif feature_config[name]['type'] == 'text':
                    batch_features[name] = values
                else:
                    batch_features[name] = values
        
        # Preprocess features
        processed_features = self.preprocess_features(batch_features, feature_config)
        
        # Move features to device
        processed_features = {
            name: value.to(self.device) if isinstance(value, torch.Tensor)
            else (tuple(v.to(self.device) for v in value) if isinstance(value, tuple)
                  else value)
            for name, value in processed_features.items()
        }
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(processed_features)
        
        # Convert predictions to numpy
        return {
            task: preds.cpu().numpy().squeeze()
            for task, preds in predictions.items()
        }
    
    def rerank_products(
        self,
        user_features: Dict[str, Any],
        product_features: List[Dict[str, Any]],
        task_weights: Dict[str, float] = None,
        top_k: int = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank products for a given user.
        
        Args:
            user_features: Dictionary of user features
            product_features: List of product feature dictionaries
            task_weights: Optional dictionary of task weights (defaults to config weights)
            top_k: Optional number of top products to return
            
        Returns:
            List of (product_idx, score) tuples sorted by score
        """
        # Get predictions for all tasks
        task_predictions = self.predict_scores(user_features, product_features)
        
        # Use config weights if not provided
        if task_weights is None:
            task_weights = {
                task: spec['weight']
                for task, spec in self.config['tasks'].items()
                if spec['enabled']
            }
        
        # Compute weighted sum of task predictions
        total_weights = sum(task_weights.values())
        combined_scores = np.zeros(len(product_features))
        
        for task, weight in task_weights.items():
            if task in task_predictions:
                combined_scores += (weight / total_weights) * task_predictions[task]
        
        # Sort products by score
        ranked_products = list(enumerate(combined_scores))
        ranked_products.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k if specified
        if top_k is not None:
            ranked_products = ranked_products[:top_k]
        
        return ranked_products

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--config_path", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on")
    args = parser.parse_args()
    
    # Initialize reranker
    reranker = ReRankerInference(args.model_path, args.config_path, args.device)
    
    # Example usage with dummy data
    user_features = {
        "user": np.random.randn(200),  # User embedding
        "geo": "New York City",  # Text feature
        "country": "United States"  # Text feature
    }
    
    product_features = [
        {
            "product": np.random.randn(200),  # Product embedding
            "price": np.array([10.0]),  # Numeric feature
            "category": np.array([1])  # Categorical feature
        }
        for _ in range(100)  # 100 candidate products
    ]
    
    # Get reranked products
    ranked_products = reranker.rerank_products(
        user_features,
        product_features,
        top_k=10
    )
    
    # Print results
    logger.info("Top 10 products:")
    for idx, score in ranked_products:
        logger.info(f"Product {idx}: Score = {score:.4f}")

if __name__ == "__main__":
    main() 