import torch
from torch.utils.data import Dataset
from typing import Dict, Union, List, Any
import numpy as np

def pad_sequences(sequences: List[List[int]], max_length: int, padding_value: int = 0, truncation: str = "right"):
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
            if name not in features or not self.feature_config[name]['enabled']:
                continue
                
            feature = features[name]
            feature_config = self.feature_config[name]
            
            try:
                if feature_config['type'] == 'pretrained_embedding':
                    # Convert object array to float array if needed
                    if isinstance(feature, np.ndarray) and feature.dtype == np.dtype('O'):
                        feature = np.stack(feature).astype(np.float32)
                    self.features[name] = torch.FloatTensor(feature)
                elif feature_config['type'] == 'binary_embedding':
                    if isinstance(feature, np.ndarray) and feature.dtype == np.dtype('O'):
                        feature = np.stack(feature).astype(np.float32)
                    self.features[name] = torch.FloatTensor(feature)
                elif feature_config['type'] == 'categorical':
                    if isinstance(feature, np.ndarray) and feature.dtype == np.dtype('O'):
                        feature = np.array(feature.tolist(), dtype=np.int64)
                    self.features[name] = torch.LongTensor(feature)
                elif feature_config['type'] == 'numeric':
                    if isinstance(feature, np.ndarray) and feature.dtype == np.dtype('O'):
                        feature = np.array(feature.tolist(), dtype=np.float32)
                    self.features[name] = torch.FloatTensor(feature.reshape(-1, 1))
                else:
                    if isinstance(feature, np.ndarray) and feature.dtype == np.dtype('O'):
                        feature = np.stack(feature).astype(np.float32)
                    self.features[name] = torch.FloatTensor(feature)
            except Exception as e:
                print(f"Error processing feature {name}: {str(e)}")
                print(f"Feature type: {feature_config['type']}")
                print(f"Feature shape: {feature.shape if hasattr(feature, 'shape') else len(feature)}")
                print(f"Feature dtype: {feature.dtype if hasattr(feature, 'dtype') else type(feature)}")
                raise
        
        # Process labels
        self.labels = {}
        for name, label in labels.items():
            self.labels[name] = torch.FloatTensor(label)
        
        # Print feature shapes for debugging
        print("\nFeature shapes:")
        for name, feature in self.features.items():
            print(f"{name}: {feature.shape}")
        print("\nLabel shapes:")
        for name, label in self.labels.items():
            print(f"{name}: {label.shape}")
        
    def __len__(self):
        return len(next(iter(self.labels.values())))
    
    def __getitem__(self, idx):
        # Return features based on their type
        batch_features = {}
        for name, feature in self.features.items():
            if name in self.enabled_features and self.feature_config[name]['enabled']:
                batch_features[name] = feature[idx]
        
        return (
            batch_features,
            {name: label[idx] for name, label in self.labels.items()}
        ) 