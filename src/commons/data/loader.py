import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.preprocessing import LabelEncoder

def load_and_process_data(config: Dict) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Load and process the data.
    
    Args:
        config: Configuration dictionary containing feature and data settings
        
    Returns:
        Tuple of (features, labels) where:
            features: Dictionary of processed features
            labels: Dictionary of labels
    """
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