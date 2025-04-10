import torch
from .dataset import pad_sequences

def collate_fn(batch, config):
    """Custom collate function to handle different feature types."""
    feature_config = config['features']['feature_configs']
    enabled_features = config['features']['enabled_features']
    data_config = config['data']['input']
    
    # Initialize feature lists only for enabled features that are present in the batch
    features = {name: [] for name in enabled_features if feature_config[name]['enabled'] and name in batch[0][0]}
    labels = {name: [] for name in batch[0][1]}
    
    # Collect features and labels from batch
    for sample_features, sample_labels in batch:
        for name, value in sample_features.items():
            if name in features:  # Only collect enabled features
                features[name].append(value)
        for name, value in sample_labels.items():
            labels[name].append(value)
    
    # Process features based on their type
    processed_features = {}
    for name, values in features.items():
        if not values:  # Skip empty feature lists
            continue
            
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
            try:
                processed_features[name] = torch.stack(values)
            except:
                print(f"Error stacking feature {name}. Values: {values}")
                raise
    
    # Stack labels
    processed_labels = {
        name: torch.stack(values)
        for name, values in labels.items()
    }
    
    return processed_features, processed_labels 