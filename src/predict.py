import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from commons.inference.predictor import Predictor

def run_inference(model_path: str, data_dir: str, output_path: str, batch_size: int = 32):
    """Run inference for a single model configuration."""
    print(f"\nRunning inference for model: {model_path}")
    
    # Load test data
    print("Loading test data...")
    features_df = pd.read_csv(os.path.join(data_dir, 'features.csv'))
    embeddings_dict = {}  # Add embeddings if needed
    
    # Initialize predictor
    print("Initializing predictor...")
    predictor = Predictor(model_path)
    
    # Generate predictions
    print("Generating predictions...")
    
    # Print feature shapes for debugging
    print("\nFeature shapes:")
    for col in features_df.columns:
        if col.endswith('_0'):
            base_col = col[:-2]
            print(f"{base_col}: {features_df[col].values.shape}")
        else:
            print(f"{col}: {features_df[col].values.shape}")
    
    predictions = predictor.predict(features_df, embeddings_dict, batch_size=batch_size)
    
    # Print label shapes for debugging
    print("\nLabel shapes:")
    for task, preds in predictions.items():
        print(f"{task}: {preds.shape}")
    
    # Flatten predictions and save
    flattened_predictions = {task: preds.flatten() for task, preds in predictions.items()}
    pd.DataFrame(flattened_predictions).to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

def main():
    """Run inference for all model configurations."""
    # Define configurations
    configs = [
        {
            'name': 'base',
            'model_path': 'models/final_model_base_config.pt',
            'output_path': 'predictions_base.csv'
        },
        {
            'name': 'moe',
            'model_path': 'models/final_model_moe_config.pt',
            'output_path': 'predictions_moe.csv'
        },
        {
            'name': 'multihead',
            'model_path': 'models/final_model_multihead_sequence_config.pt',
            'output_path': 'predictions_multihead.csv'
        },
        {
            'name': 'tuning',
            'model_path': 'models/final_model_tuning_config.pt',
            'output_path': 'predictions_tuning.csv'
        }
    ]
    
    # Run inference for each configuration
    for config in configs:
        try:
            run_inference(
                model_path=config['model_path'],
                data_dir='data/test',
                output_path=config['output_path'],
                batch_size=32
            )
        except Exception as e:
            print(f"\nError running inference for {config['name']} configuration:")
            print(str(e))

if __name__ == '__main__':
    main() 