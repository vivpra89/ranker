# Databricks notebook source
import numpy as np
import pandas as pd
import json
from pathlib import Path

def prepare_test_data(
    features_path: str,
    embeddings_path: str,
    labels_path: str,
    output_path: str
):
    """Prepare test data in the format expected by the evaluator."""
    
    # Load data
    features_df = pd.read_csv(features_path)
    embeddings_dict = np.load(embeddings_path, allow_pickle=True).item()
    labels_df = pd.read_csv(labels_path)
    
    # Extract user and product embeddings from dictionary
    user_embeddings = np.array(embeddings_dict.get('user_embeddings', []), dtype=np.float32)
    product_embeddings = np.array(embeddings_dict.get('product_embeddings', []), dtype=np.float32)
    
    # Prepare test data dictionary
    test_data = {
        'user_features': {
            'user': user_embeddings.tolist(),
            'geo': features_df['geo'].tolist(),
            'country': features_df['country'].tolist()
        },
        'product_features': {
            'product': product_embeddings.tolist(),
            'price': features_df['price_0'].tolist(),
            'category': features_df['category_0'].tolist()
        },
        'labels': {
            'click': labels_df['click'].tolist(),
            'purchase': labels_df['purchase'].tolist(),
            'add_to_cart': labels_df['add_to_cart'].tolist()
        }
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(test_data, f)

if __name__ == "__main__":
    # Prepare paths
    data_dir = Path("data/test")
    features_path = data_dir / "features.csv"
    embeddings_path = data_dir / "embeddings.npy"
    labels_path = data_dir / "labels.csv"
    output_path = data_dir / "test_data.json"
    
    print("Preparing test data...")
    print(f"Features path: {features_path}")
    print(f"Embeddings path: {embeddings_path}")
    print(f"Labels path: {labels_path}")
    print(f"Output path: {output_path}")
    
    # Prepare test data
    prepare_test_data(
        features_path=str(features_path),
        embeddings_path=str(embeddings_path),
        labels_path=str(labels_path),
        output_path=str(output_path)
    )
    print("Test data preparation completed!") 