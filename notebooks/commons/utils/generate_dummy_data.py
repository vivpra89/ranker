# Databricks notebook source
import numpy as np
import pandas as pd
from pathlib import Path

def generate_dummy_data(
    num_samples: int = 1000,
    embedding_dim: int = 64,
    output_dir: str = "data/test"
):
    """Generate dummy data for testing the ranker model.
    
    Args:
        num_samples: Number of samples to generate
        embedding_dim: Dimension of embeddings
        output_dir: Directory to save the generated data
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dummy features
    features_df = pd.DataFrame({
        # Original features
        'geo': np.random.choice(['US', 'UK', 'CA', 'AU'], size=num_samples),
        'country': np.random.choice(['USA', 'GBR', 'CAN', 'AUS'], size=num_samples),
        'price_0': np.random.uniform(10, 1000, size=num_samples),
        'category_0': np.random.choice(['Electronics', 'Books', 'Clothing', 'Home'], size=num_samples),
        
        # New features
        'page_type': np.random.choice(['Home', 'Category', 'Search', 'Product', 'Cart', 'Wishlist', 'Account', 'Other'], size=num_samples),
        'age': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55-64', '65+'], size=num_samples),
        'gender': np.random.choice(['M', 'F', 'Other'], size=num_samples),
        'region': np.random.choice([f'Region_{i}' for i in range(10)], size=num_samples),
        
        'product_body_part': np.random.choice([f'Body_{i}' for i in range(20)], size=num_samples),
        'product_gender': np.random.choice(['Men', 'Women', 'Unisex'], size=num_samples),
        'product_age_desc': np.random.choice(['Kids', 'Teen', 'Adult', 'Senior', 'All', 'Other'], size=num_samples),
        'product_taxonomy_id': np.random.randint(0, 100, size=num_samples),
        
        'anchor_body_part': np.random.choice([f'Body_{i}' for i in range(20)], size=num_samples),
        'anchor_gender': np.random.choice(['Men', 'Women', 'Unisex'], size=num_samples),
        'anchor_age_desc': np.random.choice(['Kids', 'Teen', 'Adult', 'Senior', 'All', 'Other'], size=num_samples),
        'anchor_taxonomy_id': np.random.randint(0, 100, size=num_samples),
        
        'event_dow': np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], size=num_samples)
    })
    
    # Generate dummy embeddings
    embeddings_dict = {
        # Original embeddings
        'user_embeddings': np.random.normal(0, 1, size=(num_samples, embedding_dim)).astype(np.float32),
        'product_embeddings': np.random.normal(0, 1, size=(num_samples, embedding_dim)).astype(np.float32),
        
        # New embeddings
        'anchor_embeds_binary': (np.random.random((num_samples, embedding_dim)) > 0.5).astype(np.float32),
        'anchor_embeds': np.random.normal(0, 1, size=(num_samples, embedding_dim)).astype(np.float32),
        'als_prod_embeds': np.random.normal(0, 1, size=(num_samples, embedding_dim)).astype(np.float32),
        'als_prod_embeds_binary': (np.random.random((num_samples, embedding_dim)) > 0.5).astype(np.float32),
        'als_user_embeds': np.random.normal(0, 1, size=(num_samples, embedding_dim)).astype(np.float32)
    }
    
    # Generate dummy labels
    labels_df = pd.DataFrame({
        'click': np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3]),
        'purchase': np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1]),
        'add_to_cart': np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])
    })
    
    # Save features to CSV
    features_df.to_csv(output_dir / "features.csv", index=False)
    
    # Save embeddings to NPY
    np.save(output_dir / "embeddings.npy", embeddings_dict)
    
    # Save labels to CSV
    labels_df.to_csv(output_dir / "labels.csv", index=False)
    
    print(f"Generated {num_samples} samples of dummy data:")
    print(f"Features saved to: {output_dir}/features.csv")
    print(f"Embeddings saved to: {output_dir}/embeddings.npy")
    print(f"Labels saved to: {output_dir}/labels.csv")
    
    return features_df, embeddings_dict, labels_df 