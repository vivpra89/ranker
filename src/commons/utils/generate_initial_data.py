from generate_dummy_data import generate_dummy_data

if __name__ == "__main__":
    # Generate dummy data once
    features_df, embeddings_dict, labels_df = generate_dummy_data(
        num_samples=1000,
        embedding_dim=64,
        output_dir="data/test"
    )
    print("Initial data generation complete. You can now use train_model.py to load and use this data.") 