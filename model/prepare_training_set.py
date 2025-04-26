# prepare_training_set.py

import os
import pandas as pd
import numpy as np

def prepare_dataset(features_dir, metadata_csv_path, output_dir="prepared_data"):
    os.makedirs(output_dir, exist_ok=True)

    # Load metadata CSV
    metadata = pd.read_csv(metadata_csv_path)

    features = []
    labels = []

    for idx, row in metadata.iterrows():
        fold = row['fold']
        file_name = row['slice_file_name'].replace('.wav', '')  # Remove '.wav'
        class_id = row['classID']

        # Correct filename format (flat structure)
        feature_file = f"fold{fold}_{file_name}.npy"
        feature_path = os.path.join(features_dir, feature_file)

        if os.path.exists(feature_path):
            feature = np.load(feature_path)
            features.append(feature)
            labels.append(class_id)
        else:
            print(f"Warning: Feature not found: {feature_path}")

    # Stack into arrays
    X = np.stack(features)
    y = np.array(labels)

    # Save clean dataset
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)

    print(f"✅ Prepared {len(X)} samples. Saved to '{output_dir}/'.")

if __name__ == "__main__":
    features_dir = "features"
    metadata_csv_path = os.path.join(os.getenv('UrbanSound8K_dataset'), "metadata", "UrbanSound8K.csv")
    prepare_dataset(features_dir, metadata_csv_path)
