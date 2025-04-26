# main.py

import os
import sys

from extract_features import process_directory

def main():
    # Read dataset path from environment variable
    dataset_path = os.getenv('UrbanSound8K_dataset')
    if dataset_path is None:
        print("Error: Environment variable 'UrbanSound8K_dataset' is not set.")
        sys.exit(1)

    if not os.path.exists(dataset_path):
        print(f"Error: Path '{dataset_path}' does not exist.")
        sys.exit(1)

    # Define output directory
    output_dir = os.path.join(os.getcwd(), 'features')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing UrbanSound8K dataset from: {dataset_path}")
    print(f"Saving extracted features into: {output_dir}/")

    # Call feature extraction
    process_directory(dataset_path, output_dir)

    print("✅ Feature extraction completed successfully.")

if __name__ == "__main__":
    main()
