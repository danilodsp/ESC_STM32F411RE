"""
Model training script for ESC audio classification.

Loads pre-extracted features and labels, splits data, trains a CNN,
and saves the model.
"""
import os
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from esc_cnn import build_model


def load_features_labels(features_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load features and labels from a directory of .npy files.

    Args:
        features_dir (str): Directory containing .npy feature files.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features and one-hot encoded labels.
    """
    features = []
    labels = []
    for file in os.listdir(features_dir):
        if file.endswith('.npy'):
            features.append(np.load(os.path.join(features_dir, file)))
            # Assumes label_filename.wav.npy
            labels.append(int(file.split('_')[0]))
    X = np.stack(features)[..., np.newaxis]
    y = to_categorical(labels)
    return X, y


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 30,
    batch_size: int = 16,
    model_save_path: str = "saved_model"
) -> None:
    """
    Train a CNN model on the provided features and labels.

    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): One-hot encoded labels.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        model_save_path (str): Path to save the trained model.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    model = build_model(input_shape=X.shape[1:], num_classes=y.shape[1])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Train ESC CNN model."
    )
    parser.add_argument(
        '--features_dir',
        type=str,
        default='features',
        help='Directory with .npy feature files.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size.'
    )
    parser.add_argument(
        '--model_save_path',
        type=str,
        default='saved_model',
        help='Path to save the trained model.'
    )
    args = parser.parse_args()
    X, y = load_features_labels(args.features_dir)
    train_model(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=args.model_save_path
    )


if __name__ == "__main__":
    main()
