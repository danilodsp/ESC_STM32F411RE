# train.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from esc_cnn import build_model
import tensorflow as tf

def load_features_labels(prepared_data_dir):
    # Load preprocessed dataset
    X = np.load(os.path.join(prepared_data_dir, "X.npy"))
    y = np.load(os.path.join(prepared_data_dir, "y.npy"))

    # Expand dims if necessary (add channels=1)
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=-1)

    # One-hot encode labels
    y = to_categorical(y)

    return X, y

def main():
    prepared_data_dir = "prepared_data"

    X, y = load_features_labels(prepared_data_dir)

    print(f"✅ Loaded dataset: X.shape = {X.shape}, y.shape = {y.shape}")

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y.argmax(axis=1)
    )

    # Build model
    model = build_model(input_shape=X.shape[1:], num_classes=y.shape[1])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='logs')
    ]

    # Train model
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks
    )

    # Save the final model
    model.save("model_data")

    print("✅ Training complete. Model saved.")

if __name__ == "__main__":
    main()
