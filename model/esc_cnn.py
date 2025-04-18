"""
CNN model builder for ESC audio classification.

Defines a simple convolutional neural network for log-mel spectrogram input.
"""
from typing import Tuple
from tensorflow.keras import layers, models

def build_model(
    input_shape: Tuple[int, int, int] = (100, 40, 1),
    num_classes: int = 10
) -> models.Model:
    """
    Build a CNN model for ESC audio classification.

    Args:
        input_shape (Tuple[int, int, int]): Shape of the input features.
        num_classes (int): Number of output classes.

    Returns:
        models.Model: Compiled Keras model.
    """
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
