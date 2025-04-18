### train.py

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from esc_cnn import build_model
import tensorflow as tf

# Load features and labels
features = []
labels = []

# Dummy loader; update for your own feature directory
import os
for file in os.listdir('features/'):
    if file.endswith('.npy'):
        features.append(np.load(os.path.join('features/', file)))
        labels.append(int(file.split('_')[0]))  # Assuming label_filename.wav.npy

X = np.stack(features)[..., np.newaxis]  # shape: (samples, 100, 40, 1)
y = to_categorical(labels)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

model = build_model(input_shape=X.shape[1:], num_classes=y.shape[1])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=16)
model.save("saved_model")
