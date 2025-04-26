# evaluate_tflite.py

import time
import numpy as np
import tensorflow as tf
import argparse
from sklearn.metrics import accuracy_score

def evaluate_tflite_model(tflite_model_path, X_path, y_path):
    # Load model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"✅ Model loaded: {tflite_model_path}")
    print(f"Input shape: {input_details[0]['shape']}")

    # Load data
    X = np.load(X_path)
    y = np.load(y_path)

    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=-1)

    y_true = y

    y_pred = []

    for i in range(len(X)):
        input_sample = X[i:i+1]

        # If int8 model, apply quantization
        if input_details[0]['dtype'] == np.int8:
            scale, zero_point = input_details[0]['quantization']
            input_sample = (input_sample / scale + zero_point).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], input_sample)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])

        pred_label = np.argmax(output)
        y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    print(f"🎯 Accuracy on evaluation set: {acc * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TFLite model accuracy.")
    parser.add_argument('--tflite_model', type=str, default='model.tflite', help='Path to TFLite model')
    parser.add_argument('--X_path', type=str, default='prepared_data/X.npy', help='Path to X.npy')
    parser.add_argument('--y_path', type=str, default='prepared_data/y.npy', help='Path to y.npy')

    args = parser.parse_args()

    evaluate_tflite_model(
        tflite_model_path=args.tflite_model,
        X_path=args.X_path,
        y_path=args.y_path
    )
