# compare_models.py

import os
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
import argparse

def load_dataset(X_path, y_path):
    X = np.load(X_path)
    y = np.load(y_path)

    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=-1)

    return X, y

def evaluate_model(interpreter, X, y, num_runs=100):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_pred = []
    times = []

    for i in range(len(X)):
        input_sample = X[i:i+1]

        # If int8 model, apply quantization
        if input_details[0]['dtype'] == np.int8:
            scale, zero_point = input_details[0]['quantization']
            input_sample = (input_sample / scale + zero_point).astype(np.int8)

        # Benchmark
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], input_sample)
        interpreter.invoke()
        end = time.perf_counter()

        output = interpreter.get_tensor(output_details[0]['index'])
        pred_label = np.argmax(output)
        y_pred.append(pred_label)
        times.append(end - start)

        if i >= num_runs:  # Only use num_runs for benchmarking
            break

    acc = accuracy_score(y[:len(y_pred)], y_pred)
    avg_time_ms = np.mean(times) * 1000  # ms

    return acc, avg_time_ms

def compare_models(float_model_path, int8_model_path, X_path, y_path, num_runs=100):
    X, y = load_dataset(X_path, y_path)

    # Float32 model
    interpreter_float = tf.lite.Interpreter(model_path=float_model_path)
    interpreter_float.allocate_tensors()
    acc_float, time_float = evaluate_model(interpreter_float, X, y, num_runs)

    # Int8 model
    interpreter_int8 = tf.lite.Interpreter(model_path=int8_model_path)
    interpreter_int8.allocate_tensors()
    acc_int8, time_int8 = evaluate_model(interpreter_int8, X, y, num_runs)

    # Display results
    print("\n📊 Model Comparison Summary:\n")
    print(f"{'Model':<10} | {'Accuracy (%)':<15} | {'Avg Inference Time (ms)':<25}")
    print("-"*55)
    print(f"{'Float32':<10} | {acc_float*100:>13.2f} | {time_float:>23.3f}")
    print(f"{'Int8':<10} | {acc_int8*100:>13.2f} | {time_int8:>23.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare float and int8 TFLite models.")
    parser.add_argument('--float_model', type=str, default='model.tflite', help='Path to float32 TFLite model.')
    parser.add_argument('--int8_model', type=str, default='quantized_model_int8.tflite', help='Path to int8 TFLite model.')
    parser.add_argument('--X_path', type=str, default='prepared_data/X.npy', help='Path to X.npy')
    parser.add_argument('--y_path', type=str, default='prepared_data/y.npy', help='Path to y.npy')
    parser.add_argument('--num_runs', type=int, default=100, help='Number of inference runs for timing')

    args = parser.parse_args()

    compare_models(
        float_model_path=args.float_model,
        int8_model_path=args.int8_model,
        X_path=args.X_path,
        y_path=args.y_path,
        num_runs=args.num_runs
    )
