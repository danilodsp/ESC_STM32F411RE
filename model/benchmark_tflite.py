# benchmark_tflite.py

import time
import numpy as np
import tensorflow as tf
import argparse

def benchmark_tflite_model(tflite_model_path, X_sample_path, num_runs=100):
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"✅ Model loaded: {tflite_model_path}")
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")

    # Load a sample input
    X = np.load(X_sample_path)
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=-1)

    # Use the first sample
    input_sample = X[0:1]

    # Quantization: if model expects int8 inputs
    if input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        input_sample = (input_sample / scale + zero_point).astype(np.int8)

    # Warm-up runs
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], input_sample)
        interpreter.invoke()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], input_sample)
        interpreter.invoke()
        end = time.perf_counter()
        times.append(end - start)

    avg_time_ms = np.mean(times) * 1000
    print(f"⚡ Average inference time over {num_runs} runs: {avg_time_ms:.3f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a TFLite model inference time.")
    parser.add_argument('--tflite_model', type=str, default='model.tflite', help='Path to TFLite model file')
    parser.add_argument('--X_sample', type=str, default='prepared_data/X.npy', help='Path to X.npy sample')
    parser.add_argument('--num_runs', type=int, default=100, help='Number of runs for averaging')

    args = parser.parse_args()

    benchmark_tflite_model(
        tflite_model_path=args.tflite_model,
        X_sample_path=args.X_sample,
        num_runs=args.num_runs
    )
