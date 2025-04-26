# quantize_export.py

import argparse
import os
import tensorflow as tf
import numpy as np

def load_representative_dataset(X_path, num_samples=100):
    X = np.load(X_path)
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=-1)

    for i in range(min(num_samples, len(X))):
        sample = X[i:i+1].astype(np.float32)
        yield [sample]

def export_tflite(saved_model_dir, output_path, quantize_int8=False, X_path=None):
    """Exports TFLite model with optional int8 quantization."""
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quantize_int8:
        if X_path is None:
            raise ValueError("Representative dataset (X.npy) is required for int8 quantization.")

        converter.representative_dataset = lambda: load_representative_dataset(X_path)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"✅ Exported TFLite model to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Export TFLite model (float32 or int8).")
    parser.add_argument('--saved_model_dir', type=str, default='saved_model', help='Path to SavedModel directory.')
    parser.add_argument('--output_path', type=str, default='model.tflite', help='Output TFLite file path.')
    parser.add_argument('--int8', action='store_true', help='Enable full int8 quantization.')
    parser.add_argument('--X_path', type=str, default='prepared_data/X.npy', help='Path to X.npy for representative dataset.')
    
    args = parser.parse_args()

    export_tflite(
        saved_model_dir=args.saved_model_dir,
        output_path=args.output_path,
        quantize_int8=args.int8,
        X_path=args.X_path
    )

if __name__ == "__main__":
    main()
