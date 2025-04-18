"""
Model quantization and export script for TensorFlow Lite.

Converts a trained Keras model to TFLite format with optional quantization.
"""
import argparse
import tensorflow as tf

def quantize_and_export(
    saved_model_dir: str = "saved_model",
    tflite_path: str = "model.tflite",
    quantize: bool = True
) -> None:
    """
    Convert a TensorFlow SavedModel to TFLite format with optional quantization.

    Args:
        saved_model_dir (str): Path to the SavedModel directory.
        tflite_path (str): Output path for the TFLite model.
        quantize (bool): Whether to apply default quantization.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    model_tflite = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(model_tflite)
    print(f"TFLite model exported to {tflite_path}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize and export a TensorFlow SavedModel to TFLite."
    )
    parser.add_argument(
        '--saved_model_dir',
        type=str,
        default='saved_model',
        help='Path to the SavedModel directory.'
    )
    parser.add_argument(
        '--tflite_path',
        type=str,
        default='model.tflite',
        help='Output path for the TFLite model.'
    )
    parser.add_argument(
        '--no_quantize',
        action='store_true',
        help='Disable quantization.'
    )
    args = parser.parse_args()
    quantize_and_export(
        saved_model_dir=args.saved_model_dir,
        tflite_path=args.tflite_path,
        quantize=not args.no_quantize
    )


if __name__ == "__main__":
    main()
