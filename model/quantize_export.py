### quantize_export.py

import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Optional: full int8 if needed
# def representative_dataset():
#     for i in range(100):
#         yield [X_train[i:i+1].astype(np.float32)]
# converter.representative_dataset = representative_dataset
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8

model_tflite = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(model_tflite)
