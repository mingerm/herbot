"""
Herbify - Medicinal Herb Inference on Raspberry Pi 4 + Coral Edge TPU
"""

import os
import time
import json
import numpy as np
from PIL import Image
import argparse

# Try to import tflite_runtime
try:
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate
    print("Using tflite_runtime")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    load_delegate = tf.lite.experimental.load_delegate
    print("Using tensorflow.lite")


class HerbClassifier:
    """Medicinal herb classifier using Edge TPU"""

    def __init__(self, model_path, class_names_path, use_edgetpu=True):
        self.model_path = model_path
        self.use_edgetpu = use_edgetpu

        # Load class names
        with open(class_names_path, 'r') as f:
            class_info = json.load(f)
            self.class_names = class_info['class_names']
            self.num_classes = class_info['num_classes']

        print(f"Loaded {self.num_classes} herb classes")

        # Load TFLite model
        self._load_model()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        self.input_dtype = self.input_details[0]['dtype']

        self.is_quantized = self.input_dtype == np.uint8

        print(f"Input shape: {self.input_shape}")
        print(f"Quantized: {self.is_quantized}")

        if self.is_quantized:
            self.input_scale, self.input_zero_point = self.input_details[0]['quantization']
            self.output_scale, self.output_zero_point = self.output_details[0]['quantization']

    def _load_model(self):
        """Load TFLite model with or without Edge TPU"""

        if self.use_edgetpu:
            try:
                print("Loading model with Edge TPU acceleration...")
                self.interpreter = Interpreter(
                    model_path=self.model_path,
                    experimental_delegates=[load_delegate('libedgetpu.so.1')]
                )
                print("Edge TPU delegate loaded successfully!")
            except Exception as e:
                print(f"Failed to load Edge TPU delegate: {e}")
                print("Falling back to CPU...")
                self.use_edgetpu = False
                self.interpreter = Interpreter(model_path=self.model_path)
        else:
            print("Loading model for CPU inference...")
            self.interpreter = Interpreter(model_path=self.model_path)

        self.interpreter.allocate_tensors()

    def preprocess_image(self, image_path):
        """Preprocess image for inference"""

        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')

        image = image.resize((self.input_width, self.input_height), Image.BILINEAR)
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)

        if self.is_quantized:
            image_array = image_array.astype(np.uint8)
        else:
            image_array = image_array.astype(np.float32) / 255.0

        return image_array

    def predict(self, image_path, top_k=5):
        """Predict herb species from image"""

        # Preprocess
        input_data = self.preprocess_image(image_path)

        # Set input
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Run inference
        start_time = time.time()
        self.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000

        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        output_data = output_data[0]

        # Dequantize if needed
        if self.is_quantized:
            output_data = (output_data.astype(np.float32) - self.output_zero_point) * self.output_scale

        # Apply softmax
        if not np.allclose(np.sum(output_data), 1.0, atol=0.1):
            exp_output = np.exp(output_data - np.max(output_data))
            output_data = exp_output / np.sum(exp_output)

        # Get top K predictions
        top_k_indices = np.argsort(output_data)[-top_k:][::-1]

        results = []
        for idx in top_k_indices:
            class_name = self.class_names[idx]
            confidence = float(output_data[idx])
            results.append((class_name, confidence))

        return results, inference_time


def main():
    """Main function"""

    parser = argparse.ArgumentParser(description='Herbify - Medicinal Herb Classification')
    parser.add_argument('--model', type=str, required=True, help='Path to TFLite model')
    parser.add_argument('--classes', type=str, required=True, help='Path to class names JSON')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--images', type=str, nargs='+', help='Paths to multiple images')
    parser.add_argument('--no-edgetpu', action='store_true', help='Disable Edge TPU')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top predictions')

    args = parser.parse_args()

    print("=" * 70)
    print("Herbify - Medicinal Herb Classification")
    print("=" * 70)

    classifier = HerbClassifier(
        model_path=args.model,
        class_names_path=args.classes,
        use_edgetpu=not args.no_edgetpu
    )

    print("\n" + "=" * 70)

    if args.image:
        print(f"\nProcessing: {args.image}")
        results, inference_time = classifier.predict(args.image, top_k=args.top_k)

        print(f"\nInference time: {inference_time:.2f} ms")
        print(f"FPS: {1000 / inference_time:.2f}")
        print("\nTop predictions:")
        for i, (class_name, confidence) in enumerate(results, 1):
            print(f"  {i}. {class_name}: {confidence * 100:.2f}%")

    elif args.images:
        print(f"\nProcessing {len(args.images)} images...")
        total_time = 0

        for img_path in args.images:
            results, inference_time = classifier.predict(img_path, top_k=args.top_k)
            total_time += inference_time

            print(f"\n{img_path}:")
            print(f"  Time: {inference_time:.2f} ms")
            print(f"  Top: {results[0][0]} ({results[0][1] * 100:.2f}%)")

        avg_time = total_time / len(args.images)
        print(f"\nAverage: {avg_time:.2f} ms ({1000 / avg_time:.2f} FPS)")

    else:
        print("\nNo images specified. Use --image or --images")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
