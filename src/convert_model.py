"""
Model conversion script to TensorFlow Lite format
Optimizes model for deployment on Raspberry Pi 4
"""
import os
import numpy as np
import tensorflow as tf
import keras

from utils import Config, load_trained_model
from data_processing import create_tf_datasets


def convert_to_tflite(model_path: str,
                     output_path: str,
                     quantization: str = 'float16',
                     representative_dataset=None):
    """
    Convert Keras model to TensorFlow Lite format
    
    Args:
        model_path: Path to the Keras model
        output_path: Path to save TFLite model
        quantization: Quantization type ('float16', 'dynamic', 'int8', None)
        representative_dataset: Dataset for full integer quantization
    
    Returns:
        Path to converted model
    """
    print("\n" + "="*60)
    print("TENSORFLOW LITE CONVERSION")
    print("="*60)
    
    # Load model
    print(f"\n📥 Loading model from {model_path}")
    model = load_trained_model(model_path)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations based on quantization type
    if quantization == 'float16':
        print("\n🔧 Applying Float16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
    elif quantization == 'dynamic':
        print("\n🔧 Applying Dynamic range quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
    elif quantization == 'int8':
        if representative_dataset is None:
            raise ValueError("Representative dataset required for int8 quantization")
        
        print("\n🔧 Applying Full Integer (int8) quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
    else:
        print("\n🔧 No quantization applied (full precision)")
    
    # Convert model
    print("\n🔄 Converting model...")
    tflite_model = converter.convert()
    
    # Save TFLite model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get model sizes
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    tflite_size = os.path.getsize(output_path) / (1024 * 1024)
    compression_ratio = (1 - tflite_size / original_size) * 100
    
    print("\n" + "="*60)
    print("CONVERSION RESULTS")
    print("="*60)
    print(f"✅ TFLite model saved to: {output_path}")
    print(f"📊 Original model size: {original_size:.2f} MB")
    print(f"📊 TFLite model size: {tflite_size:.2f} MB")
    print(f"📊 Compression ratio: {compression_ratio:.2f}%")
    print("="*60 + "\n")
    
    return output_path


def create_representative_dataset_generator(num_samples: int = 100):
    """
    Create representative dataset for int8 quantization
    
    Args:
        num_samples: Number of samples to use
    
    Returns:
        Generator function for representative dataset
    """
    print(f"\n🔄 Creating representative dataset ({num_samples} samples)...")
    
    # Load training dataset
    train_ds, _, _ = create_tf_datasets(batch_size=1)
    
    def representative_dataset():
        count = 0
        for images, _ in train_ds:
            if count >= num_samples:
                break
            yield [images]
            count += 1
    
    return representative_dataset


def verify_tflite_model(tflite_path: str, test_image_path: str = None):
    """
    Verify TFLite model by running inference
    
    Args:
        tflite_path: Path to TFLite model
        test_image_path: Optional test image path
    """
    print("\n🔍 Verifying TFLite model...")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\n📋 Model Details:")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Input type: {input_details[0]['dtype']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    print(f"   Output type: {output_details[0]['dtype']}")
    
    # Test with random input if no test image provided
    if test_image_path:
        print(f"\n🔄 Testing with image: {test_image_path}")
        img = keras.preprocessing.image.load_img(
            test_image_path, 
            target_size=Config.IMG_SIZE
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(input_details[0]['dtype'])
    else:
        print(f"\n🔄 Testing with random input...")
        img_array = np.random.rand(1, *Config.IMG_SIZE, 3).astype(input_details[0]['dtype'])
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"\n✅ Inference successful!")
    print(f"   Output: {output}")
    print(f"   Output shape: {output.shape}")
    
    return True


def benchmark_tflite_model(tflite_path: str, num_runs: int = 100):
    """
    Benchmark TFLite model inference speed
    
    Args:
        tflite_path: Path to TFLite model
        num_runs: Number of inference runs
    """
    import time
    
    print(f"\n⏱️  Benchmarking TFLite model ({num_runs} runs)...")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    
    # Prepare random input
    test_input = np.random.rand(1, *Config.IMG_SIZE, 3).astype(input_details[0]['dtype'])
    
    # Warmup run
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        end_time = time.time()
        times.append(end_time - start_time)
    
    # Calculate statistics
    mean_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    fps = 1000 / mean_time
    
    print(f"\n📊 Benchmark Results:")
    print(f"   Mean inference time: {mean_time:.2f} ms (±{std_time:.2f} ms)")
    print(f"   Min inference time: {min_time:.2f} ms")
    print(f"   Max inference time: {max_time:.2f} ms")
    print(f"   Throughput: {fps:.2f} FPS")


def generate_raspberry_pi_example():
    """Generate example code for Raspberry Pi inference"""
    
    example_code = '''
# ============================================================================
# RASPBERRY PI INFERENCE EXAMPLE
# ============================================================================
# This code shows how to use the TFLite model on Raspberry Pi 4
# 
# Installation on Raspberry Pi:
#   pip install tensorflow-lite-runtime
#   pip install pillow numpy
# ============================================================================

import numpy as np
from PIL import Image
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# Configuration
MODEL_PATH = "palm_classifier.tflite"
IMG_SIZE = (224, 224)
CLASS_NAMES = ['sawit_mentah', 'sawit_matang']
CONFIDENCE_THRESHOLD = 0.70

def load_and_preprocess_image(image_path):
    """Load and preprocess image for inference"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(interpreter, image_array):
    """Run inference on preprocessed image"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]

def main():
    # Load TFLite model
    print("Loading model...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    print("Model loaded successfully!")
    
    # Test with an image
    image_path = "test_image.jpg"
    print(f"\\nProcessing image: {image_path}")
    
    # Preprocess image
    img_array = load_and_preprocess_image(image_path)
    
    # Predict
    predictions = predict_image(interpreter, img_array)
    
    # Get results
    predicted_class_idx = np.argmax(predictions)
    confidence = predictions[predicted_class_idx]
    
    # Check if unknown
    if confidence < CONFIDENCE_THRESHOLD:
        print(f"\\n❓ Result: UNKNOWN (low confidence: {confidence:.3f})")
    else:
        predicted_class = CLASS_NAMES[predicted_class_idx]
        print(f"\\n✅ Result: {predicted_class}")
        print(f"   Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
    
    # Show all probabilities
    print(f"\\n📊 All probabilities:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"   {class_name}: {predictions[i]:.3f}")

if __name__ == "__main__":
    main()
'''
    
    return example_code


def main():
    """Main conversion pipeline"""
    print("🚀 Starting TensorFlow Lite conversion...\n")
    
    # Check if model exists
    model_path = Config.MODELS_DIR / "palm_classifier_final.keras"
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("   Please run train_model.py first!")
        return
    
    # Convert with different quantization options
    quantization_types = ['float16', 'dynamic']
    
    for quant_type in quantization_types:
        output_path = Config.MODELS_DIR / f"palm_classifier_{quant_type}.tflite"
        
        if quant_type == 'int8':
            # Create representative dataset for int8 quantization
            rep_dataset = create_representative_dataset_generator(num_samples=100)
            convert_to_tflite(
                str(model_path),
                str(output_path),
                quantization=quant_type,
                representative_dataset=rep_dataset
            )
        else:
            convert_to_tflite(
                str(model_path),
                str(output_path),
                quantization=quant_type
            )
        
        # Verify model
        verify_tflite_model(str(output_path))
        
        # Benchmark model
        benchmark_tflite_model(str(output_path), num_runs=50)
    
    # Generate Raspberry Pi example
    print("\n" + "="*60)
    print("RASPBERRY PI DEPLOYMENT GUIDE")
    print("="*60)
    
    example_code = generate_raspberry_pi_example()
    example_path = Config.MODELS_DIR / "raspberry_pi_inference_example.py"
    
    with open(example_path, 'w') as f:
        f.write(example_code)
    
    print(f"\n✅ Raspberry Pi example saved to: {example_path}")
    
    print("\n📋 Recommended model for Raspberry Pi 4:")
    print("   Use: palm_classifier_float16.tflite")
    print("   Reason: Best balance between size and accuracy")
    
    print("\n📋 Deployment steps:")
    print("   1. Copy .tflite model to Raspberry Pi")
    print("   2. Install: pip install tensorflow-lite-runtime")
    print("   3. Use the example code in raspberry_pi_inference_example.py")
    print("   4. Test inference speed and accuracy on device")
    
    print("\n🎉 Conversion completed!")


if __name__ == "__main__":
    main()