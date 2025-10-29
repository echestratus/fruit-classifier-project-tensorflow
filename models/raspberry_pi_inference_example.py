
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
    print(f"\nProcessing image: {image_path}")
    
    # Preprocess image
    img_array = load_and_preprocess_image(image_path)
    
    # Predict
    predictions = predict_image(interpreter, img_array)
    
    # Get results
    predicted_class_idx = np.argmax(predictions)
    confidence = predictions[predicted_class_idx]
    
    # Check if unknown
    if confidence < CONFIDENCE_THRESHOLD:
        print(f"\n❓ Result: UNKNOWN (low confidence: {confidence:.3f})")
    else:
        predicted_class = CLASS_NAMES[predicted_class_idx]
        print(f"\n✅ Result: {predicted_class}")
        print(f"   Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
    
    # Show all probabilities
    print(f"\n📊 All probabilities:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"   {class_name}: {predictions[i]:.3f}")

if __name__ == "__main__":
    main()
