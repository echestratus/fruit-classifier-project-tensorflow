"""
Testing script for palm fruit classification
Supports single image and batch testing with both Keras and TFLite models
"""
import os
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple
import keras
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

from utils import (
    Config,
    load_trained_model,
    preprocess_image,
    is_prediction_uncertain,
    calculate_entropy
)


def predict_with_keras_model(model, image_path: str) -> Tuple[np.ndarray, str, float, bool]:
    """
    Predict using Keras model
    
    Args:
        model: Loaded Keras model
        image_path: Path to image
    
    Returns:
        Tuple of (probabilities, predicted_class, confidence, is_unknown)
    """
    # Preprocess image
    img_array = preprocess_image(image_path, Config.IMG_SIZE)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Get predicted class
    predicted_idx = np.argmax(predictions)
    predicted_class = Config.CLASS_NAMES[predicted_idx]
    confidence = predictions[predicted_idx]
    
    # Check if uncertain/unknown
    is_unknown, reason = is_prediction_uncertain(
        predictions,
        Config.CONFIDENCE_THRESHOLD,
        Config.ENTROPY_THRESHOLD
    )
    
    return predictions, predicted_class, confidence, is_unknown


def predict_with_tflite_model(interpreter, image_path: str) -> Tuple[np.ndarray, str, float, bool]:
    """
    Predict using TFLite model
    
    Args:
        interpreter: TFLite interpreter
        image_path: Path to image
    
    Returns:
        Tuple of (probabilities, predicted_class, confidence, is_unknown)
    """
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess image
    img_array = preprocess_image(image_path, Config.IMG_SIZE)
    img_array = img_array.astype(input_details[0]['dtype'])
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Get predicted class
    predicted_idx = np.argmax(predictions)
    predicted_class = Config.CLASS_NAMES[predicted_idx]
    confidence = predictions[predicted_idx]
    
    # Check if uncertain/unknown
    is_unknown, reason = is_prediction_uncertain(
        predictions,
        Config.CONFIDENCE_THRESHOLD,
        Config.ENTROPY_THRESHOLD
    )
    
    return predictions, predicted_class, confidence, is_unknown


def test_single_image(model_path: str, image_path: str, use_tflite: bool = False):
    """
    Test model on a single image
    
    Args:
        model_path: Path to model file
        image_path: Path to test image
        use_tflite: Whether to use TFLite model
    """
    print("\n" + "="*60)
    print("SINGLE IMAGE PREDICTION")
    print("="*60)
    
    print(f"\n📷 Image: {image_path}")
    print(f"🤖 Model: {model_path}")
    print(f"🔧 Format: {'TFLite' if use_tflite else 'Keras'}")
    
    # Load model
    if use_tflite:
        print("\n📥 Loading TFLite model...")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        predictions, predicted_class, confidence, is_unknown = predict_with_tflite_model(
            interpreter, image_path
        )
    else:
        print("\n📥 Loading Keras model...")
        model = load_trained_model(model_path)
        predictions, predicted_class, confidence, is_unknown = predict_with_keras_model(
            model, image_path
        )
    
    # Calculate entropy
    entropy = calculate_entropy(predictions)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    if is_unknown:
        print(f"\n❓ Prediction: UNKNOWN")
        print(f"   Reason: Low confidence or high uncertainty")
        print(f"   Top prediction would be: {predicted_class}")
        print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    else:
        print(f"\n✅ Prediction: {predicted_class}")
        print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    print(f"\n📊 Prediction Uncertainty:")
    print(f"   Entropy: {entropy:.4f}")
    print(f"   Threshold: {Config.CONFIDENCE_THRESHOLD}")
    
    print(f"\n📊 All Class Probabilities:")
    for i, class_name in enumerate(Config.CLASS_NAMES):
        prob = predictions[i]
        bar = "█" * int(prob * 50)
        print(f"   {class_name:20s}: {prob:.4f} ({prob*100:.2f}%) {bar}")
    
    print("\n" + "="*60)
    
    # Display image with prediction
    display_image_with_prediction(
        image_path, 
        predicted_class if not is_unknown else "UNKNOWN",
        confidence,
        is_unknown
    )


def test_batch_images(model_path: str, image_dir: str, use_tflite: bool = False):
    """
    Test model on a batch of images in a directory
    
    Args:
        model_path: Path to model file
        image_dir: Directory containing test images
        use_tflite: Whether to use TFLite model
    """
    print("\n" + "="*60)
    print("BATCH IMAGE PREDICTION")
    print("="*60)
    
    print(f"\n📁 Image directory: {image_dir}")
    print(f"🤖 Model: {model_path}")
    print(f"🔧 Format: {'TFLite' if use_tflite else 'Keras'}")
    
    # Get all image files
    image_dir = Path(image_dir)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in image_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in valid_extensions]
    
    if len(image_files) == 0:
        print(f"\n❌ No images found in {image_dir}")
        return
    
    print(f"\n📷 Found {len(image_files)} images")
    
    # Load model
    if use_tflite:
        print("\n📥 Loading TFLite model...")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    else:
        print("\n📥 Loading Keras model...")
        model = load_trained_model(model_path)
    
    # Process all images
    results = []
    unknown_count = 0
    
    print("\n🔄 Processing images...\n")
    
    for img_file in image_files:
        try:
            if use_tflite:
                predictions, predicted_class, confidence, is_unknown = predict_with_tflite_model(
                    interpreter, str(img_file)
                )
            else:
                predictions, predicted_class, confidence, is_unknown = predict_with_keras_model(
                    model, str(img_file)
                )
            
            entropy = calculate_entropy(predictions)
            
            results.append({
                'filename': img_file.name,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'is_unknown': is_unknown,
                'entropy': entropy,
                'all_predictions': predictions
            })
            
            if is_unknown:
                unknown_count += 1
                status = "❓ UNKNOWN"
            else:
                status = f"✅ {predicted_class}"
            
            print(f"   {img_file.name:30s} -> {status:20s} (conf: {confidence:.3f})")
            
        except Exception as e:
            print(f"   ❌ Error processing {img_file.name}: {e}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("BATCH PREDICTION SUMMARY")
    print("="*60)
    
    print(f"\n📊 Total images processed: {len(results)}")
    print(f"📊 Unknown predictions: {unknown_count} ({unknown_count/len(results)*100:.1f}%)")
    
    # Class distribution
    print(f"\n📊 Prediction Distribution:")
    for class_name in Config.CLASS_NAMES:
        count = sum(1 for r in results 
                   if r['predicted_class'] == class_name and not r['is_unknown'])
        print(f"   {class_name}: {count}")
    
    # Confidence statistics
    confidences = [r['confidence'] for r in results]
    print(f"\n📊 Confidence Statistics:")
    print(f"   Mean: {np.mean(confidences):.4f}")
    print(f"   Median: {np.median(confidences):.4f}")
    print(f"   Min: {np.min(confidences):.4f}")
    print(f"   Max: {np.max(confidences):.4f}")
    print(f"   Std: {np.std(confidences):.4f}")
    
    # Low confidence predictions
    low_conf_results = [r for r in results if r['confidence'] < Config.CONFIDENCE_THRESHOLD]
    print(f"\n📊 Low Confidence Predictions (< {Config.CONFIDENCE_THRESHOLD}):")
    print(f"   Count: {len(low_conf_results)}")
    if len(low_conf_results) > 0:
        print(f"   Files:")
        for r in low_conf_results[:10]:  # Show first 10
            print(f"      - {r['filename']} ({r['confidence']:.3f})")
    
    print("\n" + "="*60)
    
    return results


def display_image_with_prediction(image_path: str, 
                                  prediction: str, 
                                  confidence: float,
                                  is_unknown: bool):
    """
    Display image with prediction overlay
    
    Args:
        image_path: Path to image
        prediction: Predicted class
        confidence: Confidence score
        is_unknown: Whether prediction is unknown
    """
    try:
        img = Image.open(image_path)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        
        # Set title color based on prediction
        if is_unknown:
            color = 'orange'
            title = f"❓ UNKNOWN\n(Top: {prediction}, Conf: {confidence:.3f})"
        else:
            color = 'green'
            title = f"✅ {prediction}\n(Confidence: {confidence:.3f})"
        
        plt.title(title, fontsize=16, fontweight='bold', color=color, pad=20)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"⚠️  Could not display image: {e}")


def main():
    """Main testing pipeline with CLI arguments"""
    parser = argparse.ArgumentParser(
        description='Test palm fruit classification model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single image with Keras model
  python test_single_or_batch.py --image test_image.jpg
  
  # Test single image with TFLite model
  python test_single_or_batch.py --image test_image.jpg --tflite
  
  # Test batch of images with Keras model
  python test_single_or_batch.py --batch test_images/
  
  # Test batch with TFLite model
  python test_single_or_batch.py --batch test_images/ --tflite
  
  # Use custom model path
  python test_single_or_batch.py --image test.jpg --model custom_model.keras
        """
    )
    
    parser.add_argument('--image', type=str, help='Path to single test image')
    parser.add_argument('--batch', type=str, help='Path to directory with test images')
    parser.add_argument('--model', type=str, help='Path to model file (optional)')
    parser.add_argument('--tflite', action='store_true', help='Use TFLite model instead of Keras')
    parser.add_argument('--threshold', type=float, help='Confidence threshold for unknown detection')
    
    args = parser.parse_args()
    
    # Update threshold if provided
    if args.threshold:
        Config.CONFIDENCE_THRESHOLD = args.threshold
        print(f"✅ Using custom confidence threshold: {args.threshold}")
    
    # Determine model path
    if args.model:
        model_path = args.model
    else:
        if args.tflite:
            model_path = str(Config.MODELS_DIR / "palm_classifier_float16.tflite")
        else:
            model_path = str(Config.MODELS_DIR / "palm_classifier_final.keras")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Please run train_model.py first!")
        return
    
    # Run appropriate test
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image not found at {args.image}")
            return
        test_single_image(model_path, args.image, use_tflite=args.tflite)
        
    elif args.batch:
        if not os.path.exists(args.batch):
            print(f"❌ Directory not found at {args.batch}")
            return
        test_batch_images(model_path, args.batch, use_tflite=args.tflite)
        
    else:
        print("❌ Please specify either --image or --batch")
        parser.print_help()


if __name__ == "__main__":
    main()