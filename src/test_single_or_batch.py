"""
./src/test_single_or_batch.py
Testing script for palm fruit classification with YOLO pre-detection
Supports single image and batch testing with both Keras and TFLite models

# Install dependencies dulu
pip install ultralytics

# Test single image dengan YOLO PyTorch + Keras classifier
python test_single_or_batch.py --image test.jpg --yolo models/best.pt

# Test dengan YOLO TFLite + TFLite classifier
python test_single_or_batch.py --image test.jpg --tflite --yolo models/best.tflite --yolo-tflite

# Test batch dengan custom confidence threshold
python test_single_or_batch.py --batch test_images/ --yolo models/best.pt --yolo-conf 0.3

# Tanpa YOLO (seperti sebelumnya)
python test_single_or_batch.py --image test.jpg

"""
import os
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import keras
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Import YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  Ultralytics not installed. YOLO detection will not be available.")
    print("   Install with: pip install ultralytics")

from utils import (
    Config,
    load_trained_model,
    preprocess_image,
    is_prediction_uncertain,
    calculate_entropy
)


class YOLODetector:
    """Wrapper for YOLO detection"""
    
    def __init__(self, model_path: str, use_tflite: bool = False):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model (.pt or .tflite)
            use_tflite: Whether to use TFLite model
        """
        self.use_tflite = use_tflite
        self.model_path = model_path
        
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLO not installed")
        
        # Load YOLO model
        if use_tflite:
            # For TFLite, YOLO will handle it internally
            self.model = YOLO(model_path, task='detect')
        else:
            self.model = YOLO(model_path)
        
        print(f"✅ YOLO model loaded from {model_path}")
    
    def detect(self, image_path: str, conf_threshold: float = 0.25) -> Tuple[bool, Optional[Dict]]:
        """
        Detect palm fruits in image
        
        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold for detection
        
        Returns:
            Tuple of (has_palm, detection_info)
            detection_info contains: {
                'boxes': list of bounding boxes,
                'confidences': list of confidence scores,
                'classes': list of class names,
                'count': number of detections
            }
        """
        try:
            # Run detection
            results = self.model(image_path, conf=conf_threshold, verbose=False)
            
            # Get first result (single image)
            result = results[0]
            
            # Check if any objects detected
            if len(result.boxes) == 0:
                return False, None
            
            # Extract detection info
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
            
            # Get class names
            class_names = [result.names[cls_id] for cls_id in class_ids]
            
            detection_info = {
                'boxes': boxes,
                'confidences': confidences,
                'classes': class_names,
                'class_ids': class_ids,
                'count': len(boxes)
            }
            
            return True, detection_info
            
        except Exception as e:
            print(f"⚠️  YOLO detection error: {e}")
            return False, None
    
    def draw_detections(self, image_path: str, detection_info: Dict) -> Image.Image:
        """
        Draw bounding boxes on image
        
        Args:
            image_path: Path to image
            detection_info: Detection information from detect()
        
        Returns:
            PIL Image with bounding boxes drawn
        """
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Define colors for different classes
        colors = {
            'ripe': 'green',      # Sawit Matang
            'unripe': 'orange',   # Sawit Mentah
            0: 'green',           # Class ID 0
            1: 'orange'           # Class ID 1
        }
        
        # Draw each detection
        for i in range(detection_info['count']):
            box = detection_info['boxes'][i]
            conf = detection_info['confidences'][i]
            class_name = detection_info['classes'][i]
            class_id = detection_info['class_ids'][i]
            
            # Get color
            color = colors.get(class_name.lower(), colors.get(class_id, 'red'))
            
            # Draw rectangle
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            
            # Get text size for background
            bbox = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
            draw.text((x1, y1), label, fill='white', font=font)
        
        return img


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


def test_single_image(model_path: str, 
                     image_path: str, 
                     use_tflite: bool = False,
                     yolo_model_path: Optional[str] = None,
                     yolo_tflite: bool = False,
                     yolo_conf: float = 0.25):
    """
    Test model on a single image with YOLO pre-detection
    
    Args:
        model_path: Path to classifier model file
        image_path: Path to test image
        use_tflite: Whether to use TFLite classifier model
        yolo_model_path: Path to YOLO model (optional)
        yolo_tflite: Whether to use YOLO TFLite model
        yolo_conf: YOLO confidence threshold
    """
    print("\n" + "="*60)
    print("SINGLE IMAGE PREDICTION WITH YOLO PRE-DETECTION")
    print("="*60)
    
    print(f"\n📷 Image: {image_path}")
    print(f"🤖 Classifier Model: {model_path}")
    print(f"🔧 Classifier Format: {'TFLite' if use_tflite else 'Keras'}")
    
    # YOLO Detection Stage
    has_palm = True
    detection_info = None
    
    if yolo_model_path:
        print(f"\n🔍 YOLO Model: {yolo_model_path}")
        print(f"🔧 YOLO Format: {'TFLite' if yolo_tflite else 'PyTorch'}")
        print(f"📊 YOLO Confidence Threshold: {yolo_conf}")
        
        try:
            # Initialize YOLO detector
            yolo_detector = YOLODetector(yolo_model_path, use_tflite=yolo_tflite)
            
            # Detect palm fruits
            print("\n🔍 Running YOLO detection...")
            has_palm, detection_info = yolo_detector.detect(image_path, conf_threshold=yolo_conf)
            
            if not has_palm:
                print("\n" + "="*60)
                print("❌ NO PALM FRUIT DETECTED")
                print("="*60)
                print("   This image does not contain any palm fruits.")
                print("   Classification will be skipped.")
                print("\n" + "="*60)
                
                # Display image
                display_image_no_palm(image_path)
                return
            
            print(f"\n✅ Palm fruits detected: {detection_info['count']} object(s)")
            for i in range(detection_info['count']):
                cls_name = detection_info['classes'][i]
                conf = detection_info['confidences'][i]
                print(f"   - {cls_name}: {conf:.3f}")
                
        except Exception as e:
            print(f"\n⚠️  YOLO detection failed: {e}")
            print("   Proceeding with classification anyway...")
    
    # Classification Stage
    print("\n" + "="*60)
    print("CLASSIFICATION STAGE")
    print("="*60)
    
    # Load classifier model
    if use_tflite:
        print("\n📥 Loading TFLite classifier model...")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        predictions, predicted_class, confidence, is_unknown = predict_with_tflite_model(
            interpreter, image_path
        )
    else:
        print("\n📥 Loading Keras classifier model...")
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
    print(f"   Confidence Threshold: {Config.CONFIDENCE_THRESHOLD}")
    
    print(f"\n📊 All Class Probabilities:")
    for i, class_name in enumerate(Config.CLASS_NAMES):
        prob = predictions[i]
        bar = "█" * int(prob * 50)
        print(f"   {class_name:20s}: {prob:.4f} ({prob*100:.2f}%) {bar}")
    
    print("\n" + "="*60)
    
    # Display image with prediction and YOLO detections
    display_image_with_prediction(
        image_path, 
        predicted_class if not is_unknown else "UNKNOWN",
        confidence,
        is_unknown,
        detection_info
    )


def test_batch_images(model_path: str, 
                     image_dir: str, 
                     use_tflite: bool = False,
                     yolo_model_path: Optional[str] = None,
                     yolo_tflite: bool = False,
                     yolo_conf: float = 0.25):
    """
    Test model on a batch of images with YOLO pre-detection
    
    Args:
        model_path: Path to classifier model file
        image_dir: Directory containing test images
        use_tflite: Whether to use TFLite classifier model
        yolo_model_path: Path to YOLO model (optional)
        yolo_tflite: Whether to use YOLO TFLite model
        yolo_conf: YOLO confidence threshold
    """
    print("\n" + "="*60)
    print("BATCH IMAGE PREDICTION WITH YOLO PRE-DETECTION")
    print("="*60)
    
    print(f"\n📁 Image directory: {image_dir}")
    print(f"🤖 Classifier Model: {model_path}")
    print(f"🔧 Classifier Format: {'TFLite' if use_tflite else 'Keras'}")
    
    # Get all image files
    image_dir = Path(image_dir)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in image_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in valid_extensions]
    
    if len(image_files) == 0:
        print(f"\n❌ No images found in {image_dir}")
        return
    
    print(f"\n📷 Found {len(image_files)} images")
    
    # Initialize YOLO detector if provided
    yolo_detector = None
    if yolo_model_path:
        print(f"\n🔍 YOLO Model: {yolo_model_path}")
        print(f"🔧 YOLO Format: {'TFLite' if yolo_tflite else 'PyTorch'}")
        print(f"📊 YOLO Confidence Threshold: {yolo_conf}")
        
        try:
            yolo_detector = YOLODetector(yolo_model_path, use_tflite=yolo_tflite)
        except Exception as e:
            print(f"\n⚠️  Failed to load YOLO model: {e}")
            print("   Proceeding without YOLO detection...")
    
    # Load classifier model
    if use_tflite:
        print("\n📥 Loading TFLite classifier model...")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    else:
        print("\n📥 Loading Keras classifier model...")
        model = load_trained_model(model_path)
    
    # Process all images
    results = []
    unknown_count = 0
    no_palm_count = 0
    
    print("\n🔄 Processing images...\n")
    
    for img_file in image_files:
        try:
            # YOLO Detection
            has_palm = True
            detection_info = None
            
            if yolo_detector:
                has_palm, detection_info = yolo_detector.detect(str(img_file), conf_threshold=yolo_conf)
                
                if not has_palm:
                    no_palm_count += 1
                    results.append({
                        'filename': img_file.name,
                        'has_palm': False,
                        'yolo_detections': 0,
                        'predicted_class': None,
                        'confidence': None,
                        'is_unknown': None,
                        'entropy': None,
                        'all_predictions': None
                    })
                    print(f"   {img_file.name:30s} -> ❌ NO PALM DETECTED")
                    continue
            
            # Classification
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
                'has_palm': True,
                'yolo_detections': detection_info['count'] if detection_info else None,
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
            
            yolo_info = f" | YOLO: {detection_info['count']} obj" if detection_info else ""
            print(f"   {img_file.name:30s} -> {status:20s} (conf: {confidence:.3f}){yolo_info}")
            
        except Exception as e:
            print(f"   ❌ Error processing {img_file.name}: {e}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("BATCH PREDICTION SUMMARY")
    print("="*60)
    
    print(f"\n📊 Total images processed: {len(results)}")
    
    if yolo_detector:
        print(f"📊 Images with palm detected: {len(results) - no_palm_count} ({(len(results) - no_palm_count)/len(results)*100:.1f}%)")
        print(f"📊 Images without palm: {no_palm_count} ({no_palm_count/len(results)*100:.1f}%)")
    
    classified_results = [r for r in results if r['has_palm']]
    
    if len(classified_results) > 0:
        print(f"📊 Unknown predictions: {unknown_count} ({unknown_count/len(classified_results)*100:.1f}%)")
        
        # Class distribution
        print(f"\n📊 Prediction Distribution (classified images only):")
        for class_name in Config.CLASS_NAMES:
            count = sum(1 for r in classified_results 
                       if r['predicted_class'] == class_name and not r['is_unknown'])
            print(f"   {class_name}: {count}")
        
        # Confidence statistics
        confidences = [r['confidence'] for r in classified_results if r['confidence'] is not None]
        if confidences:
            print(f"\n📊 Confidence Statistics:")
            print(f"   Mean: {np.mean(confidences):.4f}")
            print(f"   Median: {np.median(confidences):.4f}")
            print(f"   Min: {np.min(confidences):.4f}")
            print(f"   Max: {np.max(confidences):.4f}")
            print(f"   Std: {np.std(confidences):.4f}")
        
        # Low confidence predictions
        low_conf_results = [r for r in classified_results 
                           if r['confidence'] and r['confidence'] < Config.CONFIDENCE_THRESHOLD]
        print(f"\n📊 Low Confidence Predictions (< {Config.CONFIDENCE_THRESHOLD}):")
        print(f"   Count: {len(low_conf_results)}")
        if len(low_conf_results) > 0:
            print(f"   Files:")
            for r in low_conf_results[:10]:  # Show first 10
                print(f"      - {r['filename']} ({r['confidence']:.3f})")
    
    print("\n" + "="*60)
    
    return results


def display_image_no_palm(image_path: str):
    """Display image when no palm is detected"""
    try:
        img = Image.open(image_path)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title("❌ NO PALM FRUIT DETECTED", fontsize=16, fontweight='bold', color='red', pad=20)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"⚠️  Could not display image: {e}")


def display_image_with_prediction(image_path: str, 
                                  prediction: str, 
                                  confidence: float,
                                  is_unknown: bool,
                                  detection_info: Optional[Dict] = None):
    """
    Display image with prediction overlay and YOLO detections
    
    Args:
        image_path: Path to image
        prediction: Predicted class
        confidence: Confidence score
        is_unknown: Whether prediction is unknown
        detection_info: YOLO detection information
    """
    try:
        if detection_info:
            # Draw YOLO detections
            yolo_detector = YOLODetector.__new__(YOLODetector)
            img = yolo_detector.draw_detections(image_path, detection_info)
        else:
            img = Image.open(image_path)
        
        plt.figure(figsize=(12, 10))
        plt.imshow(img)
        plt.axis('off')
        
        # Set title color based on prediction
        if is_unknown:
            color = 'orange'
            title = f"❓ UNKNOWN\n(Top: {prediction}, Conf: {confidence:.3f})"
        else:
            color = 'green'
            title = f"✅ {prediction}\n(Confidence: {confidence:.3f})"
        
        if detection_info:
            title += f"\nYOLO Detections: {detection_info['count']}"
        
        plt.title(title, fontsize=16, fontweight='bold', color=color, pad=20)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"⚠️  Could not display image: {e}")


def main():
    """Main testing pipeline with CLI arguments"""
    parser = argparse.ArgumentParser(
        description='Test palm fruit classification model with YOLO pre-detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single image with Keras classifier only
  python test_single_or_batch.py --image test_image.jpg
  
  # Test single image with YOLO + Keras classifier
  python test_single_or_batch.py --image test_image.jpg --yolo models/best.pt
  
  # Test single image with YOLO TFLite + TFLite classifier
  python test_single_or_batch.py --image test_image.jpg --tflite --yolo models/best.tflite --yolo-tflite
  
  # Test batch with YOLO + classifier
  python test_single_or_batch.py --batch test_images/ --yolo models/best.pt
  
  # Use custom models and thresholds
  python test_single_or_batch.py --image test.jpg --model custom.keras --yolo best.pt --yolo-conf 0.3
        """
    )
    
    parser.add_argument('--image', type=str, help='Path to single test image')
    parser.add_argument('--batch', type=str, help='Path to directory with test images')
    parser.add_argument('--model', type=str, help='Path to classifier model file (optional)')
    parser.add_argument('--tflite', action='store_true', help='Use TFLite classifier model')
    parser.add_argument('--threshold', type=float, help='Confidence threshold for unknown detection')
    
    # YOLO arguments
    parser.add_argument('--yolo', type=str, help='Path to YOLO model (.pt or .tflite)')
    parser.add_argument('--yolo-tflite', action='store_true', help='Use YOLO TFLite model')
    parser.add_argument('--yolo-conf', type=float, default=0.25, help='YOLO confidence threshold (default: 0.25)')
    
    args = parser.parse_args()
    
    # Check YOLO availability
    if args.yolo and not YOLO_AVAILABLE:
        print("❌ Ultralytics YOLO not installed!")
        print("   Install with: pip install ultralytics")
        return
    
    # Update threshold if provided
    if args.threshold:
        Config.CONFIDENCE_THRESHOLD = args.threshold
        print(f"✅ Using custom confidence threshold: {args.threshold}")
    
    # Determine classifier model path
    if args.model:
        model_path = args.model
    else:
        if args.tflite:
            model_path = str(Config.MODELS_DIR / "palm_classifier_float16.tflite")
        else:
            model_path = str(Config.MODELS_DIR / "palm_classifier_final.keras")
    
    # Check if classifier model exists
    if not os.path.exists(model_path):
        print(f"❌ Classifier model not found at {model_path}")
        print("   Please run train_model.py first!")
        return
    
    # Check if YOLO model exists
    if args.yolo and not os.path.exists(args.yolo):
        print(f"❌ YOLO model not found at {args.yolo}")
        return
    
    # Run appropriate test
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image not found at {args.image}")
            return
        test_single_image(
            model_path, 
            args.image, 
            use_tflite=args.tflite,
            yolo_model_path=args.yolo,
            yolo_tflite=args.yolo_tflite,
            yolo_conf=args.yolo_conf
        )
        
    elif args.batch:
        if not os.path.exists(args.batch):
            print(f"❌ Directory not found at {args.batch}")
            return
        test_batch_images(
            model_path, 
            args.batch, 
            use_tflite=args.tflite,
            yolo_model_path=args.yolo,
            yolo_tflite=args.yolo_tflite,
            yolo_conf=args.yolo_conf
        )
        
    else:
        print("❌ Please specify either --image or --batch")
        parser.print_help()


if __name__ == "__main__":
    main()