"""
Utility functions for palm fruit classification project
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import keras


# Project Configuration
class Config:
    """Configuration class for the project"""
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATASET_RAW = BASE_DIR / "dataset" / "raw"
    DATASET_PROCESSED = BASE_DIR / "dataset" / "processed"
    MODELS_DIR = BASE_DIR / "models"
    CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
    CONFIGS_DIR = BASE_DIR / "configs"
    
    # Image settings
    IMG_SIZE = (224, 224)
    IMG_CHANNELS = 3
    BATCH_SIZE = 32
    
    # Training settings
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    VALIDATION_SPLIT = 0.1
    
    # Class names
    CLASS_NAMES = ['sawit_mentah', 'sawit_matang']
    NUM_CLASSES = len(CLASS_NAMES)
    
    # Unknown detection settings
    CONFIDENCE_THRESHOLD = 0.70  # Below this = unknown
    ENTROPY_THRESHOLD = 0.6  # High entropy = uncertain
    
    # TFLite quantization options
    QUANTIZATION = 'float16'  # Options: 'float16', 'dynamic', 'int8', None
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        for dir_path in [cls.DATASET_PROCESSED / "train",
                        cls.DATASET_PROCESSED / "validation",
                        cls.MODELS_DIR,
                        cls.CHECKPOINTS_DIR,
                        cls.CONFIGS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def save_config(cls, filepath: str):
        """Save configuration to JSON file"""
        config_dict = {
            'img_size': cls.IMG_SIZE,
            'batch_size': cls.BATCH_SIZE,
            'epochs': cls.EPOCHS,
            'learning_rate': cls.LEARNING_RATE,
            'class_names': cls.CLASS_NAMES,
            'confidence_threshold': cls.CONFIDENCE_THRESHOLD,
            'entropy_threshold': cls.ENTROPY_THRESHOLD,
        }
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
        print(f"✅ Configuration saved to {filepath}")
    
    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return config_dict


def load_trained_model(model_path: str) -> keras.Model:
    """
    Load a trained Keras model
    
    Args:
        model_path: Path to the model file (.keras or .h5)
    
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"📥 Loading model from {model_path}")
    model = keras.models.load_model(model_path)
    print(f"✅ Model loaded successfully")
    return model


def preprocess_image(image_path: str, img_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess a single image for inference
    
    Args:
        image_path: Path to the image
        img_size: Target size (height, width)
    
    Returns:
        Preprocessed image array
    """
    img = keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def calculate_entropy(probabilities: np.ndarray) -> float:
    """
    Calculate Shannon entropy of prediction probabilities
    Higher entropy = more uncertain prediction
    
    Args:
        probabilities: Array of prediction probabilities
    
    Returns:
        Entropy value
    """
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-10
    probabilities = np.clip(probabilities, epsilon, 1.0)
    entropy = -np.sum(probabilities * np.log(probabilities))
    
    # Normalize entropy to [0, 1] range for binary classification
    max_entropy = -np.log(1.0 / len(probabilities))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return normalized_entropy


def is_prediction_uncertain(probabilities: np.ndarray, 
                           confidence_threshold: float = 0.70,
                           entropy_threshold: float = 0.7) -> Tuple[bool, str]:
    """
    Determine if prediction is uncertain (out-of-distribution)
    
    Args:
        probabilities: Prediction probabilities
        confidence_threshold: Minimum confidence for valid prediction
        entropy_threshold: Maximum entropy for valid prediction
    
    Returns:
        Tuple of (is_uncertain, reason)
    """
    max_prob = np.max(probabilities)
    entropy = calculate_entropy(probabilities)
    
    if max_prob < confidence_threshold:
        return True, f"Low confidence ({max_prob:.3f} < {confidence_threshold})"
    
    if entropy > entropy_threshold:
        return True, f"High entropy ({entropy:.3f} > {entropy_threshold})"
    
    return False, "Confident prediction"


def plot_training_history(history: dict, save_path: Optional[str] = None):
    """
    Plot training history (accuracy and loss)
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history['accuracy'], label='Train Accuracy', marker='o')
    axes[0].plot(history['val_accuracy'], label='Val Accuracy', marker='s')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history['loss'], label='Train Loss', marker='o')
    axes[1].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: List[str],
                         save_path: Optional[str] = None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to {save_path}")
    
    plt.show()


def print_model_summary(model: keras.Model):
    """Print detailed model summary"""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    model.summary()
    
    total_params = model.count_params()
    print(f"\n📊 Total Parameters: {total_params:,}")
    print(f"📏 Model Input Shape: {model.input_shape}")
    print(f"📏 Model Output Shape: {model.output_shape}")
    print("="*60 + "\n")


def get_gpu_info():
    """Print GPU information for TensorFlow"""
    import tensorflow as tf
    
    print("\n" + "="*60)
    print("GPU CONFIGURATION")
    print("="*60)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ {len(gpus)} GPU(s) detected:")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
            # Get GPU memory info
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print(f"   Details: {gpu_details}")
            except:
                pass
    else:
        print("⚠️  No GPU detected. Training will use CPU.")
    
    print("="*60 + "\n")


def setup_gpu_memory_growth():
    """Configure GPU memory growth to avoid OOM errors"""
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
        except RuntimeError as e:
            print(f"⚠️  GPU memory growth setup failed: {e}")


if __name__ == "__main__":
    # Test utilities
    print("🔧 Testing utility functions...")
    
    # Create directories
    Config.create_dirs()
    print("✅ Directories created")
    
    # Save config
    config_path = Config.CONFIGS_DIR / "model_config.json"
    Config.save_config(str(config_path))
    
    # Check GPU
    get_gpu_info()
    setup_gpu_memory_growth()
    
    print("\n✅ All utility functions working correctly!")