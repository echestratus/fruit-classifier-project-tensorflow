"""
Data preprocessing script for palm fruit classification
Processes raw dataset and splits into train/validation sets
"""
import os
import shutil
import numpy as np
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
from utils import Config


def clear_processed_directory():
    """Clear processed directory before preprocessing"""
    if Config.DATASET_PROCESSED.exists():
        print("🗑️  Clearing existing processed directory...")
        shutil.rmtree(Config.DATASET_PROCESSED)
    Config.DATASET_PROCESSED.mkdir(parents=True, exist_ok=True)


def get_image_files(class_dir: Path) -> list:
    """
    Get all image files from a directory
    
    Args:
        class_dir: Path to class directory
    
    Returns:
        List of image file paths
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = []
    
    for file_path in class_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            image_files.append(file_path)
    
    return image_files


def preprocess_and_save_image(src_path: Path, 
                              dst_path: Path, 
                              img_size: Tuple[int, int] = (224, 224)):
    """
    Preprocess single image and save to destination
    
    Args:
        src_path: Source image path
        dst_path: Destination image path
        img_size: Target image size
    """
    try:
        # Load image
        img = keras.preprocessing.image.load_img(src_path, target_size=img_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Convert back to uint8 for saving (0-255 range)
        img_array = (img_array * 255).astype(np.uint8)
        
        # Save processed image
        keras.preprocessing.image.save_img(dst_path, img_array)
        
    except Exception as e:
        print(f"⚠️  Error processing {src_path.name}: {e}")


def split_and_process_dataset(validation_split: float = 0.1, 
                              random_state: int = 42):
    """
    Split raw dataset into train/validation and preprocess images
    
    Args:
        validation_split: Proportion of data for validation (0.0 to 1.0)
        random_state: Random seed for reproducibility
    """
    print("\n" + "="*60)
    print("DATASET PREPROCESSING")
    print("="*60)
    
    # Clear processed directory
    clear_processed_directory()
    
    # Check if raw dataset exists
    if not Config.DATASET_RAW.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {Config.DATASET_RAW}\n"
            f"Please create the directory and add your images."
        )
    
    # Get class directories
    class_dirs = [d for d in Config.DATASET_RAW.iterdir() if d.is_dir()]
    
    if len(class_dirs) == 0:
        raise ValueError("No class directories found in raw dataset!")
    
    print(f"\n📁 Found {len(class_dirs)} classes:")
    for class_dir in class_dirs:
        print(f"   - {class_dir.name}")
    
    # Process each class
    total_train_images = 0
    total_val_images = 0
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\n🔄 Processing class: {class_name}")
        
        # Get all image files
        image_files = get_image_files(class_dir)
        print(f"   Found {len(image_files)} images")
        
        if len(image_files) == 0:
            print(f"   ⚠️  No images found in {class_name}, skipping...")
            continue
        
        # Split into train and validation
        train_files, val_files = train_test_split(
            image_files,
            test_size=validation_split,
            random_state=random_state,
            shuffle=True
        )
        
        print(f"   Split: {len(train_files)} train, {len(val_files)} validation")
        
        # Create class directories in processed dataset
        train_class_dir = Config.DATASET_PROCESSED / "train" / class_name
        val_class_dir = Config.DATASET_PROCESSED / "validation" / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Process and save training images
        print(f"   Processing training images...")
        for img_file in tqdm(train_files, desc=f"   {class_name} (train)"):
            dst_path = train_class_dir / img_file.name
            preprocess_and_save_image(img_file, dst_path, Config.IMG_SIZE)
        
        # Process and save validation images
        print(f"   Processing validation images...")
        for img_file in tqdm(val_files, desc=f"   {class_name} (val)"):
            dst_path = val_class_dir / img_file.name
            preprocess_and_save_image(img_file, dst_path, Config.IMG_SIZE)
        
        total_train_images += len(train_files)
        total_val_images += len(val_files)
    
    # Summary
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"✅ Total training images: {total_train_images}")
    print(f"✅ Total validation images: {total_val_images}")
    print(f"✅ Total images processed: {total_train_images + total_val_images}")
    print(f"✅ Images saved to: {Config.DATASET_PROCESSED}")
    print("="*60 + "\n")


def verify_processed_dataset():
    """Verify the processed dataset structure"""
    print("\n" + "="*60)
    print("DATASET VERIFICATION")
    print("="*60)
    
    for split in ['train', 'validation']:
        split_dir = Config.DATASET_PROCESSED / split
        if not split_dir.exists():
            print(f"⚠️  {split} directory not found!")
            continue
        
        print(f"\n📂 {split.upper()} SET:")
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        
        for class_dir in sorted(class_dirs):
            image_files = get_image_files(class_dir)
            print(f"   {class_dir.name}: {len(image_files)} images")
    
    print("\n" + "="*60 + "\n")


def create_tf_datasets(batch_size: int = 32,
                       shuffle: bool = True,
                       seed: int = 42):
    """
    Create TensorFlow datasets from processed images
    
    Args:
        batch_size: Batch size for training
        shuffle: Whether to shuffle training data
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataset, val_dataset, class_names)
    """
    print("\n🔄 Creating TensorFlow datasets...")
    
    # Create training dataset
    train_ds = keras.preprocessing.image_dataset_from_directory(
        Config.DATASET_PROCESSED / "train",
        image_size=Config.IMG_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        label_mode='categorical'  # One-hot encoding
    )
    
    # Create validation dataset
    val_ds = keras.preprocessing.image_dataset_from_directory(
        Config.DATASET_PROCESSED / "validation",
        image_size=Config.IMG_SIZE,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        label_mode='categorical'
    )
    
    # Get class names
    class_names = train_ds.class_names
    print(f"✅ Class names: {class_names}")
    
    # Normalize images to [0, 1] range (images are already normalized during preprocessing)
    # But we apply it again for consistency
    normalization_layer = keras.layers.Rescaling(1./255)
    
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Optimize performance with prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    
    print(f"✅ Training batches: {len(train_ds)}")
    print(f"✅ Validation batches: {len(val_ds)}")
    
    return train_ds, val_ds, class_names


def main():
    """Main preprocessing pipeline"""
    print("🚀 Starting data preprocessing pipeline...\n")
    
    # Create necessary directories
    Config.create_dirs()
    
    # Split and process dataset
    split_and_process_dataset(
        validation_split=Config.VALIDATION_SPLIT,
        random_state=42
    )
    
    # Verify processed dataset
    verify_processed_dataset()
    
    # Test TensorFlow dataset creation
    try:
        train_ds, val_ds, class_names = create_tf_datasets(
            batch_size=Config.BATCH_SIZE
        )
        print("\n✅ TensorFlow datasets created successfully!")
        print(f"✅ Ready for training with {len(class_names)} classes")
    except Exception as e:
        print(f"\n⚠️  Error creating TensorFlow datasets: {e}")
        print("   Please check if processed dataset exists and has correct structure")
    
    print("\n🎉 Data preprocessing completed!")


if __name__ == "__main__":
    main()