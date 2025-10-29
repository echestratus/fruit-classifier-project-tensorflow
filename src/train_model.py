"""
Training script for palm fruit classification model
Uses transfer learning with MobileNetV2 for efficient deployment on Raspberry Pi
"""
import os
import json
from datetime import datetime
import numpy as np
import keras
from keras import layers, models, optimizers, callbacks

from utils import (
    Config, 
    setup_gpu_memory_growth, 
    get_gpu_info,
    plot_training_history,
    print_model_summary
)
from data_processing import create_tf_datasets


def create_augmentation_layer():
    """
    Create data augmentation layer for training
    
    Returns:
        Sequential model with augmentation layers
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ], name="augmentation")


def build_transfer_learning_model(
    input_shape: tuple = (224, 224, 3),
    num_classes: int = 2,
    base_model_name: str = "MobileNetV2",
    trainable_base_layers: int = 20
):
    """
    Build transfer learning model with pre-trained base
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        base_model_name: Name of pre-trained model
        trainable_base_layers: Number of base layers to fine-tune (from the end)
    
    Returns:
        Compiled Keras model
    """
    print(f"\n🏗️  Building model with {base_model_name} backbone...")
    
    # Load pre-trained base model
    if base_model_name == "MobileNetV2":
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    elif base_model_name == "EfficientNetB0":
        base_model = keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    elif base_model_name == "MobileNetV3Small":
        base_model = keras.applications.MobileNetV3Small(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Create augmentation layer
    augmentation = create_augmentation_layer()
    
    # Build model architecture
    inputs = keras.Input(shape=input_shape)
    
    # Data augmentation (only applied during training)
    x = augmentation(inputs)
    
    # Pre-trained base model
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name=f'{base_model_name}_palm_classifier')
    
    print(f"✅ Model built with {base_model.name} backbone")
    print(f"   Total base layers: {len(base_model.layers)}")
    
    return model, base_model


def compile_model(model, learning_rate: float = 0.0001):
    """
    Compile model with optimizer and loss function
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
    """
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_accuracy')]
    )
    print(f"✅ Model compiled with Adam optimizer (lr={learning_rate})")


def create_callbacks(checkpoint_dir: str, model_name: str):
    """
    Create training callbacks
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        model_name: Name prefix for saved models
    
    Returns:
        List of Keras callbacks
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Model checkpoint - save best model
    checkpoint_path = os.path.join(
        checkpoint_dir, 
        f"{model_name}_best.keras"
    )
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Early stopping
    early_stop_cb = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate on plateau
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # TensorBoard (optional)
    log_dir = os.path.join(checkpoint_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_cb = callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    
    return [checkpoint_cb, early_stop_cb, reduce_lr_cb, tensorboard_cb]


def train_phase_1(model, train_ds, val_ds, epochs: int = 20):
    """
    Phase 1: Train only the classification head (base frozen)
    
    Args:
        model: Keras model
        train_ds: Training dataset
        val_ds: Validation dataset
        epochs: Number of epochs
    
    Returns:
        Training history
    """
    print("\n" + "="*60)
    print("TRAINING PHASE 1: Fine-tuning classification head")
    print("="*60)
    
    # Compile with higher learning rate for head training
    compile_model(model, learning_rate=0.001)
    
    # Create callbacks
    training_callbacks = create_callbacks(
        str(Config.CHECKPOINTS_DIR),
        "palm_classifier_phase1"
    )
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=training_callbacks,
        verbose=1
    )
    
    print("\n✅ Phase 1 training completed!")
    return history


def train_phase_2(model, base_model, train_ds, val_ds, 
                  epochs: int = 30, 
                  trainable_layers: int = 20):
    """
    Phase 2: Fine-tune some layers of the base model
    
    Args:
        model: Keras model
        base_model: Base model to unfreeze
        train_ds: Training dataset
        val_ds: Validation dataset
        epochs: Number of epochs
        trainable_layers: Number of layers to unfreeze from the end
    
    Returns:
        Training history
    """
    print("\n" + "="*60)
    print("TRAINING PHASE 2: Fine-tuning base model layers")
    print("="*60)
    
    # Unfreeze the last N layers of base model
    base_model.trainable = True
    total_layers = len(base_model.layers)
    
    # Freeze all layers except the last trainable_layers
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    
    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"🔓 Unfrozen {trainable_count}/{total_layers} base model layers")
    
    # Compile with lower learning rate for fine-tuning
    compile_model(model, learning_rate=Config.LEARNING_RATE)
    
    # Create callbacks
    training_callbacks = create_callbacks(
        str(Config.CHECKPOINTS_DIR),
        "palm_classifier_phase2"
    )
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=training_callbacks,
        verbose=1
    )
    
    print("\n✅ Phase 2 training completed!")
    return history


def save_final_model(model, save_path: str):
    """
    Save the final trained model
    
    Args:
        model: Trained Keras model
        save_path: Path to save the model
    """
    model.save(save_path)
    print(f"\n💾 Final model saved to: {save_path}")
    
    # Get model size
    model_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"📊 Model size: {model_size_mb:.2f} MB")


def save_training_history(history, save_path: str):
    """
    Save training history to JSON file
    
    Args:
        history: Training history object
        save_path: Path to save history
    """
    history_dict = history.history
    
    # Convert numpy arrays to lists for JSON serialization
    for key in history_dict:
        if isinstance(history_dict[key], np.ndarray):
            history_dict[key] = history_dict[key].tolist()
    
    with open(save_path, 'w') as f:
        json.dump(history_dict, f, indent=4)
    
    print(f"💾 Training history saved to: {save_path}")


def main():
    """Main training pipeline"""
    print("🚀 Starting palm fruit classification training...\n")
    
    # Setup GPU
    setup_gpu_memory_growth()
    get_gpu_info()
    
    # Create directories
    Config.create_dirs()
    
    # Load datasets
    print("\n📊 Loading datasets...")
    train_ds, val_ds, class_names = create_tf_datasets(
        batch_size=Config.BATCH_SIZE
    )
    
    print(f"✅ Classes: {class_names}")
    print(f"✅ Number of classes: {len(class_names)}")
    
    # Build model
    model, base_model = build_transfer_learning_model(
        input_shape=(*Config.IMG_SIZE, Config.IMG_CHANNELS),
        num_classes=Config.NUM_CLASSES,
        base_model_name="MobileNetV2",  # Best for Raspberry Pi
        trainable_base_layers=20
    )
    
    # Print model summary
    print_model_summary(model)
    
    # Training Phase 1: Train classification head only
    history_phase1 = train_phase_1(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=20
    )
    
    # Training Phase 2: Fine-tune base model
    history_phase2 = train_phase_2(
        model=model,
        base_model=base_model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=Config.EPOCHS,
        trainable_layers=20
    )
    
    # Combine histories
    combined_history = {
        'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
        'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
        'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
        'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss']
    }
    
    # Save final model
    final_model_path = Config.MODELS_DIR / "palm_classifier_final.keras"
    save_final_model(model, str(final_model_path))
    
    # Save training history
    history_path = Config.MODELS_DIR / "training_history.json"
    save_training_history(history_phase2, str(history_path))
    
    # Plot training history
    plot_path = Config.MODELS_DIR / "training_history.png"
    plot_training_history(combined_history, save_path=str(plot_path))
    
    # Save model configuration
    config_path = Config.CONFIGS_DIR / "model_config.json"
    Config.save_config(str(config_path))
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY! 🎉")
    print("="*60)
    print(f"📁 Model saved at: {final_model_path}")
    print(f"📁 History saved at: {history_path}")
    print(f"📁 Config saved at: {config_path}")
    print("="*60)
    
    print("\n📋 Next steps:")
    print("   1. Run evaluate_model.py to evaluate model performance")
    print("   2. Run convert_model.py to convert to TensorFlow Lite")
    print("   3. Run test_single_or_batch.py to test predictions")


if __name__ == "__main__":
    main()