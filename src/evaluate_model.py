"""
Model evaluation script for palm fruit classification
Evaluates model performance on validation dataset
"""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import keras

from utils import (
    Config,
    load_trained_model,
    plot_confusion_matrix
)
from data_processing import create_tf_datasets


def evaluate_model_on_dataset(model, dataset, class_names):
    """
    Evaluate model on a dataset
    
    Args:
        model: Trained Keras model
        dataset: TensorFlow dataset
        class_names: List of class names
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n🔄 Evaluating model...")
    
    # Evaluate model
    results = model.evaluate(dataset, verbose=1)
    
    # Get predictions
    print("\n🔄 Generating predictions...")
    y_true = []
    y_pred = []
    y_pred_probs = []
    
    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        y_pred_probs.extend(predictions)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_probs = np.array(y_pred_probs)
    
    # Calculate metrics
    metrics = {
        'loss': results[0],
        'accuracy': results[1],
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_probs': y_pred_probs,
        'class_names': class_names
    }
    
    return metrics


def print_evaluation_results(metrics):
    """
    Print detailed evaluation results
    
    Args:
        metrics: Dictionary with evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Loss: {metrics['loss']:.4f}")
    print(f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(metrics['y_true'], metrics['y_pred'])
    print(f"\n📊 Confusion Matrix:")
    print(cm)
    
    # Classification Report
    print(f"\n📊 Classification Report:")
    report = classification_report(
        metrics['y_true'], 
        metrics['y_pred'],
        target_names=metrics['class_names'],
        digits=4
    )
    print(report)
    
    # Per-class accuracy
    print(f"\n📊 Per-Class Accuracy:")
    for i, class_name in enumerate(metrics['class_names']):
        class_mask = metrics['y_true'] == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(metrics['y_pred'][class_mask] == i)
            print(f"   {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    # Confidence statistics
    max_probs = np.max(metrics['y_pred_probs'], axis=1)
    print(f"\n📊 Prediction Confidence Statistics:")
    print(f"   Mean confidence: {np.mean(max_probs):.4f}")
    print(f"   Median confidence: {np.median(max_probs):.4f}")
    print(f"   Min confidence: {np.min(max_probs):.4f}")
    print(f"   Max confidence: {np.max(max_probs):.4f}")
    print(f"   Std confidence: {np.std(max_probs):.4f}")
    
    # Low confidence predictions
    low_conf_threshold = Config.CONFIDENCE_THRESHOLD
    low_conf_mask = max_probs < low_conf_threshold
    low_conf_count = np.sum(low_conf_mask)
    low_conf_pct = (low_conf_count / len(max_probs)) * 100
    
    print(f"\n⚠️  Low Confidence Predictions (< {low_conf_threshold}):")
    print(f"   Count: {low_conf_count}/{len(max_probs)} ({low_conf_pct:.2f}%)")
    
    if low_conf_count > 0:
        print(f"   These would be flagged as 'unknown' in production")
        # Check accuracy of low confidence predictions
        low_conf_correct = np.sum(
            metrics['y_pred'][low_conf_mask] == metrics['y_true'][low_conf_mask]
        )
        low_conf_acc = low_conf_correct / low_conf_count if low_conf_count > 0 else 0
        print(f"   Accuracy of low-conf predictions: {low_conf_acc:.4f} ({low_conf_acc*100:.2f}%)")
    
    print("\n" + "="*60)


def analyze_errors(metrics):
    """
    Analyze misclassified samples
    
    Args:
        metrics: Dictionary with evaluation metrics
    """
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    
    # Find misclassified samples
    errors_mask = metrics['y_pred'] != metrics['y_true']
    num_errors = np.sum(errors_mask)
    
    print(f"\n❌ Total misclassifications: {num_errors}/{len(metrics['y_true'])}")
    
    if num_errors > 0:
        # Error distribution
        print(f"\n📊 Misclassification breakdown:")
        for true_idx, true_class in enumerate(metrics['class_names']):
            for pred_idx, pred_class in enumerate(metrics['class_names']):
                if true_idx != pred_idx:
                    count = np.sum(
                        (metrics['y_true'] == true_idx) & 
                        (metrics['y_pred'] == pred_idx)
                    )
                    if count > 0:
                        print(f"   {true_class} → {pred_class}: {count}")
        
        # Confidence of errors
        error_confidences = np.max(metrics['y_pred_probs'][errors_mask], axis=1)
        print(f"\n📊 Confidence of misclassified predictions:")
        print(f"   Mean: {np.mean(error_confidences):.4f}")
        print(f"   Median: {np.median(error_confidences):.4f}")
        print(f"   Min: {np.min(error_confidences):.4f}")
        print(f"   Max: {np.max(error_confidences):.4f}")
    
    print("\n" + "="*60)


def main():
    """Main evaluation pipeline"""
    print("🚀 Starting model evaluation...\n")
    
    # Check if model exists
    model_path = Config.MODELS_DIR / "palm_classifier_final.keras"
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("   Please run train_model.py first!")
        return
    
    # Load model
    model = load_trained_model(str(model_path))
    
    # Load validation dataset
    print("\n📊 Loading validation dataset...")
    _, val_ds, class_names = create_tf_datasets(
        batch_size=Config.BATCH_SIZE
    )
    
    # Evaluate model
    metrics = evaluate_model_on_dataset(model, val_ds, class_names)
    
    # Print results
    print_evaluation_results(metrics)
    
    # Analyze errors
    analyze_errors(metrics)
    
    # Plot confusion matrix
    cm = confusion_matrix(metrics['y_true'], metrics['y_pred'])
    cm_path = Config.MODELS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, save_path=str(cm_path))
    
    print(f"\n✅ Confusion matrix saved to: {cm_path}")
    print("\n🎉 Evaluation completed!")


if __name__ == "__main__":
    main()