"""
Advanced Feature Visualization
Additional visualization techniques for understanding CNN behavior
python3 src/advance_feature_visualization.py dataset/test/sawit-matang.jpg
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import keras
from keras import models
from PIL import Image
import cv2

from utils import Config


class AdvancedVisualizer:
    """Advanced visualization techniques"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = keras.models.load_model(str(self.model_path))
    
    def visualize_filters(self, layer_name: str, save_path=None, num_filters=64):
        """
        Visualize the actual filter weights
        
        Args:
            layer_name: Name of convolutional layer
            save_path: Path to save visualization
            num_filters: Number of filters to display
        """
        print(f"\n🔍 Visualizing filters from layer: {layer_name}")
        
        # Get layer
        layer = self.model.get_layer(layer_name)
        
        # Get filter weights
        filters, biases = layer.get_weights()
        print(f"   Filter shape: {filters.shape}")
        
        # Normalize filters
        f_min, f_max = filters.min(), filters.max()
        filters_normalized = (filters - f_min) / (f_max - f_min + 1e-8)
        
        # Get number of filters and input channels
        num_total_filters = filters.shape[3]
        num_input_channels = filters.shape[2]
        num_filters = min(num_filters, num_total_filters)
        
        # Calculate grid
        cols = 8
        rows = (num_filters + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows*2))
        fig.suptitle(f'Filter Weights - {layer_name}\nShape: {filters.shape}', 
                    fontsize=14, fontweight='bold')
        
        axes = axes.flatten() if rows > 1 else axes.reshape(-1)
        
        for i in range(num_filters):
            ax = axes[i]
            
            # Get filter for this output channel
            filt = filters_normalized[:, :, :, i]
            
            # If RGB input (3 channels), show as color image
            if num_input_channels == 3:
                ax.imshow(filt)
            else:
                # Show first input channel or average
                filt_vis = np.mean(filt, axis=2) if num_input_channels > 1 else filt[:,:,0]
                ax.imshow(filt_vis, cmap='viridis')
            
            ax.set_title(f'F{i+1}', fontsize=8)
            ax.axis('off')
        
        # Hide unused
        for i in range(num_filters, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   💾 Saved to: {save_path}")
        
        plt.show()
    
    def create_feature_statistics(self, image_path: str, save_path=None):
        """
        Create statistical analysis of feature maps
        
        Args:
            image_path: Path to input image
            save_path: Path to save visualization
        """
        print(f"\n📊 Creating feature statistics for: {image_path}")
        
        # Preprocess image
        img = keras.preprocessing.image.load_img(image_path, target_size=Config.IMG_SIZE)
        img_array = keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get conv layers
        conv_layers = [layer for layer in self.model.layers 
                      if 'conv' in layer.name.lower()]
        
        # Extract features
        stats = []
        for layer in conv_layers:
            extractor = models.Model(inputs=self.model.input, outputs=layer.output)
            features = extractor.predict(img_array, verbose=0)[0]
            
            stats.append({
                'layer': layer.name,
                'shape': features.shape,
                'mean': np.mean(features),
                'std': np.std(features),
                'min': np.min(features),
                'max': np.max(features),
                'sparsity': np.mean(features == 0) * 100  # % of zero activations
            })
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Map Statistics Across Layers', 
                    fontsize=16, fontweight='bold')
        
        layers = [s['layer'] for s in stats]
        
        # Mean activation
        ax = axes[0, 0]
        means = [s['mean'] for s in stats]
        ax.plot(means, marker='o', linewidth=2, markersize=8)
        ax.set_title('Mean Activation', fontsize=12, fontweight='bold')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean Value')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Standard deviation
        ax = axes[0, 1]
        stds = [s['std'] for s in stats]
        ax.plot(stds, marker='s', linewidth=2, markersize=8, color='orange')
        ax.set_title('Standard Deviation', fontsize=12, fontweight='bold')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Std Value')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Min/Max range
        ax = axes[1, 0]
        mins = [s['min'] for s in stats]
        maxs = [s['max'] for s in stats]
        x = range(len(layers))
        ax.fill_between(x, mins, maxs, alpha=0.3)
        ax.plot(mins, marker='v', label='Min', linewidth=2)
        ax.plot(maxs, marker='^', label='Max', linewidth=2)
        ax.set_title('Activation Range', fontsize=12, fontweight='bold')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Value')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Sparsity
        ax = axes[1, 1]
        sparsity = [s['sparsity'] for s in stats]
        ax.bar(range(len(layers)), sparsity, alpha=0.7, color='green')
        ax.set_title('Sparsity (% Zero Activations)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Sparsity (%)')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   💾 Saved to: {save_path}")
        
        plt.show()
    
    def create_layer_depth_visualization(self, image_path: str, save_path=None):
        """
        Show how features evolve through network depth
        
        Args:
            image_path: Path to input image
            save_path: Path to save visualization
        """
        print(f"\n🌊 Creating depth visualization for: {image_path}")
        
        # Preprocess
        img = keras.preprocessing.image.load_img(image_path, target_size=Config.IMG_SIZE)
        img_array = keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get conv layers
        conv_layers = [layer for layer in self.model.layers 
                      if 'conv' in layer.name.lower()][:8]  # First 8 layers
        
        # Create figure
        num_layers = len(conv_layers)
        fig, axes = plt.subplots(1, num_layers + 1, figsize=(20, 3))
        
        # Show original
        axes[0].imshow(img)
        axes[0].set_title('Input', fontsize=10, fontweight='bold')
        axes[0].axis('off')
        
        # Extract and show features
        for idx, layer in enumerate(conv_layers, 1):
            extractor = models.Model(inputs=self.model.input, outputs=layer.output)
            features = extractor.predict(img_array, verbose=0)[0]
            
            # Average across channels
            avg_features = np.mean(features, axis=-1)
            
            # Normalize
            avg_features = (avg_features - avg_features.min()) / \
                          (avg_features.max() - avg_features.min() + 1e-8)
            
            # Display
            axes[idx].imshow(avg_features, cmap='viridis')
            axes[idx].set_title(f'L{idx}\n{features.shape[2]}ch', 
                              fontsize=9, fontweight='bold')
            axes[idx].axis('off')
        
        plt.suptitle('Feature Evolution Through Network Depth', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   💾 Saved to: {save_path}")
        
        plt.show()


def main():
    """Main execution for advanced visualizations"""
    print("="*60)
    print("🎨 ADVANCED FEATURE VISUALIZATION")
    print("="*60)
    
    # Configuration
    model_path = Config.MODELS_DIR / "palm_classifier_final.keras"
    output_dir = Config.MODELS_DIR / "advanced_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check model
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    # Initialize
    visualizer = AdvancedVisualizer(str(model_path))
    
    # Get first conv layer
    conv_layers = [layer.name for layer in visualizer.model.layers 
                  if 'conv' in layer.name.lower()]
    
    if not conv_layers:
        print("❌ No convolutional layers found")
        return
    
    print(f"\n📊 Found {len(conv_layers)} convolutional layers")
    
    # 1. Visualize first layer filters
    print("\n" + "="*60)
    print("1. FILTER WEIGHTS VISUALIZATION")
    print("="*60)
    first_conv = conv_layers[0]
    filter_save_path = output_dir / f"{first_conv}_filters.png"
    visualizer.visualize_filters(first_conv, save_path=str(filter_save_path))
    
    # Find test image
    val_dir = Config.DATASET_PROCESSED / "validation"
    test_image = None
    
    if val_dir.exists():
        for class_dir in val_dir.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob("*.jpg"))
                if images:
                    test_image = images[0]
                    break
    
    if test_image:
        # 2. Feature statistics
        print("\n" + "="*60)
        print("2. FEATURE STATISTICS")
        print("="*60)
        stats_save_path = output_dir / f"feature_statistics.png"
        visualizer.create_feature_statistics(
            str(test_image), 
            save_path=str(stats_save_path)
        )
        
        # 3. Depth visualization
        print("\n" + "="*60)
        print("3. NETWORK DEPTH VISUALIZATION")
        print("="*60)
        depth_save_path = output_dir / f"depth_visualization.png"
        visualizer.create_layer_depth_visualization(
            str(test_image),
            save_path=str(depth_save_path)
        )
    
    print("\n" + "="*60)
    print("✅ ADVANCED VISUALIZATIONS COMPLETED!")
    print("="*60)
    print(f"📁 Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()