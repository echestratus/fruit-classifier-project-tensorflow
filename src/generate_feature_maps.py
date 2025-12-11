"""
Feature Map Visualization Script - FIXED v2
Generate and visualize feature maps from different layers of the palm fruit classification model
python3 src/generate_feature_maps.py dataset/test/sawit-matang.jpg
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import keras
from keras import models
import cv2
from PIL import Image
from utils import Config

class FeatureMapVisualizer:
    """Generate and visualize feature maps from CNN layers"""
    
    def __init__(self, model_path: str):
        """
        Initialize visualizer
        
        Args:
            model_path: Path to the trained Keras model (.keras or .h5)
        """
        self.model_path = Path(model_path)
        self.model = None
        self.mobilenet_model = None
        self.augmentation_layer = None
        self.feature_extractors = {}
        
        # Check if model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        # Load model
        print(f"📥 Loading model from {self.model_path}")
        self.model = keras.models.load_model(str(self.model_path))
        print(f"✅ Model loaded successfully")
        print(f"   Total layers: {len(self.model.layers)}")
        
        # Extract MobileNetV2 and augmentation layer
        for layer in self.model.layers:
            if 'mobilenet' in layer.name.lower():
                self.mobilenet_model = layer
                print(f"   Found base model: {layer.name}")
            if 'augmentation' in layer.name.lower():
                self.augmentation_layer = layer
                print(f"   Found augmentation: {layer.name}")
    
    def get_conv_layer_names(self):
        """
        Get names of all convolutional layers from MobileNetV2
        
        Returns:
            List of tuples (layer_name, layer_object)
        """
        if self.mobilenet_model is None:
            print("❌ MobileNetV2 model not found!")
            return []
        
        conv_layers = []
        
        def extract_conv_layers(model, prefix=""):
            """Recursively extract conv layers from nested models"""
            for layer in model.layers:
                layer_name = f"{prefix}{layer.name}" if prefix else layer.name
                
                # Check if it's a conv layer or relevant layer
                if isinstance(layer, keras.layers.Conv2D) or \
                   isinstance(layer, keras.layers.DepthwiseConv2D) or \
                   isinstance(layer, keras.layers.BatchNormalization) or \
                   'conv' in layer.name.lower() or \
                   'depthwise' in layer.name.lower():
                    conv_layers.append((layer_name, layer))
                
                # If it's a nested model, recursively extract
                if isinstance(layer, keras.Model) and hasattr(layer, 'layers'):
                    extract_conv_layers(layer, prefix=f"{layer_name}/")
        
        # Extract from MobileNetV2 model
        extract_conv_layers(self.mobilenet_model, prefix=f"{self.mobilenet_model.name}/")
        
        return conv_layers
    
    def get_layer_outputs(self, layer_list=None, max_layers=20):
        """
        Create feature extractor models for specified layers
        
        Args:
            layer_list: List of tuples (layer_name, layer_object) to extract features from
                       If None, extract from all conv layers
            max_layers: Maximum number of layers to extract (to avoid memory issues)
        """
        if layer_list is None:
            layer_list = self.get_conv_layer_names()
        
        if not layer_list:
            print("❌ No convolutional layers found in the model!")
            return
        
        # Select evenly distributed layers if too many
        if len(layer_list) > max_layers:
            print(f"   Selecting {max_layers} evenly distributed layers from {len(layer_list)} total")
            step = len(layer_list) // max_layers
            layer_list = [layer_list[i] for i in range(0, len(layer_list), step)][:max_layers]
        
        print(f"\n🔍 Creating feature extractors for {len(layer_list)} layers:")
        
        if self.mobilenet_model is None:
            print("❌ MobileNetV2 model not found!")
            return
        
        success_count = 0
        for layer_name, layer_obj in layer_list:
            try:
                # Create feature extractor for MobileNetV2 layer
                mobilenet_extractor = models.Model(
                    inputs=self.mobilenet_model.input,
                    outputs=layer_obj.output
                )
                
                # Now create full pipeline: input -> augmentation -> mobilenet extractor
                if self.augmentation_layer:
                    # Create functional model with augmentation
                    inputs = keras.Input(shape=Config.IMG_SIZE + (3,))
                    x = self.augmentation_layer(inputs, training=False)
                    outputs = mobilenet_extractor(x, training=False)
                    feature_extractor = models.Model(inputs=inputs, outputs=outputs)
                else:
                    # No augmentation, use mobilenet directly
                    feature_extractor = mobilenet_extractor
                
                self.feature_extractors[layer_name] = feature_extractor
                print(f"   ✅ {layer_name}")
                success_count += 1
                
            except Exception as e:
                print(f"   ❌ Failed: {layer_name} - {str(e)[:50]}")
        
        if success_count == 0:
            print("\n❌ No feature extractors created successfully!")
        else:
            print(f"\n✅ Created {success_count} feature extractors")
    
    def preprocess_image(self, image_path: str):
        """
        Preprocess image for model input
        
        Args:
            image_path: Path to image
        
        Returns:
            Preprocessed image array and original image
        """
        img = keras.preprocessing.image.load_img(
            image_path, 
            target_size=Config.IMG_SIZE
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
    
    def extract_features(self, image_path: str):
        """
        Extract feature maps from all registered layers
        
        Args:
            image_path: Path to input image
        
        Returns:
            Dictionary of {layer_name: feature_maps}
        """
        print(f"\n🔄 Extracting features from: {image_path}")
        
        # Preprocess image
        img_array, original_img = self.preprocess_image(image_path)
        
        # Extract features from each layer
        feature_maps = {}
        for i, (layer_name, extractor) in enumerate(self.feature_extractors.items(), 1):
            try:
                features = extractor.predict(img_array, verbose=0)
                feature_maps[layer_name] = features[0]  # Remove batch dimension
                print(f"   [{i}/{len(self.feature_extractors)}] {layer_name}: {features[0].shape}")
            except Exception as e:
                print(f"   ❌ Failed to extract from {layer_name}: {e}")
        
        return feature_maps, original_img
    
    def visualize_feature_map(self, feature_map, layer_name, num_filters=16, 
                             save_path=None, figsize=(20, 12)):
        """
        Visualize feature maps from a single layer
        
        Args:
            feature_map: Feature map array (H, W, C)
            layer_name: Name of the layer
            num_filters: Number of filters to display
            save_path: Path to save the figure
            figsize: Figure size
        """
        # Get dimensions
        height, width, num_channels = feature_map.shape
        num_filters = min(num_filters, num_channels)
        
        # Calculate grid size
        cols = 4
        rows = (num_filters + cols - 1) // cols
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle(f'Feature Maps - {layer_name}\nShape: {feature_map.shape}', 
                    fontsize=16, fontweight='bold')
        
        # Flatten axes for easier iteration
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        axes = axes.flatten()
        
        # Plot each filter
        for i in range(num_filters):
            ax = axes[i]
            
            # Get feature map for this filter
            fmap = feature_map[:, :, i]
            
            # Normalize
            if fmap.size == 0 or np.all(fmap == 0):
                fmap_normalized = np.zeros_like(fmap)
            else:
                fmap_min = fmap.min()
                fmap_max = fmap.max()
                if fmap_max - fmap_min < 1e-8:
                    fmap_normalized = np.zeros_like(fmap)
                else:
                    fmap_normalized = (fmap - fmap_min) / (fmap_max - fmap_min)
            
            # Display
            im = ax.imshow(fmap_normalized, cmap='viridis', aspect='auto')
            ax.set_title(f'Filter {i+1}', fontsize=10)
            ax.axis('off')
            
            # Add colorbar
            try:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            except:
                pass
        
        # Hide unused subplots
        for i in range(num_filters, len(axes)):
            axes[i].axis('off')
        
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Save
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"   💾 Saved to: {save_path}")
            except Exception as e:
                plt.savefig(save_path, dpi=300)
                print(f"   💾 Saved to: {save_path}")
        
        plt.close(fig)
    
    def visualize_all_layers(self, image_path: str, output_dir: str, 
                            num_filters_per_layer=16):
        """
        Generate feature map visualizations for all layers
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save visualizations
            num_filters_per_layer: Number of filters to show per layer
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("FEATURE MAP GENERATION")
        print("="*60)
        
        # Extract features
        feature_maps, original_img = self.extract_features(image_path)
        
        # Save original image
        img_name = Path(image_path).stem
        original_save_path = output_dir / f"{img_name}_original.png"
        original_img.save(original_save_path)
        print(f"\n💾 Original image saved to: {original_save_path}")
        
        if not feature_maps:
            print("❌ No feature maps extracted!")
            return
        
        # Visualize each layer
        print(f"\n🎨 Generating visualizations for {len(feature_maps)} layers...")
        for layer_name, fmap in feature_maps.items():
            clean_name = layer_name.replace('/', '_').replace(' ', '_')
            save_path = output_dir / f"{img_name}_{clean_name}.png"
            
            print(f"\n📊 Processing {layer_name}...")
            print(f"   Shape: {fmap.shape}")
            
            self.visualize_feature_map(
                fmap, 
                layer_name, 
                num_filters=num_filters_per_layer,
                save_path=save_path
            )
        
        print("\n" + "="*60)
        print("✅ Feature map generation completed!")
        print(f"📁 All visualizations saved to: {output_dir}")
        print("="*60)
    
    def create_layer_comparison(self, image_path: str, save_path=None):
        """
        Create a comparison visualization showing feature maps from multiple layers
        
        Args:
            image_path: Path to input image
            save_path: Path to save the comparison figure
        """
        # Extract features
        feature_maps, original_img = self.extract_features(image_path)
        
        # Select key layers to compare
        layer_names = list(feature_maps.keys())
        if len(layer_names) > 3:
            selected_layers = [
                layer_names[0],
                layer_names[len(layer_names)//2],
                layer_names[-1]
            ]
        else:
            selected_layers = layer_names
        
        # Create comparison figure
        num_layers = len(selected_layers)
        fig, axes = plt.subplots(num_layers + 1, 1, figsize=(12, 4*(num_layers + 1)))
        
        if num_layers == 0:
            axes = [axes]
        
        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image', fontsize=16, fontweight='bold')
        axes[0].axis('off')
        
        # Feature maps
        for idx, layer_name in enumerate(selected_layers, start=1):
            fmap = feature_maps[layer_name]
            avg_fmap = np.mean(fmap, axis=-1)
            
            # Normalize
            if avg_fmap.size == 0 or np.all(avg_fmap == 0):
                avg_fmap_normalized = np.zeros_like(avg_fmap)
            else:
                fmap_min = avg_fmap.min()
                fmap_max = avg_fmap.max()
                if fmap_max - fmap_min < 1e-8:
                    avg_fmap_normalized = np.zeros_like(avg_fmap)
                else:
                    avg_fmap_normalized = (avg_fmap - fmap_min) / (fmap_max - fmap_min)
            
            im = axes[idx].imshow(avg_fmap_normalized, cmap='jet', aspect='auto')
            axes[idx].set_title(f'{layer_name}\nShape: {fmap.shape}', 
                        fontsize=14, fontweight='bold')
            axes[idx].axis('off')
            
            try:
                plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
            except:
                pass
        
        plt.subplots_adjust(hspace=0.3)
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 Comparison saved to: {save_path}")
            except Exception as e:
                plt.savefig(save_path, dpi=300)
                print(f"💾 Comparison saved to: {save_path}")
        
        plt.close(fig)
    
    def create_activation_heatmap(self, image_path: str, save_path=None):
        """
        Create Grad-CAM style heatmap
        
        Args:
            image_path: Path to input image
            save_path: Path to save heatmap
        """
        print("\n🔥 Generating activation heatmap...")
        
        # Get the last convolutional layer
        conv_layers = self.get_conv_layer_names()
        if not conv_layers:
            print("❌ No convolutional layers found")
            return
        
        # Find last conv layer (not batch norm)
        last_conv = None
        for layer_name, layer_obj in reversed(conv_layers):
            if isinstance(layer_obj, keras.layers.Conv2D) or \
               isinstance(layer_obj, keras.layers.DepthwiseConv2D):
                last_conv = (layer_name, layer_obj)
                break
        
        if not last_conv:
            last_conv = conv_layers[-1]
        
        last_conv_layer_name, last_conv_layer_obj = last_conv
        print(f"   Using layer: {last_conv_layer_name}")
        
        # Preprocess image
        img_array, original_img = self.preprocess_image(image_path)
        
        # Create extractor if not exists
        if last_conv_layer_name not in self.feature_extractors:
            mobilenet_extractor = models.Model(
                inputs=self.mobilenet_model.input,
                outputs=last_conv_layer_obj.output
            )
            
            if self.augmentation_layer:
                inputs = keras.Input(shape=Config.IMG_SIZE + (3,))
                x = self.augmentation_layer(inputs, training=False)
                outputs = mobilenet_extractor(x, training=False)
                feature_extractor = models.Model(inputs=inputs, outputs=outputs)
            else:
                feature_extractor = mobilenet_extractor
                
            self.feature_extractors[last_conv_layer_name] = feature_extractor
        
        # Extract features
        features = self.feature_extractors[last_conv_layer_name].predict(img_array, verbose=0)[0]
        
        # Average across channels
        heatmap = np.mean(features, axis=-1)
        
        # Normalize
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 1e-8:
            heatmap /= np.max(heatmap)
        
        # Resize heatmap
        heatmap_resized = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))
        
        # Colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Superimpose
        original_array = np.array(original_img)
        superimposed = cv2.addWeighted(original_array, 0.6, heatmap_colored, 0.4, 0)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap_resized, cmap='jet', aspect='auto')
        axes[1].set_title('Activation Heatmap', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(superimposed)
        axes[2].set_title('Heatmap Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f'Class Activation Map (CAM)\nLayer: {last_conv_layer_name}', 
                    fontsize=16, fontweight='bold')
        
        plt.subplots_adjust(wspace=0.1)
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 Heatmap saved to: {save_path}")
            except Exception as e:
                plt.savefig(save_path, dpi=300)
                print(f"💾 Heatmap saved to: {save_path}")
        
        plt.close(fig)


def main():
    """Main execution"""
    print("="*60)
    print("🎨 FEATURE MAP VISUALIZATION")
    print("="*60)
    
    # Configuration
    model_path = Config.MODELS_DIR / "palm_classifier_final.keras"
    output_dir = Config.MODELS_DIR / "feature_maps"
    
    # Get test image from command line
    import sys
    if len(sys.argv) > 1:
        test_images = [sys.argv[1]]
        print(f"\n📷 Using image from argument: {test_images[0]}")
    else:
        # Find images in validation set
        val_dir = Config.DATASET_PROCESSED / "validation"
        test_images = []
        
        if val_dir.exists():
            for class_dir in val_dir.iterdir():
                if class_dir.is_dir():
                    images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                    if images:
                        test_images.append(images[0])
                        if len(test_images) >= 2:
                            break
        
        if not test_images:
            print("\n❌ No test images found!")
            print("   Usage: python generate_feature_maps.py /path/to/image.jpg")
            return
        
        print(f"\n📷 Found {len(test_images)} test images")
    
    # Check model
    if not model_path.exists():
        print(f"\n❌ Model not found at {model_path}")
        return
    
    # Initialize visualizer
    visualizer = FeatureMapVisualizer(str(model_path))
    
    # Get conv layers
    conv_layers = visualizer.get_conv_layer_names()
    
    if not conv_layers:
        print("\n❌ No convolutional layers found!")
        return
    
    print(f"\n📊 Found {len(conv_layers)} layers:")
    for i, (layer_name, _) in enumerate(conv_layers[:10], 1):
        print(f"   {i}. {layer_name}")
    if len(conv_layers) > 10:
        print(f"   ... and {len(conv_layers) - 10} more")
    
    # Create feature extractors (limit to 20 to avoid memory issues)
    print("\n🔧 Setting up feature extractors...")
    visualizer.get_layer_outputs(conv_layers, max_layers=20)
    
    if not visualizer.feature_extractors:
        print("\n❌ Failed to create feature extractors!")
        return
    
    print(f"✅ Ready to extract features from {len(visualizer.feature_extractors)} layers")
    
    # Process each test image
    for test_image in test_images:
        test_image_path = Path(test_image)
        
        if not test_image_path.exists():
            print(f"\n⚠️  Image not found: {test_image}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {test_image_path.name}")
        print(f"{'='*60}")
        
        try:
            # Generate all feature maps
            print("\n1️⃣  Generating individual layer feature maps...")
            visualizer.visualize_all_layers(
                str(test_image_path), 
                str(output_dir),
                num_filters_per_layer=16
            )
            
            # Create layer comparison
            print("\n2️⃣  Creating layer comparison...")
            img_name = test_image_path.stem
            comparison_path = output_dir / f"{img_name}_comparison.png"
            visualizer.create_layer_comparison(
                str(test_image_path),
                save_path=str(comparison_path)
            )
            
            # Create activation heatmap
            print("\n3️⃣  Creating activation heatmap...")
            heatmap_path = output_dir / f"{img_name}_heatmap.png"
            visualizer.create_activation_heatmap(
                str(test_image_path),
                save_path=str(heatmap_path)
            )
            
        except Exception as e:
            print(f"\n❌ Error processing {test_image_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ ALL VISUALIZATIONS COMPLETED!")
    print("="*60)
    print(f"📁 Output directory: {output_dir}")
    print("\n📊 Generated files:")
    
    if output_dir.exists():
        files = sorted(output_dir.glob("*.png"))
        if files:
            for f in files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"   • {f.name} ({size_mb:.2f} MB)")
        else:
            print("   ⚠️  No files generated!")
    
    print("="*60)


if __name__ == "__main__":
    main()