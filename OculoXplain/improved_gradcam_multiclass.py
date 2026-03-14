"""
Phase 2 - Person 4: Improved Grad-CAM Visualization
- Improve Grad-CAM visualization (overlay heatmaps nicely on fundus images)
- Show heatmaps for multiple classes
- Better visualization with multiple disease predictions
"""

import torch
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import pandas as pd
import os
from PIL import Image

class ImprovedRetinalExplainer:
    def __init__(self, model_path=None, csv_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔬 Improved Multi-Class Retinal Disease Explainer")
        print(f"Using device: {self.device}")
        
        # Paths
        self.model_path = model_path or 'quick_model.pth'
        self.csv_path = csv_path or r'D:\OCULOXPLAIN\OculoXplain\data\ODIR-5K\full_df_with_split_augmented.csv'
        self.img_dir = r'D:\OCULOXPLAIN\OculoXplain\data\ODIR-5K\ODIR-5K\Training Images'
        
        # Load model and data
        self.model, self.label_map, self.idx_to_label = self._load_model()
        self.df = self._load_data()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Setup Grad-CAM
        self.grad_cam = self._setup_gradcam()
        
        # Disease information
        self.disease_info = {
            'N': {'name': 'Normal', 'color': (0, 255, 0), 'description': 'Healthy retina'},
            'D': {'name': 'Diabetic Retinopathy', 'color': (255, 0, 0), 'description': 'Blood vessel damage due to diabetes'},
            'G': {'name': 'Glaucoma', 'color': (255, 255, 0), 'description': 'Optic nerve damage'},
            'C': {'name': 'Cataract', 'color': (0, 255, 255), 'description': 'Lens opacity'},
            'A': {'name': 'Age-related Macular Degeneration', 'color': (255, 0, 255), 'description': 'Central vision loss'},
            'H': {'name': 'Hypertensive Retinopathy', 'color': (128, 0, 128), 'description': 'High blood pressure effects'},
            'M': {'name': 'Myopia', 'color': (255, 128, 0), 'description': 'Nearsightedness complications'},
            'O': {'name': 'Other Diseases', 'color': (128, 128, 128), 'description': 'Various other conditions'}
        }
    
    def _load_model(self):
        """Load the trained multi-class model"""
        print("Loading trained multi-class model...")
        
        # For quick_model.pth (state_dict only)
        num_classes = 8  # We know there are 8 classes from your label mapping
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        
        # Manual label mapping (from your training output)
        label_map = {'A': 0, 'C': 1, 'D': 2, 'G': 3, 'H': 4, 'M': 5, 'N': 6, 'O': 7}
        idx_to_label = {idx: label for label, idx in label_map.items()}
        
        print(f"Model loaded successfully with {num_classes} classes")
        return model, label_map, idx_to_label
        
    def _load_data(self):
        """Load the dataset"""
        df = pd.read_csv(self.csv_path)
        df['main_label'] = df['labels'].str.extract(r"\['([A-Z])'\]")
        return df[df['split'] == 'val']  # Use validation set for explanation
    
    def _setup_gradcam(self):
        """Setup Grad-CAM"""
        target_layers = [self.model.layer4[-1]]
        return GradCAM(model=self.model, target_layers=target_layers)
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize for display
        image_display = cv2.resize(image_rgb, (224, 224))
        image_display_norm = image_display.astype(np.float32) / 255.0
        
        # Preprocess for model
        image_pil = Image.fromarray(image_rgb)
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        return image_tensor, image_display_norm, image_display
    
    def predict_with_confidence(self, image_tensor):
        """Get predictions with confidence scores"""
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence_scores = probabilities[0].cpu().numpy()
        
        # Get top 3 predictions
        top_indices = np.argsort(confidence_scores)[::-1][:3]
        predictions = []
        
        for idx in top_indices:
            disease_code = self.idx_to_label[idx]
            confidence = confidence_scores[idx]
            disease_name = self.disease_info.get(disease_code, {}).get('name', disease_code)
            predictions.append({
                'code': disease_code,
                'name': disease_name,
                'confidence': confidence,
                'index': idx
            })
        
        return predictions
    
    def generate_gradcam_for_class(self, image_tensor, target_class_idx):
        """Generate Grad-CAM for specific class"""
        targets = [ClassifierOutputTarget(target_class_idx)]
        grayscale_cam = self.grad_cam(input_tensor=image_tensor, targets=targets)
        return grayscale_cam[0, :]
    
    def create_improved_visualization(self, image_path, save_path=None):
        """Create improved visualization with multiple class explanations"""
        # Load and preprocess image
        image_tensor, image_display_norm, image_display = self.load_and_preprocess_image(image_path)
        
        # Get predictions
        predictions = self.predict_with_confidence(image_tensor)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Multi-Class Retinal Disease Analysis\nImage: {os.path.basename(image_path)}', 
                     fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(image_display)
        axes[0, 0].set_title('Original Fundus Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Top 3 predictions with Grad-CAM
        for i, pred in enumerate(predictions[:3]):
            if i >= 2:  # Only show top 2 predictions + combined
                break
                
            row = i // 3
            col = i + 1
            
            # Generate Grad-CAM
            cam = self.generate_gradcam_for_class(image_tensor, pred['index'])
            
            # Create visualization
            visualization = show_cam_on_image(image_display_norm, cam, use_rgb=True)
            
            # Plot
            axes[row, col].imshow(visualization)
            title = f"{pred['name']}\n({pred['code']}) - {pred['confidence']:.1%}"
            axes[row, col].set_title(title, fontsize=11, fontweight='bold')
            axes[row, col].axis('off')
            
            # Add color-coded border
            disease_color = self.disease_info.get(pred['code'], {}).get('color', (128, 128, 128))
            color_norm = tuple(c/255.0 for c in disease_color)
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor(color_norm)
                spine.set_linewidth(3)
        
        # Combined heatmap (average of top 2)
        if len(predictions) >= 2:
            cam1 = self.generate_gradcam_for_class(image_tensor, predictions[0]['index'])
            cam2 = self.generate_gradcam_for_class(image_tensor, predictions[1]['index'])
            combined_cam = (cam1 + cam2) / 2
            
            combined_viz = show_cam_on_image(image_display_norm, combined_cam, use_rgb=True)
            axes[0, 2].imshow(combined_viz)
            axes[0, 2].set_title('Combined Heatmap\n(Top 2 Predictions)', fontsize=11, fontweight='bold')
            axes[0, 2].axis('off')
        
        # Find the section around line 165-170 where it handles the bottom-left plot:

        # Prediction summary (change from axes[1, 1] to axes[1, 0])
        axes[1, 0].axis('off')  # Changed from axes[1, 1] to axes[1, 0]
        summary_text = "🔍 PREDICTION SUMMARY\n\n"
        for i, pred in enumerate(predictions):
            bar_length = int(pred['confidence'] * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            summary_text += f"{i+1}. {pred['name']} ({pred['code']})\n"
            summary_text += f"   Confidence: {pred['confidence']:.1%} {bar}\n"
            summary_text += f"   {self.disease_info.get(pred['code'], {}).get('description', '')}\n\n"

        axes[1, 0].text(0.05, 0.95, summary_text, transform=axes[1, 0].transAxes,  # Changed from axes[1, 1]
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        # Keep the interpretation in axes[1, 2] and add something useful to axes[1, 1]
        axes[1, 1].axis('off')
        confidence_chart_text = "📊 CONFIDENCE BREAKDOWN\n\n"
        for i, pred in enumerate(predictions):
            percentage = f"{pred['confidence']:.1%}"
            confidence_chart_text += f"{pred['code']}: {percentage:<6} {'●' * int(pred['confidence'] * 10)}\n"
            
        axes[1, 1].text(0.05, 0.95, confidence_chart_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
        # Heatmap interpretation
        axes[1, 2].axis('off')
        interpretation_text = "🎯 HEATMAP INTERPRETATION\n\n"
        interpretation_text += "🔴 Red/Warm Areas:\n"
        interpretation_text += "• Regions supporting the prediction\n"
        interpretation_text += "• Key diagnostic features\n"
        interpretation_text += "• Areas of concern\n\n"
        interpretation_text += "🔵 Blue/Cool Areas:\n"
        interpretation_text += "• Regions opposing the prediction\n"
        interpretation_text += "• Less relevant areas\n"
        interpretation_text += "• Normal tissue regions\n\n"
        interpretation_text += "📍 Focus Areas:\n"
        interpretation_text += "• Optic disc (central)\n"
        interpretation_text += "• Macula (central-temporal)\n"
        interpretation_text += "• Blood vessels\n"
        interpretation_text += "• Retinal periphery"
        
        axes[1, 2].text(0.05, 0.95, interpretation_text, transform=axes[1, 2].transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        return predictions
    
    def explain_random_samples(self, num_samples=3):
        """Explain random validation samples"""
        print(f"\n🎯 Generating explanations for {num_samples} random samples...")
        
        # Get random samples from validation set
        sample_df = self.df.sample(num_samples, random_state=42)
        
        for idx, (_, row) in enumerate(sample_df.iterrows()):
            print(f"\n--- Sample {idx+1}/{num_samples} ---")
            image_path = os.path.join(self.img_dir, row['filename'])
            save_path = f'improved_gradcam_explanation_{idx+1}.png'
            
            try:
                predictions = self.create_improved_visualization(image_path, save_path)
                print(f"True label: {row['main_label']} | Predicted: {predictions[0]['code']} ({predictions[0]['confidence']:.1%})")
            except Exception as e:
                print(f"Error processing {row['filename']}: {e}")
    
    def explain_specific_image(self, image_path, save_path=None):
        """Explain a specific image"""
        print(f"\n🔍 Analyzing image: {image_path}")
        if save_path is None:
            save_path = f'explanation_{os.path.basename(image_path)}.png'
        
        try:
            predictions = self.create_improved_visualization(image_path, save_path)
            return predictions
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None

def main():
    print("🔬 Improved Multi-Class Retinal Disease Explainer")
    print("=" * 60)
    
    # Initialize explainer
    explainer = ImprovedRetinalExplainer()
    
    # Generate explanations for random samples
    explainer.explain_random_samples(num_samples=3)
    
    print("\n✅ Multi-class explainability analysis complete!")
    print("📊 Generated visualizations show:")
    print("   • Original fundus image")
    print("   • Top predictions with individual heatmaps")
    print("   • Combined heatmap highlighting key regions")
    print("   • Detailed prediction summary with confidence scores")
    print("   • Heatmap interpretation guide")

if __name__ == "__main__":
    main()