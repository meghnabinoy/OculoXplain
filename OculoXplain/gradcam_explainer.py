"""
Explainable AI for Retinal Disease Classification - Grad-CAM Implementation
Task: Person 4 (Explainability – First Step)
- Implement Grad-CAM on ResNet outputs
- Generate heatmaps showing which regions influenced the prediction
"""

import torch
import torch.nn.functional as F
from torchvision import transforms, models, datasets
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import random

class RetinalDiseaseExplainer:
    def __init__(self, model_path=None, data_dir=None):
        """
        Initialize the explainer with a trained model and data directory.
        
        Args:
            model_path: Path to the trained model (.pth file)
            data_dir: Path to the processed data directory
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Default paths
        if model_path is None:
            model_path = "resnet50_retinal_disease_model.pth"
        if data_dir is None:
            data_dir = r"D:\OCULOXPLAIN\OculoXplain\data\ODIR-5K\ODIR-5K\processed"
        
        self.model_path = model_path
        self.data_dir = data_dir  # This line was missing the assignment
        
        # Load model
        self.model = self._load_model()
        
        # Define transforms (same as training, but without augmentation)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])
        
        # Transform for visualization (without normalization)
        self.vis_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # Load validation dataset for testing
        self.val_dataset = self._load_validation_data()
        
        # Initialize Grad-CAM
        self.grad_cam = self._setup_gradcam()
        
    def _load_model(self):
        """Load the trained ResNet50 model."""
        print("Loading trained model...")
        
        # Create model architecture (same as training)
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, 2)  # 2 classes: healthy, disease
        
        # Load trained weights if available
        if os.path.exists(self.model_path):
            print(f"Loading weights from {self.model_path}")
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        else:
            print("Warning: Trained model not found. Using pretrained ResNet50 with random classifier.")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_validation_data(self):
        """Load validation dataset."""
        val_dir = os.path.join(self.data_dir, "val")
        if not os.path.exists(val_dir):
            print(f"Warning: Validation directory not found: {val_dir}")
            return None
        
        # Load without transforms initially to get original images
        val_dataset = datasets.ImageFolder(val_dir)
        print(f"Loaded {len(val_dataset)} validation images")
        return val_dataset
    
    def _setup_gradcam(self):
        """Initialize Grad-CAM with the model."""
        # Target the last convolutional layer of ResNet50
        target_layers = [self.model.layer4[-1]]
        
        grad_cam = GradCAM(model=self.model, target_layers=target_layers)
        return grad_cam
    
    def predict_image(self, image_path):
        """
        Make a prediction on a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            prediction: Class prediction (0=healthy, 1=disease)
            confidence: Confidence scores for both classes
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0].cpu().numpy()
        
        return prediction, confidence
    
    def generate_gradcam(self, image_path, target_class=None):
        """
        Generate Grad-CAM heatmap for an image.
        
        Args:
            image_path: Path to the image
            target_class: Target class for CAM (None for predicted class)
            
        Returns:
            cam_image: Image with CAM overlay
            original_image: Original image
            prediction: Model prediction
            confidence: Prediction confidence
        """
        # Load original image
        original_image = Image.open(image_path).convert('RGB')
        original_array = np.array(original_image)
        
        # Prepare input for the model
        input_tensor = self.transform(original_image).unsqueeze(0)
        
        # Get model prediction
        prediction, confidence = self.predict_image(image_path)
        
        # Set target class (use prediction if not specified)
        if target_class is None:
            target_class = prediction
        
        # Generate CAM
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.grad_cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # Get first (and only) image
        
        # Resize original image to match CAM size
        original_resized = cv2.resize(original_array, (224, 224))
        original_resized = original_resized / 255.0  # Normalize to [0, 1]
        
        # Create CAM overlay
        cam_image = show_cam_on_image(original_resized, grayscale_cam, use_rgb=True)
        
        return cam_image, original_resized, prediction, confidence
    
    def explain_random_samples(self, num_samples=5):
        """
        Generate explanations for random samples from validation set.
        
        Args:
            num_samples: Number of random samples to explain
        """
        if self.val_dataset is None:
            print("No validation dataset available.")
            return
        
        # Get random samples
        sample_indices = random.sample(range(len(self.val_dataset)), min(num_samples, len(self.val_dataset)))
        
        print(f"Generating Grad-CAM explanations for {len(sample_indices)} random samples...")
        
        # Create figure for all samples
        fig, axes = plt.subplots(len(sample_indices), 3, figsize=(15, 5*len(sample_indices)))
        if len(sample_indices) == 1:
            axes = axes.reshape(1, -1)
        
        class_names = ['Healthy', 'Disease']
        
        for i, idx in enumerate(sample_indices):
            image_path, true_label = self.val_dataset.samples[idx]
            
            # Generate Grad-CAM
            cam_image, original_image, prediction, confidence = self.generate_gradcam(image_path)
            
            # Plot original image
            axes[i, 0].imshow(original_image)
            axes[i, 0].set_title(f'Original\nTrue: {class_names[true_label]}')
            axes[i, 0].axis('off')
            
            # Plot Grad-CAM
            axes[i, 1].imshow(cam_image)
            axes[i, 1].set_title(f'Grad-CAM Heatmap\nPred: {class_names[prediction]} ({confidence[prediction]:.2f})')
            axes[i, 1].axis('off')
            
            # Plot confidence scores
            bars = axes[i, 2].bar(class_names, confidence, color=['green' if prediction == j else 'red' for j in range(len(class_names))])
            axes[i, 2].set_title(f'Confidence Scores')
            axes[i, 2].set_ylabel('Probability')
            axes[i, 2].set_ylim(0, 1)
            
            # Add confidence values on bars
            for bar, conf in zip(bars, confidence):
                height = bar.get_height()
                axes[i, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{conf:.3f}', ha='center', va='bottom')
            
            print(f"Sample {i+1}: True={class_names[true_label]}, Pred={class_names[prediction]}, Conf={confidence[prediction]:.3f}")
        
        plt.tight_layout()
        plt.savefig('gradcam_explanations.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def explain_specific_image(self, image_path, show_both_classes=True):
        """
        Generate detailed explanation for a specific image.
        
        Args:
            image_path: Path to the specific image
            show_both_classes: Whether to show CAMs for both classes
        """
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return
        
        print(f"Generating explanation for: {image_path}")
        
        # Generate CAMs for both classes if requested
        if show_both_classes:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            class_names = ['Healthy', 'Disease']
            
            for class_idx in range(2):
                cam_image, original_image, prediction, confidence = self.generate_gradcam(image_path, target_class=class_idx)
                
                # Original image
                axes[class_idx, 0].imshow(original_image)
                axes[class_idx, 0].set_title('Original Image')
                axes[class_idx, 0].axis('off')
                
                # CAM for current class
                axes[class_idx, 1].imshow(cam_image)
                axes[class_idx, 1].set_title(f'Grad-CAM for {class_names[class_idx]} Class')
                axes[class_idx, 1].axis('off')
                
                # Confidence scores
                bars = axes[class_idx, 2].bar(class_names, confidence, 
                                            color=['green' if prediction == j else 'red' for j in range(len(class_names))])
                axes[class_idx, 2].set_title(f'Prediction: {class_names[prediction]} ({confidence[prediction]:.3f})')
                axes[class_idx, 2].set_ylabel('Probability')
                axes[class_idx, 2].set_ylim(0, 1)
                
                # Add confidence values on bars
                for bar, conf in zip(bars, confidence):
                    height = bar.get_height()
                    axes[class_idx, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                          f'{conf:.3f}', ha='center', va='bottom')
        
        else:
            # Show only for predicted class
            cam_image, original_image, prediction, confidence = self.generate_gradcam(image_path)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            class_names = ['Healthy', 'Disease']
            
            # Original image
            axes[0].imshow(original_image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # CAM
            axes[1].imshow(cam_image)
            axes[1].set_title(f'Grad-CAM Heatmap')
            axes[1].axis('off')
            
            # Confidence scores
            bars = axes[2].bar(class_names, confidence, 
                             color=['green' if prediction == j else 'red' for j in range(len(class_names))])
            axes[2].set_title(f'Prediction: {class_names[prediction]} ({confidence[prediction]:.3f})')
            axes[2].set_ylabel('Probability')
            axes[2].set_ylim(0, 1)
            
            # Add confidence values on bars
            for bar, conf in zip(bars, confidence):
                height = bar.get_height()
                axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{conf:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'gradcam_explanation_{os.path.basename(image_path)}.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    """Main function to demonstrate Grad-CAM explainability."""
    print("🔬 Retinal Disease Classification - Explainable AI with Grad-CAM")
    print("=" * 60)
    
    # Initialize explainer
    explainer = RetinalDiseaseExplainer()
    
    print("\\n1. Generating explanations for random validation samples...")
    explainer.explain_random_samples(num_samples=3)
    
    print("\\n2. You can also explain specific images using:")
    print("   explainer.explain_specific_image('path/to/your/image.jpg')")
    
    # Example of explaining a specific image (if you have one)
    val_dir = os.path.join(explainer.data_dir, "val")
    if os.path.exists(val_dir):
        # Find the first available image in validation set
        for class_folder in ['healthy', 'disease']:
            class_path = os.path.join(val_dir, class_folder)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
                if images:
                    sample_image = os.path.join(class_path, images[0])
                    print(f"\\n3. Detailed explanation for sample {class_folder} image:")
                    explainer.explain_specific_image(sample_image, show_both_classes=True)
                    break
    
    print("\\n✅ Explainability analysis complete!")
    print("📊 Generated visualizations show:")
    print("   • Red/warm regions: Areas that support the prediction")
    print("   • Blue/cool regions: Areas that oppose the prediction")
    print("   • Confidence scores for both classes")

if __name__ == "__main__":
    main()
