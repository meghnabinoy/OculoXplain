# OculoXplain - Explainable AI for Rare Retinal Diseases

This project implements explainable AI for rare retinal disease detection using deep learning. The project is completed in phases as outlined below.

## ✅ Phase 1 Completed - Setup & Data Understanding

### 1. Requirements Installation
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation - COMPLETED ✅
- **Dataset Used**: ODIR-5K (Ocular Disease Intelligent Recognition)
- **Location**: `Data/Ocular_Disease_Dataset/ODIR-5K/ODIR-5K/`
- **Processed Data**: `Data/Ocular_Disease_Dataset/ODIR-5K/ODIR-5K/processed/`
- **Total Images**: 7,000 images (5,600 training, 1,400 validation)

### 3. Data Preprocessing - COMPLETED ✅
**Script**: `preprocess_odir5k_updated.py`
- Resizes images to 224x224
- Removes black borders
- Normalizes pixel values (0-1)
- Splits into train/validation (80/20)
- **Output**: Binary classification (Healthy vs Disease)

### 4. Baseline Model Training - COMPLETED ✅
**Script**: `baseline_resnet50_updated.py`
- **Model**: ResNet50 (pretrained on ImageNet)
- **Training**: 10 epochs with GPU acceleration
- **Performance**: High accuracy with proper convergence
- **Optimizations**: 
  - Batch size 128 (optimal for RTX 5060)
  - Learning rate scheduling
  - Weight decay for regularization
- **Output**: `resnet50_retinal_disease_model.pth`

### 5. Explainable AI (Grad-CAM) - COMPLETED ✅
**Script**: `gradcam_explainer.py`
- **Implementation**: Grad-CAM on ResNet50 outputs
- **Features**:
  - Generates heatmaps showing regions influencing predictions
  - Visualizes confidence scores for both classes
  - Supports both random samples and specific image analysis
- **Outputs**: 
  - `gradcam_explanations.png` - Multiple sample explanations
  - Individual explanation images for specific cases

## 🚀 How to Run the Complete Pipeline

### Step 1: Preprocess Data
```bash
python preprocess_odir5k_updated.py
```

### Step 2: Train the Model
```bash
python baseline_resnet50_updated.py
```

### Step 3: Generate Explanations
```bash
python gradcam_explainer.py
```

## 📁 Project Structure
```
├── Data/
│   └── Ocular_Disease_Dataset/
│       ├── full_df.csv
│       └── ODIR-5K/ODIR-5K/
│           ├── data.xlsx
│           ├── Training Images/
│           └── processed/
│               ├── train/ (5,600 images)
│               └── val/ (1,400 images)
├── baseline_resnet50_updated.py      # Main training script
├── preprocess_odir5k_updated.py      # Data preprocessing
├── gradcam_explainer.py               # Explainable AI implementation
├── resnet50_retinal_disease_model.pth # Trained model
├── training_curves.png                # Training progress visualization
└── requirements.txt                   # Dependencies
```

## 🎯 Results Achieved

### Model Performance
- **Architecture**: ResNet50 with transfer learning
- **Training Time**: ~50-100 minutes on RTX 5060 GPU
- **Convergence**: Stable learning with proper validation

### Explainability
- **Method**: Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Output**: Visual heatmaps highlighting important retinal regions
- **Confidence**: Model provides probability scores for both classes
- **Medical Relevance**: Helps doctors understand AI decision-making process

## 👥 Team Contributions

- **Person 1 (Data & Setup)**: Dataset preparation and exploration ✅
- **Person 2 (Preprocessing)**: Image preprocessing pipeline ✅  
- **Person 3 (Baseline Model)**: ResNet50 training and optimization ✅
- **Person 4 (Explainability)**: Grad-CAM implementation and visualization ✅

## 🔧 Technical Specifications

### Hardware Requirements
- **GPU**: NVIDIA GPU with 4GB+ VRAM (tested on RTX 5060)
- **RAM**: 8GB+ recommended
- **Storage**: 10GB+ for dataset and models

### Software Requirements
- **Python**: 3.8+
- **PyTorch**: 2.1.0+ with CUDA support
- **Key Libraries**: torchvision, pytorch-grad-cam, opencv-python, matplotlib

## 📊 Generated Outputs

1. **Trained Model**: `resnet50_retinal_disease_model.pth` (94MB)
2. **Training Curves**: `training_curves.png`
3. **Grad-CAM Explanations**: `gradcam_explanations.png`
4. **Processed Dataset**: Organized in `processed/` folder

---

**Project Status**: ✅ **COMPLETED** - Ready for deployment and further research

For any issues or questions, check the script comments or contact the development team.
