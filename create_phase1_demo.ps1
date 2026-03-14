# Phase1 demo bundle creator (auto-generated)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Paths
Set-Location -Path (Resolve-Path ".\").Path
$demo = ".\outputs\phase1_demo"

# Clean and create demo folder
if (Test-Path $demo) { Remove-Item -Path $demo -Recurse -Force -ErrorAction SilentlyContinue }
New-Item -Path $demo -ItemType Directory -Force | Out-Null

# Copy key artifacts (adjust names if needed)
$filesToCopy = @(
    ".\gradcam_explanations.png",
    ".\gradcam_explanation_1034_right.jpg.png",
    ".\improved_gradcam_explanation_1.png",
    ".\improved_gradcam_explanation_2.png",
    ".\improved_gradcam_explanation_3.png",
    ".\training_curves.png",
    ".\data\ODIR-5K\full_df_with_split.csv",
    ".\data\Ocular_Disease_Dataset\full_df_with_split.csv",
    ".\quick_model.pth",
    ".\resnet50_retinal_disease_model.pth",
    ".\resnet50_multiclass_retinal_model.pth"
)

foreach ($f in $filesToCopy) {
    if (Test-Path $f) {
        Copy-Item -Path $f -Destination $demo -Force -ErrorAction SilentlyContinue
    } else {
        Write-Verbose "File not found (skipped): $f"
    }
}

# Create README.md with captions
$readme = @"
# Phase 1 — OculoXplain Results (Demo bundle)

This folder contains the Phase‑1 outputs to present in the demo.

Files:
- `gradcam_explanations.png` — Batch Grad‑CAM visualization: for each sample shows (Original → CAM overlay → confidence bar chart). Use this to demonstrate how the binary model (Healthy vs Disease) localizes salient regions.
- `gradcam_explanation_1034_right.jpg.png` — Example single-image detailed explanation (two-class breakdown if produced with both classes).
- `improved_gradcam_explanation_1.png`, `_2.png`, `_3.png` — Improved multi-class visualizations from the multi-class explainer. Layout per image:
  - Top-left: Original fundus.
  - Top panels: class-specific Grad‑CAM overlays for top predictions (color-coded).
  - Top-right: Combined heatmap (average of top-2 cams).
  - Bottom-left / middle / right: prediction summary, confidence breakdown, and a heatmap interpretation guide.
- `training_curves.png` — Training/validation loss & accuracy over epochs (use when discussing model convergence/overfitting).
- `full_df_with_split.csv` (ODIR-5K) & `Ocular_Disease_Dataset/full_df_with_split.csv` — dataset splits and label encodings; show class imbalance and sample filenames.
- `quick_model.pth`, `resnet50_retinal_disease_model.pth`, `resnet50_multiclass_retinal_model.pth` — model checkpoints (weights).

Interpretation notes (talking points):
- Grad‑CAM overlays: warm/red regions indicate areas that increased the model's likelihood for the predicted class; cool areas indicate less influence.
- Use single-image comparisons (original vs CAM) to point out anatomical regions model focuses on (optic disc, macula, vessel clusters).
- Combined heatmaps (multi-class) highlight overlapping cues and possible ambiguity—use these to motivate clinician review.
- Training curves show learning dynamics; watch for large gap between training and validation indicating overfitting.

Limitations to mention:
- Grad‑CAM is qualitative and shows correlation, not proof of pathology.
- Visualizations should be validated with clinician annotations before deploying in clinical workflows.
"@
Set-Content -Path (Join-Path $demo 'README.md') -Value $readme -Encoding UTF8

# Optional: create a small Python montage script (requires Pillow)
$montageScript = @" 
from PIL import Image, ImageOps
import os, sys
demo = os.path.join(os.getcwd(), "outputs", "phase1_demo")
files = ["gradcam_explanations.png", "improved_gradcam_explanation_1.png", "training_curves.png"]
images = []
for f in files:
    p = os.path.join(demo, f)
    if os.path.exists(p):
        img = Image.open(p).convert("RGB")
        img = ImageOps.fit(img, (800, 500), Image.Resampling.LANCZOS)
        images.append(img)
if len(images)==0:
    print("No images found to create montage.")
    sys.exit(0)
w, h = images[0].size
montage = Image.new("RGB", (w*len(images), h), (255,255,255))
for i,img in enumerate(images):
    montage.paste(img, (i*w, 0))
out = os.path.join(demo, "phase1_montage.png")
montage.save(out, format="PNG", dpi=(150,150))
print("Saved montage to", out)
"@
Set-Content -Path ".\create_montage_phase1.py" -Value $montageScript -Encoding UTF8

# Try to run the montage script (if Python + Pillow available)
try {
    & python .\create_montage_phase1.py
} catch {
    Write-Verbose "Python montage step failed or Python not installed: $_"
}

# Zip the demo folder
$zipPath = ".\outputs\phase1_demo.zip"
if (Test-Path $zipPath) { Remove-Item -Path $zipPath -Force -ErrorAction SilentlyContinue }
Compress-Archive -Path $demo\* -DestinationPath $zipPath -Force

# Report results
$zip = Get-Item $zipPath -ErrorAction SilentlyContinue
if ($zip) {
    Write-Output "Created: $($zip.FullName) — Size(MB): $([math]::Round($zip.Length/1MB,2))"
} else {
    Write-Output "Demo folder created at: $demo"
}