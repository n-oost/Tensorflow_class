# TensorFlow Class Final Project

This repository contains the scaffold for the INFO6152 final project: preprocessing, two classification models (baseline CNN + MobileNetV2 transfer learning), optional DCGAN for CIFAR-10 truck class, benchmarking, and artifact saving.

## Contents
- `Final Project Notebook.ipynb` – Main reproducible experiment notebook.
- `final_project_outline.md` – Report outline & planning.
- `gan_cifar10_truck.py` – Standalone DCGAN training script for truck images.
- `artifacts/` – Saved models, histories, generated samples, plots, and requirements.

## Quick Start (fish shell)
```fish
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate.fish

# Install core dependencies
pip install tensorflow tensorflow-datasets matplotlib seaborn numpy pandas scikit-learn
```

## Running the Notebook
Open `Final Project Notebook.ipynb` in VS Code or Jupyter and run cells sequentially:
1. Environment setup & imports
2. CIFAR-10 load and visualization
3. Preprocessing & augmentation
4. Baseline CNN build + compile
5. Transfer learning model (MobileNetV2) build + compile
6. Data pipelines
7. Train both models (callbacks manage early stopping)
8. Plot curves and evaluate
9. Benchmark speed & compare
10. (Optional) GAN section – can be skipped if focusing only on classification

Artifacts (models, histories, plots) are saved automatically under `artifacts/`.

## Running the GAN Script
```fish
source .venv/bin/activate.fish
python gan_cifar10_truck.py --epochs 20 --subset 10000 --latent-dim 128 --out-dir artifacts/gan_truck
```
- Samples saved to `artifacts/gan_truck/` as PNGs.
- Final generator/discriminator SavedModel exports stored in same directory.

## Reproducibility
- Global seed set to `42` inside the notebook for NumPy, Python `random`, and TensorFlow.
- Mixed precision automatically enabled on GPU if available.
- Use consistent batch sizes and avoid non-deterministic ops for strict reproducibility (may vary across hardware).

## Extending
- Fine-tune MobileNetV2: set `fine_tune_at` in head builder to unfreeze last N layers after initial convergence.
- Add evaluation metrics: F1-score, per-class accuracy, confusion matrices (already included in notebook).
- Add more architectures: Vision Transformer (ViT), EfficientNet, or an RNN/Transformer for time-series tasks if you change dataset modality.
- GAN evaluation: integrate FID/IS metrics (requires additional libraries such as `tensorflow-gan`).

## Report Guidance
Use `final_project_outline.md` to structure the PDF report (6–10 pages excluding title page). Populate each section with:
- Dataset description & preprocessing decisions (justify each transformation)
- Architectural rationale & differences
- Training setup, challenges, and mitigation strategies
- Comparative tables/plots (accuracy, speed, stability)
- Conclusions & future work

## Cleaning Up
```fish
# Remove generated artifacts if needed
rm -rf artifacts/gan_truck artifacts/baseline_savedmodel artifacts/transfer_savedmodel
```

## Notes
- CIFAR-10 images are upscaled to 224x224 for transfer learning consistency; native 32x32 used for GAN.
- Requirements snapshot stored at `artifacts/requirements.txt` (extend as needed).

---
Feel free to request additional sections (e.g., advanced metrics, data augmentation experiments, or time-series adaptation).