# Final Project Outline

## Title Page & Abstract
- Title: [Working Title]
- Author: [Your Name]
- Course: INFO6152 Deep Learning with TensorFlow & Keras 2
- Date: November 30, 2025
- Abstract: 150–200 words summarizing dataset, two+ architectures, preprocessing focus, training setup, and comparative results.

## Introduction
- Motivation: Why this problem/dataset matters.
- Objectives: What you aim to demonstrate and compare.
- Dataset: Source (Kaggle/UCI/TFDS), modality (image or time series), size, characteristics.
- Relevance: How deep learning applies and expected challenges.

## Methodology

### Dataset & Preprocessing
- Acquisition: Download/source details and licensing.
- Train/val/test split strategy.
- Preprocessing (major focus):
  - For images: resizing, normalization, augmentation (rotation, flip, zoom, crop, color jitter), class balance.
  - For time series: scaling, windowing, sequence padding, missing values handling, stationarity considerations.
- Justifications: Why each step supports chosen models and improves generalization.
- Data pipeline: `tf.data`/`ImageDataGenerator`/custom generators; performance notes.

### Model Architectures (at least two)
- Model A: [e.g., Transfer Learning with CNN: VGG16/ResNet50]
  - Task: [classification/regression]
  - Input shape and key layers; fine-tuning strategy.
  - Loss, optimizer, regularization (dropout, weight decay), LR schedule.
- Model B: [e.g., RNN/LSTM/GRU or Transformer]
  - Task alignment to dataset (sequence vs. image).
  - Architecture specifics: units, layers, attention, positional encoding.
  - Loss, optimizer, regularization, LR schedule.
- Optional Model C: [e.g., GAN for augmentation/generation].
- Rationale: Why each architecture suits the dataset; differences and expectations.

### Training Process & Challenges
- Setup: batch size, epochs, callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint), mixed precision.
- Curves: training and validation loss/metrics; how they were generated.
- Challenges: overfitting, vanishing/exploding gradients, instability, data imbalance.
- Mitigations: augmentation, regularization, class weights, gradient clipping, curriculum, tuning.

## Results & Comparison
- Metrics: accuracy/F1/AUC for classification; MSE/MAE for regression; throughput (images/sec) and epoch time.
- Tables/plots: side-by-side comparison and learning curves.
- Robustness: sensitivity to noise/perturbations; stability across seeds.
- Discussion: insights and trade-offs.

## Conclusion & Future Work
- Summary: key findings and lessons.
- Limitations: data/model constraints and external validity.
- Future Work: model improvements, more data, alternative architectures.

## References
- Dataset citation(s) and licenses.
- Papers and docs for architectures and methods.
- Libraries: TensorFlow/Keras versions and other tools.

---

## Execution Plan (Notebook Alignment)
- Notebook sections mirror this outline.
- Ensure all code cells run with visible outputs.
- Save artifacts: trained models, plots, and logs under `artifacts/`.
