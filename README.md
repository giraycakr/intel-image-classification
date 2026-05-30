# Intel Image Classification with Transfer Learning

Scene classification of natural images into 6 categories using transfer learning with a fine-tuned ResNet18 model.

## Overview

This project classifies natural scene images into six categories — **buildings**, **forest**, **glacier**, **mountain**, **sea**, and **street** — using the [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) dataset, originally published on Analytics Vidhya.

A pretrained ResNet18 model is fine-tuned with discriminative learning rates and data augmentation to achieve **92% accuracy** on the test set.

## Dataset

- **Source:** [Kaggle - Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Training images:** ~14,000
- **Test images:** ~3,000
- **Classes:** 6 (buildings, forest, glacier, mountain, sea, street)
- **Image size:** 150×150 RGB

## Approach

- **Transfer Learning:** Pretrained ResNet18 (ImageNet weights) with the final convolutional block (`layer4`) and classifier head unfrozen for fine-tuning
- **Discriminative Learning Rates:** 1e-4 for `layer4`, 1e-3 for the classifier head
- **Data Augmentation:** Random rotation, color jitter (brightness, contrast, saturation, hue)
- **Training:** 5 epochs on GPU with batch size 32

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 92% |
| Macro F1 | 0.92 |
| Best Class (Forest) | 0.99 F1 |
| Hardest Classes (Glacier, Mountain) | 0.87 F1 |

The model struggles most with visually similar classes (glacier ↔ mountain, buildings ↔ street), which is expected given the visual overlap between these categories.

## Project Structure

```
├── README.md
├── intel_image_classification.ipynb
```

## How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
2. Open the notebook in Google Colab or JupyterLab
3. Update the data path to point to your downloaded dataset
4. Run all cells (GPU recommended)

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- seaborn
- scikit-learn
- numpy

## Tech Stack

- **Framework:** PyTorch
- **Model:** ResNet18 (pretrained on ImageNet)
- **Training:** Google Colab (A100 GPU)