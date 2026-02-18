# Animal Parts Segmentation with PyTorch 

A high-performance data pipeline and preprocessing framework for multi-class semantic segmentation of animal body parts. This project demonstrates advanced handling of image-mask pairs and custom dataset integration in PyTorch.

## Project Overview
Semantic segmentation in the wild is challenging due to the complex morphology of different species. This repository provides a robust foundation for building AI models that can identify and delineate specific animal parts (e.g., head, torso, legs) with high precision.

### Key Features:
* **Dynamic Data Fetching:** Integrated with `kagglehub` for automated dataset management.
* **Advanced Mask Remapping:** Custom logic to normalize inconsistent pixel labels into a continuous range suitable for Cross-Entropy loss.
* **Optimized PyTorch Pipeline:** Custom `Dataset` class implementation with on-the-fly transformations and visualization.
* **GPU Accelerated:** Designed to run seamlessly on NVIDIA T4 GPUs (Google Colab optimized).

## Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Data Science:** NumPy, Pandas
* **Image Processing:** PIL (Pillow)
* **Visualization:** Matplotlib

## Dataset Information
The project utilizes the **Animal Segmentation Dataset** from Kaggle.
* **Target:** Pixel-level classification of animal parts.
* **Preprocessing:** Includes automated resizing (224x224) and tensor normalization.

## Getting Started

### 1. Installation
```bash
pip install torch torchvision kagglehub matplotlib pillow
```
---

### 2. Usage
Clone the repo and run the Jupyter Notebook:

```bash
git clone https://github.com/CS-Abdulaziz/Animal-Parts-Segmentation-PyTorch.git
cd Animal-Parts-Segmentation-PyTorch
# Run the .ipynb file in your preferred environment
```


### Implementation Details
A core highlight of this project is the remap_mask function, which solves a common issue in segmentation datasets where label values are non-sequential. This ensures the model's output layer aligns perfectly with the ground truth.

```bash
# Strategic remapping of labels
def remap_mask(mask):

    mask = mask.long()
    unique_values = torch.unique(mask)
    remapped_mask = torch.zeros_like(mask)

    for new_val, old_val in enumerate(sorted(unique_values.tolist())):
        remapped_mask[mask == old_val] = new_val

    return remapped_mask
```

### Visualizing Results

The pipeline includes a visualization suite to inspect the quality of the dataset before training:

- Input Image: Raw RGB data.
- Ground Truth: Remapped semantic masks.
- Overlay: Combined view for accuracy verification.

### Developed by

Abdulaziz Khamis

Computer Science Student, AI Engineer & Flutter develpoer
