# Brain Tumor Detection and Classification using MRI

This project implements a deep learning pipeline for **brain tumor detection and classification** from MRI images using a **ResNet-50** convolutional neural network with PyTorch.

The model classifies MRI scans into four categories:
- Glioma tumor
- Meningioma tumor
- Pituitary tumor
- No tumor

The project includes experiments with and without data augmentation, quantitative evaluation using accuracy and F1-score, and qualitative model interpretability using **Grad-CAM**.

---

## Dataset

We use the **Brain Tumor Classification (MRI)** dataset from Kaggle:

https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

```
Dataset structure:
    data/
    ├── Training/
    │ ├── glioma_tumor/
    │ ├── meningioma_tumor/
    │ ├── no_tumor/
    │ └── pituitary_tumor/
    └── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

---

 **The dataset is not included in this repository**.  
Please download it from Kaggle and place it in a local `data/` folder.

---

## Model

- Backbone: **ResNet-50**
- Pretrained on ImageNet
- Final fully connected layer adapted for 4 classes
- Input size: `224 × 224`
- Loss: Cross-Entropy
- Optimizer: AdamW
- Normalization: ImageNet mean/std

---

## Pretrained Models

Pretrained model weights are available for download:

https://drive.google.com/drive/folders/1DYim2wMTdh1qwMDlxpv-pdnR_KAwLHff?usp=drive_link

Download the `.pt` files and place them in the `runs/` directory to reproduce evaluation and Grad-CAM results without retraining. The `runs` directory is not included so make this directroy in the root of the project. The csv files are required. If you want to train the model you will get the csv files.


---

## Experiments

Two training configurations were evaluated:

1. **No Augmentation**
   - Resize → Tensor → Normalize

2. **With Augmentation**
   - Random horizontal flip
   - Random rotation
   - Resize → Tensor → Normalize

Each model was trained for **15 epochs** with an 80/20 train-validation split from the Training set.  
Final evaluation was performed on the untouched Testing set.

Training metrics (loss and accuracy) are logged per epoch to CSV files for visualization.

---

## Evaluation Metrics

- Accuracy
- Precision / Recall / F1-score (per class)
- Macro and weighted F1
- Confusion matrix

The best model from each experiment is saved and evaluated on the test set.

---

## Model Interpretability (Grad-CAM)

Grad-CAM is used to visualize which regions of MRI images the model focuses on when making predictions.

- Grad-CAM is applied to the last convolutional layer of ResNet-50
- Both correct and misclassified examples are analyzed
- Visualizations are saved for inclusion in reports

---

## Project Structure

```
Brain-Tumor-Detection-and-Classification/
├── src/
│ ├── dataset.py # Dataset loading and transforms
│ ├── model.py # ResNet-50 model definition
│ └── train.py # Training and evaluation pipeline
│
├── notebooks/
│ ├── main.ipynb # Metrics, plots, evaluation
│ └── gradcam.ipynb # Grad-CAM visualizations
│
├── runs/
│ ├── best_noaug.pt
│ ├── best_aug.pt
│ ├── history_noaug.csv
│ └── history_aug.csv
│
├── reports/
│ └── figures/ # Saved Grad-CAM images
│
├── requirements.txt
├── README.md
└── .gitignore
```
---

---

##  How to Run

### 1. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

```
--- 

### Installing dependencies
```bash
pip install -r requirements.txt
```
---
# Authors

- Elmer Hernandez
- Enrrique Perez Alvarez