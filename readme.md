

# Lung and Colon Cancer Histopathological Image Classification

Official implementation of the research article: **"Leveraging Dual Transfer Learning and Attention-based Pooling for Lung and Colon Cancer Histopathological Image Classification"**.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()

---

## 📝 Abstract

Lung cancer remains a dominant cause of cancer mortality globally. This repository presents a novel hybrid architecture designed to improve lung and colon cancer diagnosis through enhanced classification of histopathological images. The framework combines **Dual Transfer Learning** with **Neighbor Feature Attention-based Pooling (NFP)** to address the challenges of medical image processing. By leveraging the complementary strengths of DenseNet-169 and InceptionResNet101-V2, the proposed method achieves an accuracy of **96.3%** on the LC25000 dataset, surpassing current state-of-the-art methods.

---

## 🚀 Key Features

*   **Dual Transfer Learning**: Utilizes DenseNet-169 and InceptionResNet101-V2 in parallel to capture complementary features—feature propagation (DenseNet) and multi-scale representations (Inception).
*   **Neighbor Feature Attention-based Pooling (NFP)**: Replaces standard Global Average Pooling to preserve spatial dependencies and tissue context, which are critical for accurate histopathological analysis.
*   **Feature-level Fusion**: Concatenates high-level features from both backbones to create a robust representation for classification.
*   **High Performance**: Achieves state-of-the-art results on the LC25000 dataset.

---

## 🛠️ Installation

Follow these steps to set up the environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/LungColonCancer_Classification.git
    cd LungColonCancer_Classification
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 📂 Dataset Preparation

This project uses the **LC25000 Dataset** (Lung and Colon Histopathological Images).

1.  Download the dataset from the [official source](https://github.com/tampapath/lung_colon_image_set).
2.  Extract the contents.
3.  Place the `lung_colon_image_set` folder inside the `data/` directory.

The final directory structure should look like this:

```
LungColonCancer_Classification/
│
├── data/
│   └── lung_colon_image_set/
│       ├── lung_aca       # Lung Adenocarcinoma
│       ├── lung_n         # Lung Benign
│       ├── lung_scc       # Lung Squamous Cell Carcinoma
│       ├── colon_aca      # Colon Adenocarcinoma
│       └── colon_n        # Colon Benign
│
├── models/                # Neural Network Architectures
├── utils/                 # Helper functions
├── train.py               # Main Training Script
├── evaluate.py            # Evaluation Script
└── requirements.txt
```

---

## 🏋️ Training

To train the model from scratch using the parameters described in the manuscript (Batch size=32, LR=0.001, Epochs=50):

```bash
python train.py
```

**Training Features:**
*   **Automatic Splitting**: The dataset is split into 80% Train, 10% Validation, and 10% Test.
*   **Augmentation**: Applies Random Rotation, Flipping, Zoom, and Brightness adjustments.
*   **TensorBoard Logging**: Metrics are logged to the `runs/` directory. To visualize training progress, run:
    ```bash
    tensorboard --logdir=runs
    ```
*   **Best Model Saving**: The model weights with the highest validation accuracy are saved automatically as `best_model.pth`.

> **⚠️ Hardware Note:** The manuscript utilized an **NVIDIA RTX 3090 (24GB VRAM)** with an image size of 768×768. If you have limited VRAM, please reduce the `BATCH_SIZE` in `train.py` (e.g., to 16 or 8) to avoid CUDA Out of Memory errors.

---

## 🧪 Evaluation

To evaluate the trained model and generate performance metrics:

```bash
python evaluate.py
```

**Outputs:**
The script will generate and save the following results in the `results/` directory:
*   `results/confusion_matrix.png`
*   `results/roc_curve.png`
*   Classification Report (Accuracy, Precision, Recall, F1-Score).

---

## 🙏 Acknowledgements

The LC25000 Dataset creators:

> Borkowski, A. A., Bui, M. M., Thomas, L. B., Wilson, C. P., DeLand, L. A., & Mastorides, S. M. (2019). Lung and colon cancer histopathological image dataset (lc25000). arXiv preprint arXiv:1912.12142.

**Dataset Link:** [https://github.com/tampapath/lung_colon_image_set](https://github.com/tampapath/lung_colon_image_set)

---
```
```
