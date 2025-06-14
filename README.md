
# 🧠 Brain Tumor Classification using CNN & Streamlit

This project is a deep learning-based image classification system that detects brain tumors from MRI scans. It supports four categories:
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

It uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras and provides a clean Streamlit-based web interface for real-time predictions and dataset analysis.

> ℹ️ Added comments For better understanding

---

## 📁 Project Structure

```
BRAIN_TUMOR_PROJECT/
├── data/
│   ├── train/
│   └── test/
├── training.py              # Model training script
├── main.py                  # Streamlit web app
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── .gitignore               # Ignored files (models, logs, etc.)
```

---

## 🧠 Model Architecture

The model is a CNN with:
- Multiple `Conv2D` + `BatchNormalization` layers
- `MaxPooling`, `Flatten`, and `Dense` layers
- `Dropout` for regularization
- Trained with `EarlyStopping` and `ModelCheckpoint`

---

## 📦 Requirements

Install all dependencies via:

```bash
pip install -r requirements.txt
```

Main dependencies include:
- `tensorflow`
- `streamlit`
- `numpy`, `pandas`, `matplotlib`
- `Pillow`, `scikit-learn`

---

## 🚀 Usage

### 🔧 1. Train the Model

```bash
python training.py
```

This will:
- Load MRI images from `data/train/` and `data/test/`
- Train the CNN with image augmentation
- Save the best model as `best_model.keras`

### 🌐 2. Run the Web App

streamlit run main.py


### 🧪 3. Upload MRI Image

- Choose image enhancement (contrast/sharpness)
- Upload a `.jpg`, `.jpeg`, or `.png` MRI image
- View:
  - Predicted tumor type
  - Confidence score
  - Prediction confidence bar graph
  - Prediction pie chart
  - Model summary


### 📊 4. Dataset Insights

Click the **"Show Dataset Insights"** button to visualize class distributions in both train and test sets.

---

## 🗂️ Dataset Format

Organized as:

```
data/
├── train/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── test/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/



## 🤝 Acknowledgements

- Dataset sourced from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- Built using TensorFlow, Streamlit, and best practices in image classification


## ⭐️ Star the Repo

If you found this useful, consider giving it a ⭐ to show support!
