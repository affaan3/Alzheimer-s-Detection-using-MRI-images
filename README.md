# 🧠 Alzheimer’s Disease MRI Classification

### Deep Learning Models for Multi-Class Alzheimer Detection

This repository contains three deep-learning models implemented in Jupyter Notebooks to classify MRI brain scans into four stages of Alzheimer’s Disease.
The included models are EfficientNet, VGG16 (Transfer Learning), and a Custom CNN.

---

## 📁 Project Structure

📦 Project Root
│
├── 🧠 Alzheimer_MRI_4_classes_dataset
│     ├── 🔴 MildDemented
│     ├── 🟠 ModerateDemented
│     ├── 🟢 NonDemented
│     └── 🟡 VeryMildDemented
│
├── 📓 notebooks
│     ├── 📘 EfficientNet.ipynb
│     ├── 🏛️ VGG16.IPNYB.ipynb
│     └── 🧩 alzheimer-detection.ipynb
│
├── 📄 README.md
├── 📄 requirements.txt
└── 📄 .gitignore

---

## 🚀 Models Included

### 1️⃣ EfficientNet-B0

• Lightweight, high accuracy
• Fast convergence with transfer learning

### 2️⃣ VGG16 (Transfer Learning)

• Pretrained on ImageNet
• Strong baseline for MRI classification

### 3️⃣ Custom CNN Model

• Built from scratch
• Convolution → MaxPooling → Dropout → Dense

---

## 🧠 Dataset Description

Dataset used: Alzheimer MRI 4-Classes Dataset
Contains four categories of dementia severity:

• 🟢 NonDemented
• 🟡 VeryMildDemented
• 🟠 MildDemented
• 🔴 ModerateDemented

Images are grayscale MRI brain scans categorized by clinical stages.

---

## 🧪 Training Pipeline

✔ Image resizing and normalization
✔ Data augmentation (flips, rotation, zoom, shift)
✔ Label encoding
✔ Train/validation split
✔ EarlyStopping + ModelCheckpoint
✔ Transfer learning for EfficientNet and VGG16

---

## 📈 Expected Performance

• Accuracy range: 90% – 94%
• EfficientNet provides best results
• VeryMildDemented class shows strongest recall

---

## ▶️ How to Run

1. Install dependencies (requirements.txt)
2. Launch Jupyter Notebook
3. Open any notebook inside the “notebooks” folder
4. Run all cells to train and evaluate the models

---

## 🔮 Future Improvements

• Add Grad-CAM visual explanations
• Convert models to TensorFlow Lite / ONNX
• Deploy with FastAPI or Streamlit
• Add Docker support
✅ .gitignore
Just tell me!
