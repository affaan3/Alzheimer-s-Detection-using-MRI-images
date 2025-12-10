# **🧠 Alzheimer’s Disease MRI Classification Using Deep Learning**

A complete **MRI-based Alzheimer’s Disease classification system** built with **CNNs, Transfer Learning**, and **advanced preprocessing techniques**.
This project trains three different deep-learning models to classify brain MRI scans into four stages of Alzheimer’s:

* **NonDemented**
* **VeryMildDemented**
* **MildDemented**
* **ModerateDemented**

---

## **✨ Features**

* ✔️ Three separate deep-learning models
  — **EfficientNet-B0**, **VGG16**, **Custom CNN**
* ✔️ Full preprocessing pipeline (resize, normalization, augmentation)
* ✔️ 4-class softmax classification
* ✔️ Training graphs: accuracy, loss, evaluation metrics
* ✔️ Supports transfer learning & fine-tuning
* ✔️ Works directly with MRI grayscale images
* ✔️ Modular notebook-based workflow

---

## **📁 Project Structure**

```
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
```

---

## **🛠️ Requirements**

Install dependencies before running the notebooks:

```bash
pip install tensorflow keras numpy pandas matplotlib opencv-python seaborn scikit-learn jupyter
```

Optional (if using PyTorch instead):

```bash
pip install torch torchvision torchaudio
```

---

## **🚀 How It Works**

### **1. Dataset Loading**

MRI images are loaded from four folders representing disease stages.

### **2. Image Preprocessing**

Includes:

* Resizing to 224×224
* Grayscale normalization
* Data Augmentation
* One-hot label encoding

### **3. Model Training**

The notebooks train three different models:

#### **EfficientNet-B0**

Transfer learning + fine-tuning for best accuracy.

#### **VGG16**

ImageNet pretrained backbone + custom dense layers.

#### **Custom CNN**

Built from scratch including:

* Conv2D
* MaxPooling2D
* Dropout
* Dense softmax layer

### **4. Evaluation Metrics**

Includes:

* Accuracy
* Loss curves
* Confusion matrix
* Precision, Recall, F1-score
* ROC-AUC

---

## **📊 Output**

Each notebook generates:

| Output                        | Description                     |
| ----------------------------- | ------------------------------- |
| Training Accuracy/Loss Graphs | Model performance visualization |
| Confusion Matrix              | Class-wise evaluation           |
| Classification Report         | Precision/Recall/F1 Score       |
| Saved Model (`.h5`)           | Optional model export           |

---

## **▶️ How to Run**

1. Open the project folder
2. Install dependencies
3. Launch Jupyter Notebook:

```bash
jupyter notebook
```

4. Open any model file inside `notebooks/`:

* EfficientNet.ipynb
* VGG16.IPNYB.ipynb
* alzheimer-detection.ipynb

5. Run all cells to start training.

---

## **🧩 Customization**

### **Change input size**

Update:

```python
img_size = (224, 224)
```

### **Change model backbone**

For EfficientNet:

```python
EfficientNetB3, EfficientNetV2B0
```

For VGG:

```python
VGG19
```

### **Add more augmentation**

Add to `ImageDataGenerator`.

---

## **📌 Notes**

* Dataset must be placed exactly as shown in the folder structure.
* GPU recommended for EfficientNet training.
* For best results, use at least 20–25 epochs per model.
