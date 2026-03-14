# 🩺 X-Ray Pneumonia Detection using EfficientNet (PyTorch)

A deep learning project for detecting **Pneumonia from Chest X‑ray images** using **EfficientNet‑B0 with Transfer Learning in PyTorch**.

This project demonstrates how **Computer Vision and Deep Learning can assist medical diagnosis** by automatically classifying chest X‑rays.

---

# 📌 Project Overview

Pneumonia is a serious lung infection that can be detected through **Chest X‑ray imaging**. In this project we build a **deep learning model** that can automatically classify X‑ray images into:

- **NORMAL**
- **PNEUMONIA**

The model uses **EfficientNet‑B0**, a modern convolutional neural network architecture optimized for **accuracy and computational efficiency**.

---

# 🧠 Model Architecture

The architecture uses **transfer learning** from EfficientNet.

```
Input X‑Ray Image (224x224)
        │
        ▼
EfficientNet-B0 Feature Extractor
(MBConv Blocks + Squeeze-Excitation)
        │
        ▼
Global Average Pooling
        │
        ▼
Dropout (0.2)
        │
        ▼
Fully Connected Layer
(1280 → 2)
        │
        ▼
Output Classes
NORMAL / PNEUMONIA
```

---

# 🔬 Key Concepts Used

## Transfer Learning
EfficientNet was pre-trained on **ImageNet (1M+ images)** and reused as a **feature extractor**. Only the final classifier layer was modified for our task.

## EfficientNet
EfficientNet uses **compound scaling** to balance:

- Network depth
- Network width
- Input resolution

This allows high accuracy with fewer parameters.

## CrossEntropyLoss
Used for classification tasks. It internally applies **LogSoftmax**, so we **do not manually apply Softmax during training**.

## Adam Optimizer
Adaptive learning rate optimizer that helps models converge faster.

---

# 📊 Model Performance

| Metric | Value |
|------|------|
| Training Epochs | 5 |
| Validation Accuracy | ~93% |
| Framework | PyTorch |
| Backbone | EfficientNet‑B0 |

---

# 📂 Project Structure

```
xray-pneumonia-detection
│
├── train.py
├── predict.py
├── requirements.txt
├── pneumonia_model.pth
├── README.md
├── LICENSE
└── .gitignore
```

Description:

- **train.py** → training pipeline
- **predict.py** → single image inference
- **requirements.txt** → dependencies
- **pneumonia_model.pth** → trained model weights

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Aquib78/xray-pneumonia-detection.git
cd xray-pneumonia-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch torchvision
pip install scikit-learn pillow matplotlib streamlit
```

---

# 🗂 Dataset

Dataset used:

**Chest X-Ray Pneumonia Dataset**

Kaggle link:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Dataset structure:

```
data/
   train/
      NORMAL/
      PNEUMONIA/
   val/
      NORMAL/
      PNEUMONIA/
```

⚠️ The dataset is **not included in this repository due to size limitations**.

---

# 🏋️ Training the Model

Run the training script:

```bash
python train.py
```

Training process:

1. Load dataset
2. Apply image transformations
3. Train EfficientNet-B0
4. Calculate validation accuracy
5. Save trained model

The trained model will be saved as:

```
pneumonia_model.pth
```

---

# 🔍 Running Predictions

To test the model on a single X‑ray image:

```bash
python predict.py
```

Example output:

```
Prediction: Pneumonia
Confidence: 0.92
```

---

# 🚀 Future Improvements

Phase‑2 of this project will include:

- Multi‑disease X‑ray classification
- Grad‑CAM visualization
- Streamlit Web App for deployment
- Larger medical datasets
- Model explainability

---

# 🛠 Technologies Used

- Python
- PyTorch
- Torchvision
- EfficientNet
- Streamlit
- Computer Vision
- Deep Learning

---

# 👨‍💻 Author

**Aquib Hussain**

GitHub:

https://github.com/Aquib78

---

# 📜 License

This project is licensed under the **MIT License**.

---

# ⭐ If you like this project

Please consider **starring the repository** ⭐
