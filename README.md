# 😷 Face Mask Detection using Deep Learning

This project implements a **deep learning model** to detect whether a person is wearing a face mask or not in real-time. It uses **Convolutional Neural Networks (CNNs)** for image classification and can be integrated with a webcam feed for live detection.

---

## 📌 Features

* Detects **face with mask** 😷 and **face without mask** 🚫
* Works on both **images** and **real-time video streams**
* Uses a **CNN model** (can be built with TensorFlow/Keras or PyTorch)
* Provides **accuracy and loss metrics** during training
* Easy to extend with **new datasets**

---

## 📂 Project Structure

```
├── data/                  # Dataset (face with/without masks)
├── notebooks/             # Jupyter notebooks for experiments
├── src/                   # Source code
│   ├── dataset.py         # Dataset preprocessing
│   ├── model.py           # CNN architecture
│   ├── train.py           # Training script
│   ├── evaluate.py        # Model evaluation
│   ├── detect_mask.py     # Real-time face mask detection
│   └── utils.py           # Helper functions
├── results/               # Trained models, logs, and outputs
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/face-mask-detection.git
cd face-mask-detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare Dataset

* Download dataset (e.g., [Kaggle Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)).
* Place it inside the `data/` folder with structure:

```
data/
│── with_mask/
│── without_mask/
```

### 4️⃣ Train Model

```bash
python src/train.py --epochs 20 --batch_size 32
```

### 5️⃣ Evaluate Model

```bash
python src/evaluate.py --model results/mask_model.pth
```

### 6️⃣ Run Real-Time Detection

```bash
python src/detect_mask.py --model results/mask_model.pth
```

---

## 📊 Results

* Achieved **\~98% accuracy** on validation set.
* Real-time detection runs at **\~20 FPS** on CPU.
---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** TensorFlow / PyTorch, OpenCV, NumPy, Matplotlib
* **Dataset:** Custom/Kaggle datasets

---

## 📌 Applications

* Public safety in **airports, malls, and offices**
* **Smart surveillance systems**
* Healthcare & pandemic monitoring

---

## 🤝 Contributing

Contributions are welcome!

1. Fork this repo
2. Create your branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature-name`)
5. Open a Pull Request

---

## 🙌 Acknowledgements

* [Kaggle Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
* [OpenCV](https://opencv.org/) for face detection
* [TensorFlow](https://www.tensorflow.org/) / [PyTorch](https://pytorch.org/) for deep learning models

---
