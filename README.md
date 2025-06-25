# Deep-Learning-with-Keras-and-Tensorflow
The Labs folder contains all of the labs for course 3/13 in my IBM professional certificate titled Deep learning with Keras and Tensorflow

# 🗑️ Waste Classification Using Transfer Learning (VGG16)

This project applies **transfer learning with VGG16** to classify waste products as either recyclable (`R`) or organic (`O`). The project involves feature extraction, data augmentation, fine-tuning, and model evaluation using deep learning techniques. Two models are trained: one using **frozen base features**, and the other using a **fine-tuned VGG16** backbone.

---

## 📂 Dataset

The dataset consists of pre-split images of recyclable and organic waste, provided in a zipped format:

- Training set: `o-vs-r-split/train/`
- Test set: `o-vs-r-split/test/`

Each image belongs to one of two folders: `R` (recyclable) or `O` (organic).

---

## 🛠️ Key Features

- 📥 **Automated download and extraction** of the dataset with a progress bar  
- ⚙️ **Image augmentation** using Keras' `ImageDataGenerator`
- 🧠 **Transfer learning** using VGG16 pretrained on ImageNet
- 📉 Training with **learning rate decay**, **early stopping**, and **model checkpointing**
- 🏗️ Model comparison: Feature extractor model vs. Fine-tuned model
- 📊 Evaluation with classification report, accuracy/loss curves, and confusion matrix
- 🖼️ Visualization of predictions with actual vs. predicted labels on test images

---

## 🧪 Models Trained

### 1. **Extract Features Model**
- VGG16 used as a frozen feature extractor
- Custom classification head added
- Trained for 10 epochs with image augmentation

### 2. **Fine-Tuned Model**
- Same architecture as above, but selectively unfreezes deeper VGG16 layers (starting from `block5_conv3`)
- Fine-tuned on the same training data with a smaller learning rate

---

## 🧠 Libraries Used

- Python, TensorFlow, Keras
- Matplotlib, Seaborn
- scikit-learn (for evaluation metrics)
- tqdm (for progress bar)
