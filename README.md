# 👁️ Deep A-Eye

> **AI-Powered Diabetic Retinopathy Detection and Severity Classification using Ultra-Widefield Retinal Imaging**

**Graduation Project (2025–2026)**
College of Computer Science & Information Technology
Imam Abdulrahman Bin Faisal University

---

# Overview

Deep A-Eye is an AI-powered framework for the automated detection and severity classification of **Diabetic Retinopathy (DR)** from **Ultra-Widefield (UWF)** retinal images.

This repository contains the **Artificial Intelligence and Machine Learning** components of the project, including dataset preparation, model training, evaluation, explainability, and performance analysis.

The framework combines **Computer Vision**, **Deep Learning**, and **Explainable AI (XAI)** to classify retinal images into five clinical severity levels: No DR (Grade 0), Mild NPDR (Grade 1), Moderate NPDR (Grade 2), Severe NPDR (Grade 3), and Proliferative DR (Grade 4). It adopts a **late-fusion architecture** that integrates global retinal image features with lesion-level information and evaluates multiple deep learning models, including **ResNet-50** and **Swin Transformer (Swin-S)**.

> **Note:** The web application and deployment modules are maintained separately and will be integrated into this repository in a future update.

---

#  Features

* Automated diabetic retinopathy detection
* Five-level DR severity classification
* Late-fusion deep learning framework
* ResNet-50 and Swin Transformer models
* Grad-CAM explainability
* Attention map visualization
* Dataset preparation pipeline
* Model training pipeline
* Model evaluation and performance analysis

---

#  AI Models

| Model                     | Purpose                                               |
| ------------------------- | ----------------------------------------------------- |
| ResNet-50                 | Diabetic Retinopathy Classification                   |
| Swin Transformer (Swin-S) | Vision Transformer-based DR Classification            |
| Late Fusion               | Combines image features with lesion-level information |
| Grad-CAM                  | Model explainability                                  |

---

#  Results

| Model         |   Accuracy |   F1-score |       QWK |
| ------------- | ---------: | ---------: | --------: |
| Swin-S        |     86.50% |     84.12% |     95.18 |
| **ResNet-50** | **86.50%** | **84.26%** | **95.73** |

**ResNet-50 achieved the highest Quadratic Weighted Kappa (QWK), demonstrating the strongest agreement with clinical grading.**

---

# 💻 Technologies

### Programming

* Python

### Deep Learning

* PyTorch
* torchvision
* timm

### Computer Vision

* OpenCV
* Grad-CAM

### Hardware Acceleration

* Apple Silicon (MPS)

---

# 📂 Repository Structure

```text
Deep-A-Eye/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── checkpoints/
├── data/
├── datasets/
├── evaluation/
├── explainability/
├── models/
├── training/
└── testing/
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Deep-A-Eye.git
```

Navigate to the project:

```bash
cd Deep-A-Eye
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# 👥 Team

This project was developed by a team of six Computer Science students at **Imam Abdulrahman Bin Faisal University**.

| Member          | GitHub              |
| --------------- | ------------------- |
| Reham Alyami    | @rthecoolest        |
| Ethaar Alsolami | @ethaaralsolami-bot |
| Dhay Alqahtani  | @dhayii             |
| Noor Albaqshi   | @Noorbaqshi         |
| Rund Aloraifi   | @USERNAME           |
| Dhay Alomar     | @dyalmr3            |


---

# My Contributions

My primary contributions to the project included:

* Implemented the deep learning training and evaluation pipeline.
* Configured the project environment using Python, PyTorch, and Apple Silicon (MPS) acceleration.
* Integrated project dependencies and deep learning libraries.
* Contributed to the late-fusion and ensemble prediction workflow.
* Implemented Grad-CAM explainability and visualization outputs.
* Participated in model testing, debugging, and performance evaluation.

---

# Supervisors

* Ms. Hanoof Mohammed Algofari
* Dr. Atta-ur-Rahman

---

# License

This repository is provided for academic and educational purposes only as part of the Bachelor of Science in Computer Science program at Imam Abdulrahman Bin Faisal University.
