# Plant Disease Detection and Advisory System

Welcome to the **Plant Disease Detection** project! 🌿
This project is an end-to-end deep learning application that detects plant leaf diseases from a photo and gives treatment and prevention advice to farmers in English, Nepali and Hindi. It was built as my final-year project and is designed as a portfolio piece to show skills in deep learning, model deployment and web app development.

**Live Demo:** [Try the app here](https://plant-disease-detection-4e3l9fkwh9aj52qfydchgc.streamlit.app)

---

## Project Requirements

### 1. Model Development (Deep Learning)

#### Objectives
Develop an image classification model using transfer learning (EfficientNet-B4) to identify plant leaf diseases from a single photo, and to reject images that are not leaves.

#### Key Specifications
- **Data Source**: Custom dataset of 100,000+ leaf and non-leaf images on [Kaggle](https://www.kaggle.com/datasets/afrojkhan0220/leaf-nonleaf-image), covering crops such as apple, corn, grape, potato, tomato, rice, wheat and soybean.
- **Classes**: 61 total, made up of 60 plant conditions (diseases and healthy leaves) and 1 "NOT A LEAF" class.
- **Model**: EfficientNet-B4 with transfer learning, input size 380 x 380.
- **Performance**: 95.06% validation accuracy and 94.7% macro-average F1-score.
- **Training Strategy**: Three-phase transfer learning (frozen base layers first, then progressive unfreezing and fine-tuning with a reduced learning rate). Data augmentation (flip, rotation, zoom, brightness, contrast) was applied to underrepresented classes to handle class imbalance.
- **Scope**: Single-image classification with top-3 predictions and confidence scores.

---

### 2. Web Application and Advisory (Deployment)

#### Objectives
Deploy the trained model as an easy-to-use web app that helps farmers understand the problem and what to do next.

#### Key Specifications
- **Web App**: Built with Streamlit, with image upload and instant prediction.
- **Knowledge Base**: Information, treatment, prevention and advice for every class.
- **Multi-language Advisory**: English, Nepali and Hindi.
- **Model Hosting**: The trained model is downloaded automatically from Google Drive on first run.
- **Live Demo**: [Try the app here](https://plant-disease-detection-4e3l9fkwh9aj52qfydchgc.streamlit.app)

---

## Repository Structure
```
plant-disease-detection/
│
├── main.py                 # Streamlit app (UI, prediction, advisory)
├── class_names.json        # Class labels
├── requirements.txt        # Python dependencies
├── README.md               # Project overview
└── LICENSE                 # License information
```

---

## How to Run
```bash
git clone https://github.com/Afrojkhan-glitch/plant-disease-detection.git
cd plant-disease-detection
pip install -r requirements.txt
streamlit run main.py
```
On the first run the app downloads the trained model automatically.

### Download the Dataset
The dataset is over 1 GB, so it is not stored in this repository. You can download it from Kaggle:
```python
import kagglehub
path = kagglehub.dataset_download("afrojkhan0220/leaf-nonleaf-image")
print(path)
```

---

## License
This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and share this project with proper attribution.

---

## About Me
Hi there! I'm **Afroj Ahmad Khan**, an aspiring data analyst with a growing interest in data science and machine learning.
I enjoy building end-to-end projects, from cleaning data and writing SQL to training deep learning models and deploying them as web apps.

This project was my final-year project. It combines computer vision, model deployment and a multi-language advisory system to help farmers detect plant diseases early.
