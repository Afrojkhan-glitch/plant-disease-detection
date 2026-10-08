# Plant Disease Detection and Advisory System

Welcome to the **Plant Disease Detection** project! 🌿
This project is an end-to-end deep learning application that detects plant leaf diseases from a photo and gives treatment and prevention advice to farmers in English, Nepali and Hindi. It was built as my final-year project and is designed as a portfolio piece to show skills in deep learning, model deployment and web app development.

---

## Project Requirements

### 1. Model Development (Deep Learning)

#### Objectives
Develop an image classification model using transfer learning (EfficientNet-B4) to identify plant leaf diseases from a single photo, and to reject images that are not leaves.

#### Key Specifications
- **Data Source**: Plant leaf image datasets covering crops such as apple, corn, grape, potato, tomato, rice, wheat and soybean ([add dataset names and links]).
- **Classes**: 61 total, made up of 60 plant conditions (diseases and healthy leaves) and 1 "NOT A LEAF" class.
- **Model**: EfficientNet-B4 with transfer learning, input size 380 x 380.
- **Performance**: XX% accuracy on the test set.
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
- **Live Demo**: [add your Streamlit link here]

---

## Repository Structure
```
plant-disease-detection/
│
├── main.py                 # Streamlit app (UI, prediction, advisory)
├── class_names.json        # Class labels
├── requirements.txt        # Python dependencies
├── README.md               # Project overview
```

---

## How to Run
```bash
git clone https://github.com/Afrojkhan-glitch/plant-disease-detection.git
cd plant-disease-detection
pip install -r requirements.txt
streamlit run main.py
```
The dataset is over 1 GB, so it is not stored in this repository. On the first run the app downloads the trained model automatically.

## About Me
Hi there! I'm **Afroj Khan**, an aspiring data science and machine learning  with a growing interest in data analyst .
I enjoy building end-to-end projects, from cleaning data and writing SQL to training deep learning models and deploying them as web apps.
This project was my final-year project. It combines computer vision, model deployment and a multi-language advisory system 
to help farmers detect plant diseases early.
