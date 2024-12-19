# Plant-Leaf-Disease-Detection-System

This repository contains a project focused on detecting diseases in plant leaves using machine learning and image processing techniques. The goal is to help farmers and agriculturalists diagnose plant diseases accurately and take timely action to protect crops.  

## Features  
- **Image Preprocessing**: Includes resizing, normalization, and augmentation.  
- **Disease Detection**: Utilizes Convolutional Neural Networks (CNNs) for classification.  
- **Evaluation Metrics**: Provides accuracy, precision, recall, and F1-score.  
- **Visualization**: Graphs and confusion matrices to interpret results.  

---

## Project Structure  

```plaintext
Plant-Leaf-Disease-Detection/
│
├── README.md              # Project overview
├── LICENSE                # License information
├── requirements.txt       # Dependencies and libraries
├── dataset/               # Dataset used for the project
│   ├── train/             
│   └── test/
├── models/                # Saved machine learning models
├── src/                   # Source code
│   ├── preprocessing.py   # Image preprocessing
│   ├── model_training.py  # Model training
│   ├── predict.py         # Prediction script
│   └── utils.py           # Helper functions
├── notebooks/             # Jupyter notebooks
├── results/               # Output results, graphs, metrics
└── .gitignore             # Files and directories to ignore
