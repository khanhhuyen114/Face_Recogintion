# Facial Recognition using PCA

## Introduction
This repository contains a Python implementation of a basic Principal Component Analysis (PCA) module applied to facial recognition. The goal is to demonstrate how PCA can be used to reduce the dimensionality of facial image data, improving the efficiency and performance of facial recognition algorithms.

## Repository Structure
    ├── README.md          <- The top-level README for developers using this project.
    ├── att_faces # Dataset folder
    │ ├── s1 # Sample images for subject 1
    │ └── ... # Other subjects
    ├── main.py # Main script for facial recognition using PCA
    └── requirements.txt # Required Python libraries


## Dataset
The `att_faces` directory contains the facial images used in this project. Each subdirectory `s1`, `s2`, ..., `s40` represents a different subject. The README in the `att_faces` directory provides more information about the dataset.

## Installation
To set up the project environment:
1. Clone this repository:
   ```bash
   git clone https://github.com/vuxminhan/Principal-component-analysis.git
   cd Principal-component-analysis
   pip install -r requirements.txt
2. Execute code:
   ```bash
   python3 main.py
