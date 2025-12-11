A Non-Invasive Blood Glucose Monitoring System Based on Smartphone PPG Signals
1. Introduction

This project implements a deep-learning-based framework for estimating physiological parameters from PPG (Photoplethysmography) signals collected using a smartphone camera. Although the current experiment uses systolic blood pressure as the target variable, the methodology and system structure are designed to be extendable to non-invasive blood glucose estimation when paired glucose–PPG datasets are available.

2. Project Objectives

To extract time-domain and morphological features from smartphone-based PPG signals.

To evaluate the feasibility of LSTM-based regression models for physiological parameter estimation.

To build a prototype system that demonstrates the workflow of non-invasive monitoring based on PPG signals.

To provide a reproducible experimental pipeline for further research and extension to blood glucose estimation.

3. File Structure
.
├── lstm_dropt_predict.py        # Main training script using PPG features and LSTM regression
├── ppg_params.mat               # PPG feature dataset (version 1)
├── ppg_params_new.mat           # PPG feature dataset (version 2)
├── systolicpres.mat             # Systolic pressure labels
├── systolicpres_new.mat         # Alternative systolic pressure labels
└── README.md

4. Method Overview
4.1 Feature Extraction

The .mat files contain precomputed PPG features, such as:

Peak amplitude

Pulse intervals

Morphological descriptors

Rise time and decay time

These features serve as model inputs.

4.2 Data Normalization

The project uses:

Z-score normalization

Min-max scaling

A correlation heatmap is included to analyze feature relationships.

4.3 Model Architecture

The current model is an LSTM-based regression network:

LSTM layer (32 units)

Dropout layer (0.1)

Dense output layer (1 unit)

Loss: Mean Squared Error

Optimizer: Adam

Training runs for 1000 epochs with batch size 100.

4.4 Evaluation

The model outputs:

RMSE for training and testing sets

R² coefficient

Prediction vs. ground truth plots

These results provide insight into model fit and generalization.

5. How to Run
Step 1 — Install Dependencies
pip install numpy scipy pandas seaborn matplotlib scikit-learn keras heartpy
Step 2 — Ensure .mat Files Are Placed in Root Directory

The script loads:

ppg_params.mat

systolicpres.mat

Step 3 — Run the Main Script
python lstm_dropt_predict.py
The script will execute the full pipeline: data loading, normalization, model training, evaluation, and visualization.

6. Experimental Notes

The current target variable is systolic blood pressure.

The same framework can be adapted to blood glucose estimation when appropriate datasets are available.

LSTM is used to test time-step-based modeling, although feature-based regression models (e.g., MLP, Random Forest) may also be considered.

7. Potential Improvements

Incorporate additional PPG morphological and frequency-domain features.

Introduce alternative models such as 1D-CNN, GRU, or Transformer-based regressors.

Add validation sets and early stopping strategies.

Deploy the model to mobile platforms for real-time estimation.

Acquire paired PPG–blood glucose datasets for actual glucose regression.

8. Contact

Maintainer: Shiqi Fang
Email: fsq136656@gmail.com
