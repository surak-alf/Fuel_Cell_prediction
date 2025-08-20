# Fuel Cell Performance Prediction

This project demonstrates data-driven performance prediction for fuel cells using machine learning. It generates synthetic data based on a physics-based polarization curve model and trains both linear regression and neural network models to predict cell voltage from current density and temperature.

## Project Structure
``````
fuel-cell-prediction/
├── data/ # Data directory
│ ├── raw/ # Raw data (empty in this project)
│ ├── processed/ # Processed data (empty in this project)
│ └── synthetic_data.csv # Generated synthetic data
├── models/ # Trained model files
├── notebooks/ # Jupyter notebooks for exploration
├── src/ # Source code
│ ├── data_generation.py # Synthetic data generation
│ ├── model_training.py # Model training and evaluation
│ ├── visualization.py # Data and result visualization
│ └── prediction.py # Prediction functions
├── config.py # Configuration settings
├── requirements.txt # Python dependencies
├── main.py # Main application
└── README.md # Project documentation
``````


## Installation

1. Clone or download this project
2. Install the required dependencies:

pip install -r requirements.txt
