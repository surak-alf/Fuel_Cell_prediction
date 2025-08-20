"""
Configuration settings for the fuel cell prediction project.
"""

# Data generation parameters
DATA_CONFIG = {
    'n_samples': 2000,
    'current_density_range': (0, 1.5),  # A/cm²
    'temperature_range': (303, 363),    # K (30°C to 90°C)
    'noise_std': 0.02,                  # Standard deviation of noise (V)
    'test_size': 0.2,                   # Proportion of data for testing
    'random_state': 42,                 # For reproducibility
}

# Model parameters
MODEL_CONFIG = {
    'linear_regression': {
        'fit_intercept': True,
    },
    'neural_network': {
        'hidden_layer_sizes': (10, 5),
        'activation': 'relu',
        'solver': 'adam',
        'max_iter': 2000,
        'random_state': 42,
        'early_stopping': True,
    }
}

# Visualization parameters
VIS_CONFIG = {
    'figsize': (10, 6),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
}

# File paths
PATHS = {
    'synthetic_data': 'data/synthetic_data.csv',
    'linear_model': 'models/trained_linear_model.pkl',
    'nn_model': 'models/trained_nn_model.pkl',
    'visualizations': 'visualizations/'
}