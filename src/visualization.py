"""
Module for creating visualizations of the fuel cell data and predictions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import os
import config

def setup_visualization():
    """Set up matplotlib style and parameters."""
    plt.style.use(config.VIS_CONFIG['style'])
    plt.rcParams['figure.figsize'] = config.VIS_CONFIG['figsize']
    plt.rcParams['figure.dpi'] = config.VIS_CONFIG['dpi']
    plt.rcParams['savefig.bbox'] = 'tight'

def create_data_distribution_plots(data, save_path=None):
    """
    Create distribution plots for the dataset.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    save_path : str, optional
        Path to save the plots
    """
    setup_visualization()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Current density distribution
    axes[0, 0].hist(data['Current_Density_A_cm2'], bins=30, 
                   color=config.VIS_CONFIG['colors'][0], alpha=0.7)
    axes[0, 0].set_xlabel('Current Density (A/cm²)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Current Density Distribution')
    
    # Temperature distribution
    axes[0, 1].hist(data['Temperature_K'], bins=30, 
                   color=config.VIS_CONFIG['colors'][1], alpha=0.7)
    axes[0, 1].set_xlabel('Temperature (K)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Temperature Distribution')
    
    # Voltage distribution
    axes[1, 0].hist(data['Cell_Voltage_V'], bins=30, 
                   color=config.VIS_CONFIG['colors'][2], alpha=0.7)
    axes[1, 0].set_xlabel('Cell Voltage (V)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Cell Voltage Distribution')
    
    # Current density vs voltage
    axes[1, 1].scatter(data['Current_Density_A_cm2'], data['Cell_Voltage_V'], 
                      alpha=0.6, color=config.VIS_CONFIG['colors'][3])
    axes[1, 1].set_xlabel('Current Density (A/cm²)')
    axes[1, 1].set_ylabel('Cell Voltage (V)')
    axes[1, 1].set_title('Current Density vs. Cell Voltage')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'data_distributions.png'))
    plt.show()

def create_prediction_comparison(y_true, y_pred_lr, y_pred_nn, model_names, save_path=None):
    """
    Create comparison plots for model predictions.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred_lr : array-like
        Linear regression predictions
    y_pred_nn : array-like
        Neural network predictions
    model_names : list
        Names of the models
    save_path : str, optional
        Path to save the plots
    """
    setup_visualization()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot for Linear Regression
    axes[0].scatter(y_true, y_pred_lr, alpha=0.6, color=config.VIS_CONFIG['colors'][0])
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Voltage (V)')
    axes[0].set_ylabel('Predicted Voltage (V)')
    axes[0].set_title(f'{model_names[0]}: Actual vs Predicted')
    
    # Plot for Neural Network
    axes[1].scatter(y_true, y_pred_nn, alpha=0.6, color=config.VIS_CONFIG['colors'][1])
    axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Voltage (V)')
    axes[1].set_ylabel('Predicted Voltage (V)')
    axes[1].set_title(f'{model_names[1]}: Actual vs Predicted')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'prediction_comparison.png'))
    plt.show()

def create_3d_surface_plot(model, scaler, model_type, save_path=None):
    """
    Create a 3D surface plot of model predictions.
    
    Parameters:
    -----------
    model : trained model
        Model to visualize
    scaler : StandardScaler
        Scaler used for preprocessing
    model_type : str
        Type of model ('linear' or 'neural_network')
    save_path : str, optional
        Path to save the plot
    """
    setup_visualization()
    
    # Generate a grid of values for visualization
    j_grid = np.linspace(0, 1.5, 20)
    t_grid = np.linspace(303, 363, 20)
    J, T = np.meshgrid(j_grid, t_grid)
    
    # Prepare grid for prediction
    grid_data = pd.DataFrame({
        'Current_Density_A_cm2': J.ravel(),
        'Temperature_K': T.ravel()
    })
    
    # Predict using the model
    if model_type == 'neural_network':
        grid_scaled = scaler.transform(grid_data)
        pred = model.predict(grid_scaled)
    else:
        pred = model.predict(grid_data)
    
    pred_grid = pred.reshape(J.shape)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(J, T, pred_grid, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Current Density (A/cm²)')
    ax.set_ylabel('Temperature (K)')
    ax.set_zlabel('Cell Voltage (V)')
    ax.set_title(f'{model_type.title()} Model Predictions')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    if save_path:
        plt.savefig(os.path.join(save_path, f'3d_surface_{model_type}.png'))
    plt.show()

def create_residual_plot(y_true, y_pred, model_name, save_path=None):
    """
    Create a residual plot for model evaluation.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the plot
    """
    setup_visualization()
    
    residuals = y_true - y_pred
    
    plt.scatter(y_pred, residuals, alpha=0.6, color=config.VIS_CONFIG['colors'][4])
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot for {model_name}')
    
    if save_path:
        plt.savefig(os.path.join(save_path, f'residuals_{model_name.lower().replace(" ", "_")}.png'))
    plt.show()

if __name__ == "__main__":
    # Example usage
    print("Visualization module loaded successfully!")