"""
Module for making predictions with trained models.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import config

def predict_voltage(current_density, temperature, model_path, model_type='linear', scaler=None):
    """
    Predict cell voltage for given conditions.
    
    Parameters:
    -----------
    current_density : float or array-like
        Current density values (A/cm²)
    temperature : float or array-like
        Temperature values (K)
    model_path : str
        Path to the trained model
    model_type : str
        Type of model ('linear' or 'neural_network')
    scaler : StandardScaler, optional
        Scaler for neural network models
        
    Returns:
    --------
    predictions : array-like
        Predicted voltage values
    """
    # Load the model
    model = joblib.load(model_path)
    
    # Prepare input data
    if isinstance(current_density, (int, float)):
        current_density = [current_density]
    if isinstance(temperature, (int, float)):
        temperature = [temperature]
    
    input_data = pd.DataFrame({
        'Current_Density_A_cm2': current_density,
        'Temperature_K': temperature
    })
    
    # Make predictions
    if model_type == 'neural_network' and scaler is not None:
        input_scaled = scaler.transform(input_data)
        predictions = model.predict(input_scaled)
    else:
        predictions = model.predict(input_data)
    
    return predictions

def predict_multiple_conditions(conditions, model_path, model_type='linear', scaler=None):
    """
    Predict cell voltage for multiple conditions.
    
    Parameters:
    -----------
    conditions : list of tuples
        List of (current_density, temperature) pairs
    model_path : str
        Path to the trained model
    model_type : str
        Type of model ('linear' or 'neural_network')
    scaler : StandardScaler, optional
        Scaler for neural network models
        
    Returns:
    --------
    results : pandas.DataFrame
        DataFrame with inputs and predictions
    """
    current_densities = []
    temperatures = []
    
    for cd, temp in conditions:
        current_densities.append(cd)
        temperatures.append(temp)
    
    predictions = predict_voltage(
        current_densities, temperatures, model_path, model_type, scaler
    )
    
    results = pd.DataFrame({
        'Current_Density_A_cm2': current_densities,
        'Temperature_K': temperatures,
        'Predicted_Voltage_V': predictions
    })
    
    return results

if __name__ == "__main__":
    # Example usage
    conditions = [
        (0.2, 323),
        (0.8, 343),
        (1.2, 353)
    ]
    
    # Load scaler (for neural network)
    # Note: In a real application, you would need to save and load the scaler
    print("Prediction module loaded successfully!")