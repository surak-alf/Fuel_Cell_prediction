"""
Module for generating synthetic fuel cell data.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import config

def polarization_curve(current_density, temperature, E0=1.2, R=0.2, alpha=0.5, beta=0.1):
    """
    Simplified polarization curve model for a fuel cell
    
    Parameters:
    -----------
    current_density : array-like
        Current density values (A/cm²)
    temperature : array-like
        Temperature values (K)
    E0 : float, optional
        Open circuit voltage (V)
    R : float, optional
        Ohmic resistance (Ω·cm²)
    alpha : float, optional
        Activation polarization parameter
    beta : float, optional
        Concentration polarization parameter
        
    Returns:
    --------
    voltage : array-like
        Calculated cell voltage (V)
    """
    # Temperature effect on performance (simplified)
    temp_effect = 1 + 0.005 * (temperature - 333)  # 333K reference temperature
    
    # Calculate voltage components
    activation_loss = alpha * np.log(current_density + 1e-10)  # Avoid log(0)
    ohmic_loss = R * current_density
    concentration_loss = beta * current_density**2
    
    # Total voltage
    voltage = E0 * temp_effect - activation_loss - ohmic_loss - concentration_loss
    
    return voltage

def generate_synthetic_data():
    """
    Generate synthetic fuel cell data with noise.
    
    Returns:
    --------
    data : pandas.DataFrame
        Synthetic dataset with current density, temperature and voltage
    """
    # Get configuration
    cfg = config.DATA_CONFIG
    
    # Generate random data points
    np.random.seed(cfg['random_state'])
    current_density = np.random.uniform(*cfg['current_density_range'], cfg['n_samples'])
    temperature = np.random.uniform(*cfg['temperature_range'], cfg['n_samples'])
    
    # Calculate cell voltage using the polarization curve model
    cell_voltage = polarization_curve(current_density, temperature)
    
    # Add random noise to simulate measurement variability
    noise = np.random.normal(0, cfg['noise_std'], cfg['n_samples'])
    cell_voltage += noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'Current_Density_A_cm2': current_density,
        'Temperature_K': temperature,
        'Cell_Voltage_V': cell_voltage
    })
    
    return data

def save_data(data, filepath):
    """
    Save generated data to CSV file.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to save
    filepath : str
        Path to save the data
    """
    data.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

def load_data(filepath):
    """
    Load data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to load the data from
        
    Returns:
    --------
    data : pandas.DataFrame
        Loaded data
    """
    return pd.read_csv(filepath)

if __name__ == "__main__":
    # Generate and save data when run as a script
    data = generate_synthetic_data()
    save_data(data, config.PATHS['synthetic_data'])
    print("Data generation completed successfully!")