"""
Module for training machine learning models on fuel cell data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import config

def prepare_data(data):
    """
    Prepare data for machine learning.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data with features and target
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Split and scaled data
    scaler : StandardScaler
        Fitted scaler for later use
    """
    # Separate features and target
    X = data[['Current_Density_A_cm2', 'Temperature_K']]
    y = data['Cell_Voltage_V']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.DATA_CONFIG['test_size'], 
        random_state=config.DATA_CONFIG['random_state']
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_linear_regression(X_train, y_train):
    """
    Train a linear regression model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
        
    Returns:
    --------
    model : LinearRegression
        Trained linear regression model
    """
    model = LinearRegression(**config.MODEL_CONFIG['linear_regression'])
    model.fit(X_train, y_train)
    return model

def train_neural_network(X_train_scaled, y_train):
    """
    Train a neural network model.
    
    Parameters:
    -----------
    X_train_scaled : array-like
        Scaled training features
    y_train : array-like
        Training target
        
    Returns:
    --------
    model : MLPRegressor
        Trained neural network model
    """
    model = MLPRegressor(**config.MODEL_CONFIG['neural_network'])
    model.fit(X_train_scaled, y_train)
    return model

def evaluate_model(model, X, y, scaler=None, model_type='linear'):
    """
    Evaluate a model's performance.
    
    Parameters:
    -----------
    model : trained model
        The model to evaluate
    X : array-like
        Features for evaluation
    y : array-like
        True target values
    scaler : StandardScaler, optional
        Scaler for neural network models
    model_type : str
        Type of model ('linear' or 'neural_network')
        
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics
    y_pred : array-like
        Model predictions
    """
    # Make predictions
    if model_type == 'neural_network' and scaler is not None:
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
    else:
        y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mse)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    return metrics, y_pred

def save_model(model, filepath):
    """
    Save a trained model to disk.
    
    Parameters:
    -----------
    model : trained model
        Model to save
    filepath : str
        Path to save the model
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to load the model from
        
    Returns:
    --------
    model : trained model
        Loaded model
    """
    return joblib.load(filepath)

if __name__ == "__main__":
    # Load data
    data = pd.read_csv(config.PATHS['synthetic_data'])
    
    # Prepare data
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(data)
    
    # Train models
    print("Training Linear Regression model...")
    lr_model = train_linear_regression(X_train, y_train)
    
    print("Training Neural Network model...")
    nn_model = train_neural_network(X_train_scaled, y_train)
    
    # Evaluate models
    lr_metrics, lr_pred = evaluate_model(lr_model, X_test, y_test, model_type='linear')
    nn_metrics, nn_pred = evaluate_model(nn_model, X_test, y_test, scaler, model_type='neural_network')
    
    # Print results
    print("\nLinear Regression Performance:")
    for metric, value in lr_metrics.items():
        print(f"{metric}: {value:.6f}")
    
    print("\nNeural Network Performance:")
    for metric, value in nn_metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # Save models
    save_model(lr_model, config.PATHS['linear_model'])
    save_model(nn_model, config.PATHS['nn_model'])
    
    print("\nModel training completed successfully!")