"""
Main application for fuel cell performance prediction.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Import project modules
from src.data_generation import generate_synthetic_data, save_data, load_data
from src.model_training import prepare_data, train_linear_regression, train_neural_network, evaluate_model, save_model
from src.visualization import create_data_distribution_plots, create_prediction_comparison, create_3d_surface_plot, create_residual_plot
from src.prediction import predict_multiple_conditions

import config

def main():
    """Main function to run the complete workflow."""
    print("=" * 50)
    print("Fuel Cell Performance Prediction Project")
    print("=" * 50)
    
    # Step 1: Generate synthetic data
    print("\n1. Generating synthetic data...")
    data = generate_synthetic_data()
    save_data(data, config.PATHS['synthetic_data'])
    print(f"Generated {len(data)} samples")
    
    # Step 2: Explore the data
    print("\n2. Exploring data distributions...")
    create_data_distribution_plots(data, config.PATHS['visualizations'])
    
    # Step 3: Prepare data for machine learning
    print("\n3. Preparing data for machine learning...")
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(data)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Step 4: Train models
    print("\n4. Training machine learning models...")
    print("Training Linear Regression model...")
    lr_model = train_linear_regression(X_train, y_train)
    
    print("Training Neural Network model...")
    nn_model = train_neural_network(X_train_scaled, y_train)
    
    # Step 5: Evaluate models
    print("\n5. Evaluating model performance...")
    lr_metrics, lr_pred = evaluate_model(lr_model, X_test, y_test, model_type='linear')
    nn_metrics, nn_pred = evaluate_model(nn_model, X_test, y_test, scaler, model_type='neural_network')
    
    print("\nLinear Regression Performance:")
    for metric, value in lr_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    print("\nNeural Network Performance:")
    for metric, value in nn_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # Step 6: Create visualizations
    print("\n6. Creating visualizations...")
    create_prediction_comparison(
        y_test, lr_pred, nn_pred, 
        ['Linear Regression', 'Neural Network'],
        config.PATHS['visualizations']
    )
    
    create_3d_surface_plot(
        lr_model, scaler, 'linear',
        config.PATHS['visualizations']
    )
    
    create_3d_surface_plot(
        nn_model, scaler, 'neural_network',
        config.PATHS['visualizations']
    )
    
    create_residual_plot(
        y_test, lr_pred, 'Linear Regression',
        config.PATHS['visualizations']
    )
    
    create_residual_plot(
        y_test, nn_pred, 'Neural Network',
        config.PATHS['visualizations']
    )
    
    # Step 7: Save models
    print("\n7. Saving trained models...")
    save_model(lr_model, config.PATHS['linear_model'])
    save_model(nn_model, config.PATHS['nn_model'])
    
    # Step 8: Demonstrate predictions
    print("\n8. Making predictions for new conditions...")
    new_conditions = [
        (0.2, 323),   # Low current, moderate temperature
        (0.8, 343),   # Medium current, high temperature
        (1.2, 353),   # High current, very high temperature
        (0.5, 313),   # Medium current, low temperature
    ]
    
    # Predict with both models
    lr_results = predict_multiple_conditions(
        new_conditions, config.PATHS['linear_model'], 'linear'
    )
    
    nn_results = predict_multiple_conditions(
        new_conditions, config.PATHS['nn_model'], 'neural_network', scaler
    )
    
    print("\nLinear Regression Predictions:")
    print(lr_results.to_string(index=False))
    
    print("\nNeural Network Predictions:")
    print(nn_results.to_string(index=False))
    
    print("\n" + "=" * 50)
    print("Project completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()