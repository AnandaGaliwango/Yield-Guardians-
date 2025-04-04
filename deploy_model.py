#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import os

# Define the Student Model (must match the architecture used during training)
class StudentModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Class for model deployment
class CropYieldPredictor:
    def __init__(self, model_path='student_crop_yield_model.joblib'):
        """
        Initialize the predictor with a pre-trained model.
        
        Args:
            model_path (str): Path to the trained model file
        """
        # Load the trained model
        self.model = joblib.load(model_path)
        self.model.eval()  # Set the model to evaluation mode
        
        # Initialize preprocessing components
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        self.trained_on_features = ["CROP", "STATIONNAME", "YP", "WPA", "CLIMATEZONE"]
        
        print(f"Model loaded successfully from {model_path}")
    
    def preprocess_data(self, data):
        """
        Preprocess input data for prediction.
        
        Args:
            data (pd.DataFrame): Input data with required features
            
        Returns:
            torch.Tensor: Processed data ready for model prediction
        """
        # Check that all required features are present
        missing_features = [feat for feat in self.trained_on_features if feat not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select only the features used during training
        data = data[self.trained_on_features].copy()
        
        # Handle categorical features
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                data[col] = self.label_encoders[col].fit_transform(data[col].astype(str))
            else:
                # Handle unseen categories during deployment by setting them to -1 or another strategy
                data[col] = data[col].astype(str)
                for category in data[col].unique():
                    if category not in self.label_encoders[col].classes_:
                        # Handle unseen category (map to most frequent or use a default value)
                        data.loc[data[col] == category, col] = self.label_encoders[col].classes_[0]
                data[col] = self.label_encoders[col].transform(data[col])
        
        # Handle numerical features
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
            # First time fitting the scaler
            data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
        else:
            # Using already fitted scaler
            data[numeric_cols] = self.scaler.transform(data[numeric_cols])
        
        # Convert to tensor
        return torch.tensor(data.values.astype(np.float32))
    
    def predict(self, data):
        """
        Make crop yield predictions on new data.
        
        Args:
            data (pd.DataFrame): Input data with required features
            
        Returns:
            np.ndarray: Predicted crop yields
        """
        # Preprocess the input data
        processed_data = self.preprocess_data(data)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(processed_data)
        
        # Return predictions as numpy array
        return predictions.numpy()
    
    def predict_from_csv(self, csv_path):
        """
        Load data from CSV and make predictions.
        
        Args:
            csv_path (str): Path to CSV file with input data
            
        Returns:
            pd.DataFrame: Original data with predictions added
        """
        # Load the data
        data = pd.read_csv(csv_path)
        
        # Make predictions
        predictions = self.predict(data)
        
        # Add predictions to the dataframe
        data['Predicted_YA'] = predictions
        
        return data

# Example usage
if __name__ == "__main__":
    # Initialize the predictor
    predictor = CropYieldPredictor(model_path='student_crop_yield_model.joblib')
    
    # Example 1: Predict from a CSV file
    input_file = "GygaUganda - Station.csv"  # Your input file
    if os.path.exists(input_file):
        results = predictor.predict_from_csv(input_file)
        print("\nPrediction results:")
        print(results[['CROP', 'STATIONNAME', 'YA', 'Predicted_YA']].head())
        
        # Save results to CSV
        results.to_csv("prediction_results.csv", index=False)
        print(f"Full results saved to prediction_results.csv")
    else:
        print(f"Input file {input_file} not found.")
    
    # Example 2: Predict with custom data
    print("\nMaking prediction with sample data:")
    sample_data = pd.DataFrame({
        'CROP': ['Maize', 'Beans', 'Maize'],
        'STATIONNAME': ['Station1', 'Station2', 'Station3'],
        'YP': [5.2, 3.8, 4.7],
        'WPA': [0.85, 0.76, 0.89],
        'CLIMATEZONE': ['Tropical', 'Temperate', 'Tropical']
    })
    
    try:
        predictions = predictor.predict(sample_data)
        sample_data['Predicted_YA'] = predictions
        print(sample_data)
    except Exception as e:
        print(f"Error making prediction with sample data: {e}") 