#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


#finput_path = "/kaggle/input/"
#input_path = " c:\Users\ANANDA\Downloads\fork-of-model4-kdum.py"
#files = os.listdir(input_path)
#print("Files in /kaggle/input/:", files)
# Define dataset directory path
#dataset_path = "/kaggle/input/gycamodel2"
#dataset_path = "c:\Users\ANANDA\Documents\"
# List all files in the directory
#files = os.listdir(dataset_path)
#print("Files in gycamodel2:", files)

# Define the correct file path
#file_path = "/kaggle/input/gycamodel2/GygaModelRunsUganda.xlsx"
file_path = "c:\\Users\\ANANDA\\Documents\\GygaModelRunsUganda.xlsx"
# Load the Excel file
xls = pd.ExcelFile(file_path)
for sheet_name in xls.sheet_names:
    df = xls.parse(sheet_name)
    df.to_csv(f"{sheet_name}.csv", index=False)
    print(f"Saved {sheet_name}.csv")
# List all files in the specified directory
file_path = "Station.csv"  # Adjust the file path as necessary
df = pd.read_csv(file_path)
print("First few rows of the data:")
print(df.head())


# In[ ] 

# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import joblib

# In[4]:


# Define Teacher Models
class TeacherModel1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TeacherModel1, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class TeacherModel2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TeacherModel2, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class TeacherModel3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TeacherModel3, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Define Student Model
class StudentModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Load dataset (with basic preprocessing)

def load_dataset(file_path, target_column='YA'):  # Added target_column parameter
    df = pd.read_csv(file_path)
    df = df.dropna()

    # Separate features and target
    y = df[target_column].values.astype(np.float32)  # Target is YA
    X = df.drop(target_column, axis=1)  # Features are all other columns

    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    if categorical_cols:
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    scaler = MinMaxScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X = X.values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return torch.tensor(X_train), torch.tensor(X_test), torch.tensor(y_train), torch.tensor(y_test)



# Train teacher models
def train_teacher(model, X_train, y_train, epochs=100, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_data = torch.utils.data.TensorDataset(X_train, y_train.view(-1, 1))  # Reshape y
    dataloader = DataLoader(train_data, batch_size=4, shuffle=True, drop_last=False)

    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
# Train student model using teacher predictions
def train_student(student_model, teacher_models, X_train, y_train, epochs=100, lr=0.01):
    optimizer = optim.Adam(student_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_data = torch.utils.data.TensorDataset(X_train, y_train.view(-1, 1))  # Reshape y
    dataloader = DataLoader(train_data, batch_size=4, shuffle=True, drop_last=False)

    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            teacher_preds = [teacher(batch_X).detach() for teacher in teacher_models]
            avg_teacher_preds = torch.mean(torch.stack(teacher_preds), dim=0)
            student_preds = student_model(batch_X)
            loss = criterion(student_preds, avg_teacher_preds)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Student Loss: {loss.item():.4f}")

# Evaluate student model (RMSE and R²)
def evaluate_student(student_model, X_test, y_test):
    student_model.eval()
    with torch.no_grad():
        student_preds = student_model(X_test)
        mse = mean_squared_error(y_test.numpy(), student_preds.numpy())
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test.numpy(), student_preds.numpy())
        accuracy = max(0, 100 - (mse * 100))
        print(f"RMSE: {rmse:.4f}")
        print(f"R-squared: {r2:.4f}")
        print(f"Test Loss (MSE): {mse:.4f}")
        print(f"Accuracy: {accuracy:.2f}%")
# Example Usage
file_path = "Station.csv"
def load_dataset_specific_features(file_path, target_column='YA', feature_columns=None):
    """
    Loads a CSV file and selects specific columns as features.

    Args:
        file_path (str): The path to the CSV file.
        target_column (str, optional): The name of the target column. Defaults to 'YA'.
        feature_columns (list, optional): A list of column names to use as features.
                                         If None, all columns except the target are used.
                                         Defaults to None.

    Returns:
        tuple: X_train, X_test, y_train, y_test as PyTorch tensors.
    """
    df = pd.read_csv(file_path)
    df = df.dropna()

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the CSV file.")

    y = df[target_column].values.astype(np.float32)

    if feature_columns is None:
        X = df.drop(target_column, axis=1)
    else:
        for col in feature_columns:
            if col not in df.columns:
                raise ValueError(f"Feature column '{col}' not found in the CSV file.")
        X = df[feature_columns].copy()  # Use only the specified columns

    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    if categorical_cols:
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    scaler = MinMaxScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X = X.values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return torch.tensor(X_train), torch.tensor(X_test), torch.tensor(y_train), torch.tensor(y_test)

# Example usage:
file_path = "Station.csv"
target_column = 'YA'
selected_features = ["CROP", "STATIONNAME", "YP", "WPA","CLIMATEZONE"]  # Specify the features you want to use

X_train, X_test, y_train, y_test = load_dataset_specific_features(file_path, target_column, selected_features)

#X_train, X_test, y_train, y_test = load_dataset(file_path)
input_dim = X_train.shape[1]
output_dim = 1  # Predict a single value (YA)

# Initialize and train teachers
teacher1 = TeacherModel1(input_dim, output_dim)
teacher2 = TeacherModel2(input_dim, output_dim)
teacher3 = TeacherModel3(input_dim, output_dim)
print("Training Teacher Models...")

train_teacher(teacher1, X_train, y_train)  # Pass X_train and y_train
train_teacher(teacher2, X_train, y_train)  # Pass X_train and y_train
train_teacher(teacher3, X_train, y_train)  # Pass X_train and y_train
# Train student model using teacher predictions

student = StudentModel(input_dim, output_dim)
print("Training Student Model...")
train_student(student, [teacher1, teacher2, teacher3], X_train, y_train) # Pass X_train and y_train

student_model_filename = 'student_crop_yield_model.joblib'  # Choose a descriptive filename
joblib.dump(student, student_model_filename)
print(f"Trained student model saved as {student_model_filename}")


# Evaluate Student Model
print("Evaluating Student Model...")
evaluate_student(student, X_test, y_test) # Pass X_test and y_test




import pandas as pd




import joblib

current_directory = os.getcwd()
print(f"The current working directory is: {current_directory}")