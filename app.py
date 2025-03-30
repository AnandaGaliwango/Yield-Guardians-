from flask import Flask, request, jsonify
import joblib
import numpy as np
import torch
import torch.nn as nn

app = Flask(__name__)

# **PASTE THE StudentModel CLASS DEFINITION HERE**
class StudentModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Load the trained model
try:
    model = joblib.load('student_crop_yield_model.joblib')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        data = request.get_json()
        # **IMPORTANT:** You'll need to determine the 'input_dim'
        # and extract the features from the 'data' dictionary
        # in the correct order and format for your model.
        # Example (assuming 3 input features):
        features = np.array([data['feature1'], data['feature2'], data['feature3']]).astype(np.float32).reshape(1, -1)

        # You might need to perform the same preprocessing here
        # that you did before training (e.g., scaling).

        # Make the prediction
        # **Important:** You might need to convert the NumPy array to a PyTorch tensor
        input_tensor = torch.tensor(features)
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            prediction = model(input_tensor).numpy()[0]

        return jsonify({'prediction': prediction.tolist()}) # Convert NumPy array to list for JSON

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)