# Crop Prediction Model with Knowledge Distillation

This project implements a crop prediction system using knowledge distillation techniques. It predicts suitable crops based on environmental and soil conditions, leveraging machine learning models deployed on Google Cloud Functions.


## Project Overview

This project utilizes knowledge distillation to train a smaller, more efficient "student" model from larger, more complex "teacher" models. The resulting model is deployed as a RESTful API using Google Cloud Functions, allowing for scalable and accessible crop predictions.

## Project Structure
crop-prediction/
├── main.py                        # Flask application
├── deploy_model.py                # Model deployment code
├── model4-kdum.py                # Training script
├── student_crop_yield_model.joblib # Trained model
├── requirements.txt               # Dependencies
└── Dockerfile                     # Container configuration


##  Authors
- Naggayi Daphne Pearl 23/U/13097/EVE
- Ananda Galiwango 23/U/25628/PS


## Features

- **Knowledge Distillation:** Trains an efficient student model from multiple teacher models.
- **Cloud Deployment:** Deploys the model as a serverless API using Google Cloud Functions (2nd gen).
- **RESTful API:** Provides a simple API endpoint for making crop predictions.
- **Numerical and Categorical Features:** Handles diverse input features with appropriate preprocessing.
- **Comprehensive Data Preprocessing:** Includes scaling and encoding steps.
- **Model Inference:** Performs predictions using the trained student model.
- **CORS Support:** Includes Cross-Origin Resource Sharing (CORS) headers for web application compatibility.

## Input Features

The API accepts a JSON payload with a `features` array. The array should contain 21 numerical values representing the following features in order:

1.  Station Location (encoded)
2.  Latitude
3.  Longitude
4.  Elevation
5.  Temperature
6.  Rainfall
7.  Humidity
8.  Solar Radiation
9.  Wind Speed
10. Soil Type (encoded)
11. Soil pH
12. Soil Moisture
13. Soil Temperature
14. Nitrogen Content
15. Phosphorus Content
16. Potassium Content
17. Organic Matter
18. Season (encoded)
19. Month (encoded)
20. Year
21. Previous Crop (encoded)

**Note:** Categorical features (Station Location, Soil Type, Season, Month, Previous Crop) are encoded into numerical values before being passed to the model.

## Setup and Installation

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/yourusername/crop-prediction.git]([https://github.com/yourusername/crop-prediction.git](https://github.com/Naggayi-Daphne-Pearl/Crop-Prediction-project))
    cd crop-prediction
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Google Cloud Functions:**

    * Ensure you have the Google Cloud SDK (gcloud CLI) installed and configured.
    * Enable the Cloud Functions API, Cloud Build API, and Cloud Run API for your project.
    * Deploy the function:

        ```bash
        gcloud functions deploy predict-handler-v2 \
        --runtime python311 \
        --trigger-http \
        --allow-unauthenticated \
        --memory 512MB \
        --timeout 3600s \
        --region us-central1
        ```

    * Note the deployed function's URL from the output.

## Testing

1.  **Test with `test_request.py`:**

    ```bash
    python test_request.py
    ```

2.  **Test with `curl`:**

    ```bash
    curl -X POST \
    -H "Content-Type: application/json" \
    -d '{
  "data": [
    {
      "STATIONNAME": "Arua",
      "LONGITUDE": 30.92,
      "LATITUDE": 3.05,
      "ELEVATION_METER": 1211,
      "CROP": "Rainfed maize",
      "YW": 7.99,
      "YP": 14.04,
      "WPA": 3.5,
      "CLIMATEZONE": 7601
    },
    {
      "STATIONNAME": "Namulonge",
      "LONGITUDE": 32.615,
      "LATITUDE": 0.525,
      "ELEVATION_METER": 1160,
      "CROP": "Rainfed maize",
      "YW": 7.76,
      "YP": 13.02,
      "WPA": 5.99,
      "CLIMATEZONE": 7501
    }
  ]' \
    https://actual-yield-predictor-4065ab6b-53bb-4156-abd0.cranecloud.io/predict
    ```

    * Replace `YOUR_FUNCTION_URL` with your Cloud Function's URL.

3.  **Test with Postman:**

    * Create a POST request to your function URL.
    * Set the `Content-Type` header to `application/json`.
    * Provide the input features in the JSON body as shown in the `curl` example.

## Usage

Send a POST request to your deployed Cloud Function's URL with the input features in the JSON format specified above.

**Response Format:**

```json
{
    "prediction": "Crop Name",
    "confidence": 0.95
}





