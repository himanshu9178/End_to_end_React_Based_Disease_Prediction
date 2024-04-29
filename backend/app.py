from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from mangum import Mangum

# Initialize FastAPI app
app = FastAPI()

# Load the saved scaler and model with pickle
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# New CORS setup
origins = [
    "http://localhost:3000",  # React app's address
    "http://192.168.54.227:3000",
    "http://dia-bucket.s3-website.eu-north-1.amazonaws.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic model for input validation
class PredictionInput(BaseModel):
    Age: int
    Gender: str
    Polyuria: str
    Polydipsia: str
    sudden_weight_loss: str
    weakness: str
    Polyphagia: str
    Genital_thrush: str
    visual_blurring: str
    Itching: str
    Irritability: str
    delayed_healing: str
    partial_paresis: str
    muscle_stiffness: str
    Alopecia: str
    Obesity: str

@app.post("/predict")
async def predict_diabetes(input_data: PredictionInput):
    try:
        # Convert input data to dictionary, replacing underscores with spaces in keys
        input_dict = input_data.dict()
        corrected_input_dict = {key.replace('_', ' '): value for key, value in input_dict.items()}
        new_input = pd.DataFrame([corrected_input_dict])

        # Handling categorical features using pandas get_dummies
        categorical_features = ['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
                                'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability',
                                'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity']
        new_input_encoded = pd.get_dummies(new_input, columns=categorical_features)

        # Ensuring all model-required features are present
        model_features = model.feature_names_in_  # This assumes your model object has a `feature_names_in_` attribute.
        missing_features = set(model_features) - set(new_input_encoded.columns)
        for feature in missing_features:
            new_input_encoded[feature] = 0

        # Order columns as expected by the model
        new_input_encoded = new_input_encoded[model_features]
        new_input_encoded['Age'] = scaler.transform(new_input_encoded[['Age']].values.reshape(-1, 1))
        new_input_prediction = model.predict(new_input_encoded)
        prediction_result = 'Diabetic' if new_input_prediction[0] == 1 else 'Not Diabetic'

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"prediction": prediction_result}

# AWS Lambda handler
handler = Mangum(app)
