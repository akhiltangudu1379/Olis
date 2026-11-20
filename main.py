import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from model_pipeline import CancellationModelPipeline

# The existing Colab joblib artifact refers to __main__.CancellationModelPipeline.
# Make that class resolvable before deserializing this trusted model artifact.
setattr(sys.modules['__main__'], 'CancellationModelPipeline', CancellationModelPipeline)
pipeline = joblib.load('cancellation_model.joblib')

app = FastAPI(title='Olist Cancellation Prediction API', version='1.0.0')

class OrderData(BaseModel):
    purchase_year: int = Field(ge=2000, le=2100)
    purchase_month: int = Field(ge=1, le=12)
    purchase_day: int = Field(ge=1, le=31)
    purchase_hour: int = Field(ge=0, le=23)
    purchase_dayofweek: int = Field(ge=0, le=6)
    customer_state: str
    customer_city: str

@app.get('/')
def home():
    return {'message': 'Olist Cancellation Prediction API is active', 'docs_url': '/docs'}

@app.get('/health')
def health():
    return {'status': 'healthy'}

@app.post('/predict')
def predict(order: OrderData):
    try:
        data = pd.DataFrame([order.model_dump()])
        prediction = int(pipeline.predict(data)[0])
        probability = float(pipeline.predict_proba(data)[0])
        return {'cancellation_prediction': prediction,
                'cancellation_probability': round(probability, 4),
                'status': 'Canceled' if prediction == 1 else 'Not Canceled'}
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
