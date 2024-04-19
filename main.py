# Put the code for your API here.
from fastapi import FastAPI
import numpy as np
import pickle
import os
from pydantic import BaseModel, Field
import pandas as pd
import json
from starter.ml.model import inference
from starter.ml.data import process_data

os.system('python starter/train_model.py')

app = FastAPI()

class CensusRow(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')
    # salary: str

@app.get('/')
def welcome():
    return 'Welcome'

@app.post('/inference')
def perform_inference(X: CensusRow):
    
    X_df = pd.DataFrame([X.dict(by_alias=True)], index=[0])

    with open(os.path.join('model', 'model.pkl'), 'rb') as f:
        save_data = pickle.load(f)
        model = save_data['model']
        encoder = save_data['encoder']
        lb = save_data['lb']
    
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, _, encoder, lb = process_data(X_df, categorical_features=cat_features, training=False, encoder=encoder, lb=lb)

    preds = inference(model, X)

    return int(preds[0])