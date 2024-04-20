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

app = FastAPI()

class CensusRow(BaseModel):
    age: int = Field(examples=[50])
    workclass: str = Field(examples=['Self-emp-not-inc'])
    fnlgt: int = Field(examples=[83311])
    education: str = Field(examples=['Bachelors'])
    education_num: int = Field(alias='education-num', examples=[13])
    marital_status: str = Field(alias='marital-status', examples=['Married-civ-spouse'])
    occupation: str = Field(examples=['Exec-managerial'])
    relationship: str = Field(examples=['Husband'])
    race: str = Field(examples=['White'])
    sex: str = Field(examples=['Male'])
    capital_gain: int = Field(alias='capital-gain', examples=[0])
    capital_loss: int = Field(alias='capital-loss', examples=[0])
    hours_per_week: int = Field(alias='hours-per-week', examples=[13])
    native_country: str = Field(alias='native-country', examples=['United-States'])

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