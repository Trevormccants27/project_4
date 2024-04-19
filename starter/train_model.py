# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import os
import pickle
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, compute_model_metrics_per_slice

# Add code to load in the data.
data = pd.read_csv(os.path.join('data', 'census.csv'))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

with open(os.path.join('model', 'model.pkl'), 'wb') as f:
    pickle.dump(model, f)

# Test model
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print('OVERALL METRICS')
print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'fbeta: {fbeta}')

with open('slice_output.txt', 'w') as f:
    for cat in cat_features:
        results = compute_model_metrics_per_slice(y_test, preds, test[cat])
        f.write(f'METRICS FOR {cat}\n')
        f.write(str(results) + '\n')