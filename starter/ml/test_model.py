import pytest
import numpy as np
import numbers
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from . model import train_model, compute_model_metrics, inference

@pytest.fixture
def data():
    X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
    return X, y

@pytest.fixture
def model(data):
    X, y = data
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

def test_train_model(data):
    X, y = data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_inference(data, model):
    X, y = data
    y_pred = inference(model, X)
    assert y_pred.shape == y.shape

def test_compute_model_metrics():
    y = np.random.randint(0,2,size=(100,))
    y_pred = np.random.randint(0,2,size=(100,))
    precision, recall, fbeta = compute_model_metrics(y, y_pred)
    assert isinstance(precision, numbers.Number)
    assert isinstance(recall, numbers.Number)
    assert isinstance(fbeta, numbers.Number)