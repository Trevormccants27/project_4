import pytest
import numpy as np
import numbers
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from . model import train_model, compute_model_metrics, inference, compute_model_metrics_per_slice

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

@pytest.fixture
def preds():
    y = np.random.randint(0,2,size=(100,))
    y_pred = np.random.randint(0,2,size=(100,))
    return y, y_pred

def test_train_model(data):
    X, y = data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_inference(data, model):
    X, y = data
    y_pred = inference(model, X)
    assert y_pred.shape == y.shape

def test_compute_model_metrics(preds):
    y, y_pred = preds
    precision, recall, fbeta = compute_model_metrics(y, y_pred)
    assert isinstance(precision, numbers.Number)
    assert isinstance(recall, numbers.Number)
    assert isinstance(fbeta, numbers.Number)

def test_compute_model_metrics_per_slice(preds):
    y, y_pred = preds
    slices = np.random.randint(0,5, size=(len(y),))
    results = compute_model_metrics_per_slice(y, y_pred, slices)
    assert len(results.keys()) == 5
    for k, v in results.items():
        assert len(v) == 3
        assert k in set(slices)