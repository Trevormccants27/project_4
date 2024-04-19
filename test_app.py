from main import app
from fastapi.testclient import TestClient

import pandas as pd
import os
import json

client = TestClient(app)

def test_get():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == 'Welcome'

def test_post_0():
    data = pd.read_csv(os.path.join('data', 'census.csv'))
    row = data[data['salary'] == '<=50K'].iloc[0].to_dict()
    r = client.post('/inference', data=json.dumps(row))
    assert r.status_code == 200
    assert r.json() == 0

def test_post_1():
    data = pd.read_csv(os.path.join('data', 'census.csv'))
    row = data[data['salary'] == '>50K'].iloc[0].to_dict()
    r = client.post('/inference', data=json.dumps(row))
    assert r.status_code == 200
    assert r.json() == 1