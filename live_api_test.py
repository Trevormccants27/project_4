import requests
import os
import json
import pandas as pd

URL = 'https://project-4-bdo3.onrender.com/'

print('Testing get')
r = requests.get(URL)
print(r.status_code)
print(r.json())

data = pd.read_csv(os.path.join('data', 'census.csv'))

print('Testing Post 0')
row = data[data['salary'] == '<=50K'].iloc[0].to_dict()
r = requests.post(URL + 'inference', json=row)
print(r.status_code)
print(r.json())

print('Testing Post 1')
row = data[data['salary'] == '>50K'].iloc[0].to_dict()
r = requests.post(URL + 'inference', json=row)
print(r.status_code)
print(r.json())