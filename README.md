# hw1_MLOps

```python
import requests
import json
```

#### Добавить модели


```python
headers = {"accept": "application/json", "content-type": "application/json"}
params = {"model_key": 1, "model_type": "LogisticRegression"}
r = requests.post('http://127.0.0.1:5000/model/add', headers=headers, data=json.dumps(params))
print(r, r.text)

headers = {"accept": "application/json", "content-type": "application/json"}
params = {"model_key": 2, "model_type": "RandomForestClassifier"}
r = requests.post('http://127.0.0.1:5000/model/add', headers=headers, data=json.dumps(params))
print(r, r.text)
```

    <Response [200]> "Successfully added model 1"
    
    <Response [200]> "Successfully added model 2"
    
    

#### Обучить модель 1


```python
model_params = {'random_state': 42}
params = {"model_key": 1, "model_params" : model_params}
r = requests.post('http://127.0.0.1:5000/model/fit', headers=headers, data=json.dumps(params))
print(r, r.text)
```

    <Response [200]> "Successfully fitted model 1"
    
    

#### Получить предсказание модели 1


```python
params = {"model_key" : 1}
r = requests.get('http://127.0.0.1:5000/model/predict', headers=headers, data = json.dumps(params))
print(r, r.text)
```

    <Response [200]> "{\"Model 1 prediction\": [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}"
    
    

#### Получить список доступных моделей


```python
r = requests.get('http://127.0.0.1:5000/available_models')
print(r, r.text)
```

    <Response [200]> {"1":"LogisticRegression","2":"RandomForestClassifier"}
    
    

#### Обучить модель 2


```python
model_params = {'n_estimators': 105, 'max_depth': 5}
params = {"model_key": 2, "model_params" : model_params}
r = requests.post('http://127.0.0.1:5000/model/fit', headers=headers, data=json.dumps(params))
print(r, r.text)
```

    <Response [200]> "Successfully fitted model 2"
    
    

#### Удалить модель 1


```python
params = {"model_key" : 1}
r = requests.delete('http://127.0.0.1:5000/model/delete', headers=headers, data = json.dumps(params))
print(r, r.text)

r = requests.get('http://127.0.0.1:5000/available_models', headers=headers)
print(r, r.text)
```

    <Response [200]> "Successfully deleted model 1"
    
    <Response [200]> {"2":"RandomForestClassifier"}
    
    


```python

```
