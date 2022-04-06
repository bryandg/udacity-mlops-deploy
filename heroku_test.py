import requests
import json

data = {
        "age": 44,
        "workclass": "Private",
        "fnlgt": 198282,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States",
        "salary": ">50K",
    }


r = requests.get("https://salary-inference-app.herokuapp.com")
print(r.status_code)
print(r.json())

r = requests.post("https://salary-inference-app.herokuapp.com/inference/", data=json.dumps(data))
print(r.status_code)
print(r.json())
