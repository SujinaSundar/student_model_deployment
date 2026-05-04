from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open("model.pkl","rb"))

@app.get("/")
def home():
    return {"status": "ok"}
from pydantic import BaseModel

class InputData(BaseModel):
    hours: float

@app.post("/predict")
def predict(data:InputData):
    prediction = model.predict([[data.hours]])
    return {"Result" : int(prediction[0])}
        

