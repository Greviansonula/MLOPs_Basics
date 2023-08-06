from fastapi import FastAPI
from inference_onnx import ColaONNXPredictor

predictor = ColaONNXPredictor("./models/model.onnx")
app =FastAPI(title="MLOPs Basics App")

@app.get("/")
async def home():
    return "<h2>This is a sample NLP Project</h2>"

@app.get("/predict")
async def get_prediction(text: str):
    return predictor.predict({"sentence": "This is a samole mes"})