from fastapi import FastAPI
import uvicorn
import joblib
from tcgm import TCGMClassifier

app = FastAPI(debug=True)
model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"text": "TimeCost Gradient Machine Powered Fraud Detection API"}

# @app.get("/predict")
# def predict(data: dict):
#     features = [data[key] for key in sorted(data.keys())]
#     prediction = model.predict([features])
#     return {"prediction": int(prediction[0])}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)