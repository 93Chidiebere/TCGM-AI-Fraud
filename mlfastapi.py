from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
import joblib
from tcgm import TCGMClassifier
import numpy as np

# Initialize app and load model
app = FastAPI(
    title="TCGM Powered Fraud Detection API",
    description="API for detecting fraudulent transactions using TimeCost Gradient Machine",
    version="1.0.0",
    debug=True                
)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature names expected by the model
FEATURE_ORDER = [
    'amount',
    'origin_balance_change',
    'destination_balance_change',
    'origin_error',
    'destination_error',
    'origin_zero_after',
    'destination_zero_before',
    'amount_to_origin_balance',
    'amount_to_destination_balance',
    'origin_out_degree',
    'destination_in_degree',
    'origin_pagerank',
    'destination_pagerank',
    'velocity'
]

# Request body model for single transaction
class FraudRequest(BaseModel):
    transaction_id: Optional[str] = None  # Optional ID for tracking
    amount: float
    origin_balance_change: float
    destination_balance_change: float
    origin_error: float
    destination_error: float
    origin_zero_after: int
    destination_zero_before: int
    amount_to_origin_balance: float
    amount_to_destination_balance: float
    origin_out_degree: int
    destination_in_degree: int
    origin_pagerank: float
    destination_pagerank: float
    velocity: float

# Request body model for batch processing
class FraudRequestBatch(BaseModel):
    transactions: List[FraudRequest]
    
    class Config:
        json_schema_extra = {
            "example": {
                "transactions": [
                    {
                        "transaction_id": "TXN001",
                        "amount": 250000,
                        "origin_balance_change": 5000,
                        "destination_balance_change": 300000,
                        "origin_error": 0,
                        "destination_error": 0,
                        "origin_zero_after": 0,
                        "destination_zero_before": 0,
                        "amount_to_origin_balance": 1.0,
                        "amount_to_destination_balance": 0.5,
                        "origin_out_degree": 10,
                        "destination_in_degree": 1,
                        "origin_pagerank": 0,
                        "destination_pagerank": 0,
                        "velocity": 0
                    },
                    {
                        "transaction_id": "TXN002",
                        "amount": 10000,
                        "origin_balance_change": -10000,
                        "destination_balance_change": 10000,
                        "origin_error": 0,
                        "destination_error": 0,
                        "origin_zero_after": 0,
                        "destination_zero_before": 0,
                        "amount_to_origin_balance": 0.5,
                        "amount_to_destination_balance": 0.1,
                        "origin_out_degree": 5,
                        "destination_in_degree": 2,
                        "origin_pagerank": 0.001,
                        "destination_pagerank": 0.002,
                        "velocity": 10
                    }
                ]
            }
        }

# Health check endpoint
@app.get("/")
def home():
    return{
        "service": "TimeCost Gradient Machine Fraud Detection API",
        "status": "running",
        "endpoints": {
            "real_time": "/predict, /predict_proba",
            "batch": "/predict_batch, /predict_proba_batch"
        }
    }

# # Prediction endpoint - Single transaction
# @app.post("/predict")
# def predict_fraud(request: FraudRequest):
#     """
#     Returns binary fraud prediction (0 or 1) for a single transaction
#     """
#     raw_features = np.array([
#         getattr(request, feature) for feature in FEATURE_ORDER]).reshape(1, -1)

#     scaled_features = scaler.transform(raw_features)
#     prediction = model.predict(scaled_features)[0]
#     fraud_label = "Fraudulent" if prediction == 1 else "Legitimate"

#     result = {
#         "prediction": int(prediction),
#         "label": fraud_label
#     }
    
#     if request.transaction_id:
#         result["transaction_id"] = request.transaction_id
    
#     return result

# Predict Probability endpoint - Single transaction
@app.post("/predict_proba")
def predict_fraud_proba(request: FraudRequest):
    """
    Returns fraud probability with prediction for a single transaction
    """
    raw_features = np.array([
        getattr(request, feature) for feature in FEATURE_ORDER]).reshape(1, -1)

    scaled_features = scaler.transform(raw_features)

    if hasattr(model, "predict_proba"):
        fraud_probability = float(model.predict_proba(scaled_features)[0][1])
        prediction = 1 if fraud_probability > 0.5 else 0
        fraud_label = "Fraudulent" if prediction == 1 else "Legitimate"
        
        result = {
            "fraud_probability": fraud_probability,
            "prediction": int(prediction),
            "label": fraud_label
        }
        
        if request.transaction_id:
            result["transaction_id"] = request.transaction_id
            
        return result
    else:
        return {"error": "Model doesn't support probability prediction"}


# ========== BATCH PROCESSING ENDPOINTS ==========

# @app.post("/predict_batch")
# def predict_fraud_batch(request: FraudRequestBatch):
#     """
#     Returns binary fraud predictions for multiple transactions (batch processing)
#     OPTIMIZED: Processes all transactions in one vectorized operation
#     """
#     if not request.transactions:
#         raise HTTPException(status_code=400, detail="No transactions provided")
    
#     if len(request.transactions) > 10000:
#         raise HTTPException(status_code=400, detail="Maximum 10,000 transactions per batch")
    
#     try:
#         # Convert all transactions to feature matrix (vectorized)
#         raw_features = np.array([[
#             getattr(txn, feature) for feature in FEATURE_ORDER
#         ] for txn in request.transactions])
        
#         # Scale all features at once
#         scaled_features = scaler.transform(raw_features)
        
#         # Predict all at once (much faster than loop)
#         predictions = model.predict(scaled_features)
        
#         # Build response
#         results = []
#         for i, (txn, pred) in enumerate(zip(request.transactions, predictions)):
#             result = {
#                 "prediction": int(pred),
#                 "label": "Fraudulent" if pred == 1 else "Legitimate"
#             }
            
#             if txn.transaction_id:
#                 result["transaction_id"] = txn.transaction_id
#             else:
#                 result["transaction_id"] = f"batch_{i}"
                
#             results.append(result)
        
#         return {
#             "total_transactions": len(results),
#             "fraudulent_count": int(predictions.sum()),
#             "legitimate_count": int(len(predictions) - predictions.sum()),
#             "results": results
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/predict_proba_batch")
def predict_fraud_proba_batch(request: FraudRequestBatch):
    """
    Returns fraud probabilities for multiple transactions (batch processing)
    OPTIMIZED: Processes all transactions in one vectorized operation
    """
    if not request.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided")
    
    if len(request.transactions) > 10000:
        raise HTTPException(status_code=400, detail="Maximum 10,000 transactions per batch")
    
    if not hasattr(model, "predict_proba"):
        raise HTTPException(status_code=400, detail="Model doesn't support probability prediction")
    
    try:
        # Convert all transactions to feature matrix (vectorized)
        raw_features = np.array([[
            getattr(txn, feature) for feature in FEATURE_ORDER
        ] for txn in request.transactions])
        
        # Scale all features at once
        scaled_features = scaler.transform(raw_features)
        
        # Get probabilities for all transactions at once
        probabilities = model.predict_proba(scaled_features)[:, 1]  # Get fraud class probability
        predictions = (probabilities > 0.5).astype(int)
        
        # Build response
        results = []
        for i, (txn, prob, pred) in enumerate(zip(request.transactions, probabilities, predictions)):
            result = {
                "fraud_probability": float(prob),
                "prediction": int(pred),
                "label": "Fraudulent" if pred == 1 else "Legitimate"
            }
            
            if txn.transaction_id:
                result["transaction_id"] = txn.transaction_id
            else:
                result["transaction_id"] = f"batch_{i}"
                
            results.append(result)
        
        # Calculate statistics
        fraudulent_count = int(predictions.sum())
        avg_fraud_probability = float(probabilities.mean())
        high_risk_count = int((probabilities > 0.7).sum())
        
        return {
            "total_transactions": len(results),
            "fraudulent_count": fraudulent_count,
            "legitimate_count": len(predictions) - fraudulent_count,
            "average_fraud_probability": avg_fraud_probability,
            "high_risk_count": high_risk_count,
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")