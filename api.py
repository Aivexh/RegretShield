# api.py
"""FastAPI interface for Regret Shield."""

from src.recommender import AlternativeRecommender
from src.intervention_engine import InterventionEngine
import uvicorn
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import sys
sys.path.append('.')


# Load artifacts
print("Loading Regret Shield artifacts...")
artifacts = joblib.load('models/regret_shield_artifacts.pkl')
feature_engineer = artifacts['feature_engineer']
model = artifacts['model']
products_df = artifacts['products']
users_df = artifacts['users']

# Initialize components
intervention_engine = InterventionEngine(threshold=0.6)
recommender = AlternativeRecommender(products_df, feature_engineer, model)


# API Models
class PredictionRequest(BaseModel):
    user_id: int
    product_id: int
    time_on_page: float = Field(..., ge=0, le=300)
    scroll_depth: float = Field(..., ge=0, le=1)
    num_images_viewed: int = Field(..., ge=0, le=20)
    review_hover_count: int = Field(..., ge=0, le=50)
    add_to_wishlist: int = Field(..., ge=0, le=1)
    cart_abandonment: int = Field(..., ge=0, le=1)

    class Config:
        schema_extra = {
            "example": {
                "user_id": 123,
                "product_id": 456,
                "time_on_page": 45.5,
                "scroll_depth": 0.7,
                "num_images_viewed": 4,
                "review_hover_count": 3,
                "add_to_wishlist": 0,
                "cart_abandonment": 0
            }
        }


class AlternativeProduct(BaseModel):
    product_id: int
    category: str
    price: float
    return_risk: float
    rating: float


class PredictionResponse(BaseModel):
    return_probability: float
    decision: str  # "allow", "warn", "block"
    warning_message: Optional[str]
    alternatives: List[AlternativeProduct]
    risk_factors: Optional[Dict]
    threshold_used: float


# Initialize FastAPI
app = FastAPI(
    title="Regret Shield API",
    description="Pre-purchase return prediction and alternative recommendation system",
    version="1.0.0"
)


@app.get("/")
async def root():
    return {
        "message": "Regret Shield API is running",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "stats": "GET /stats"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "artifacts_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict_return(request: PredictionRequest):
    """Predict return probability and get recommendations."""

    # Check if user and product exist
    if request.user_id not in users_df['user_id'].values:
        raise HTTPException(
            status_code=404, detail=f"User {request.user_id} not found")

    if request.product_id not in products_df['product_id'].values:
        raise HTTPException(
            status_code=404, detail=f"Product {request.product_id} not found")

    # Create feature vector
    input_data = pd.DataFrame([{
        'user_id': request.user_id,
        'product_id': request.product_id,
        'time_on_page': request.time_on_page,
        'scroll_depth': request.scroll_depth,
        'num_images_viewed': request.num_images_viewed,
        'review_hover_count': request.review_hover_count,
        'add_to_wishlist': request.add_to_wishlist,
        'cart_abandonment': request.cart_abandonment,
        'return': 0  # Placeholder
    }])

    # Add product category
    product = products_df[products_df['product_id']
                          == request.product_id].iloc[0]
    input_data['category'] = product['category']

    # Transform features
    try:
        X = feature_engineer.transform(input_data)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Feature engineering failed: {str(e)}")

    # Predict return probability
    return_prob = float(model.predict_proba(X, 'xgboost')[0])

    # Get intervention decision
    intervention = intervention_engine.evaluate_purchase(
        return_prob,
        product_info={'product_id': request.product_id},
        user_info={'user_id': request.user_id}
    )

    # Get SHAP explanation
    try:
        explanation = model.explain_prediction(X, 'xgboost')
        risk_factors = explanation
    except:
        risk_factors = None

    # Get alternative recommendations
    original_features = request.dict()
    recommendations = recommender.recommend_alternatives(
        request.user_id,
        request.product_id,
        original_features,
        top_k=3
    )

    # Format alternatives
    alternatives = [
        AlternativeProduct(
            product_id=rec['product_id'],
            category=rec['category'],
            price=rec['price'],
            return_risk=rec['return_risk'],
            rating=rec['rating']
        )
        for rec in recommendations
    ]

    return PredictionResponse(
        return_probability=return_prob,
        decision=intervention['decision'],
        warning_message=intervention['warning_message'],
        alternatives=alternatives,
        risk_factors=risk_factors,
        threshold_used=intervention['threshold_used']
    )


@app.get("/stats")
async def get_stats():
    """Get intervention statistics."""
    return intervention_engine.get_stats()


# At the bottom of api.py, change:
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
