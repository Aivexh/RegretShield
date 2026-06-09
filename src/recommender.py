# src/recommender.py - Updated version
"""Alternative product recommendation system."""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


class AlternativeRecommender:
    """Recommend alternative products with lower return risk."""

    def __init__(self, products_df: pd.DataFrame, feature_engineer, model):
        self.products_df = products_df
        self.feature_engineer = feature_engineer
        self.model = model
        self.similarity_bonus = 0.3
        self.price_penalty = 0.0005
        self.best_model_name = getattr(
            model, 'best_model_name', 'gradient_boosting')

    def recommend_alternatives(self,
                               user_id: int,
                               product_id: int,
                               original_features: Dict,
                               top_k: int = 3) -> List[Dict]:
        """Recommend alternative products with lower return risk."""
        # Get original product info
        original_product = self.products_df[self.products_df['product_id']
                                            == product_id].iloc[0]
        original_category = original_product['category']
        original_price = original_product['price']

        # Find candidate products (same category, different product)
        candidates = self.products_df[
            (self.products_df['category'] == original_category) &
            (self.products_df['product_id'] != product_id)
        ].copy()

        if len(candidates) < top_k:
            # Fallback to similar categories or all products
            candidates = self.products_df[self.products_df['product_id'] != product_id].copy(
            )

        # Score each candidate
        scores = []

        for _, product in candidates.iterrows():
            try:
                # Create features for this candidate
                candidate_features = self._create_features_for_user_product(
                    user_id, product['product_id'], original_features
                )

                # Predict return risk
                return_risk = self.model.predict_proba(
                    candidate_features, self.best_model_name)[0]

                # Calculate recommendation score
                # Lower risk is better, similar price is better
                price_diff_ratio = abs(
                    product['price'] - original_price) / (original_price + 1)
                similarity_score = 1.0 if product['category'] == original_category else 0.5

                score = (1 - return_risk) + (similarity_score *
                                             self.similarity_bonus) - (price_diff_ratio * self.price_penalty)

                scores.append({
                    'product_id': int(product['product_id']),
                    'category': product['category'],
                    'price': float(product['price']),
                    'return_risk': float(return_risk),
                    'rating': float(product['rating_score']),
                    'score': float(score)
                })
            except Exception as e:
                print(
                    f"Warning: Could not score product {product['product_id']}: {e}")
                continue

        # Sort by score and return top K
        recommendations = sorted(
            scores, key=lambda x: x['score'], reverse=True)[:top_k]

        return recommendations

    def _create_features_for_user_product(self, user_id: int, product_id: int,
                                          original_features: Dict) -> pd.DataFrame:
        """Create feature vector for a user-product pair."""
        # Get product info
        product = self.products_df[self.products_df['product_id']
                                   == product_id].iloc[0]

        # Create a DataFrame with minimal features needed
        features = pd.DataFrame([{
            'user_id': user_id,
            'product_id': product_id,
            'category': product['category'],
            'time_on_page': original_features.get('time_on_page', 30),
            'scroll_depth': original_features.get('scroll_depth', 0.5),
            'num_images_viewed': original_features.get('num_images_viewed', 3),
            'review_hover_count': original_features.get('review_hover_count', 2),
            'add_to_wishlist': original_features.get('add_to_wishlist', 0),
            'cart_abandonment': original_features.get('cart_abandonment', 0),
            'return': 0  # Placeholder
        }])

        # Apply feature engineering
        transformed = self.feature_engineer.transform(features)

        return transformed
