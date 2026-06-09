# src/feature_engineering.py
"""Feature engineering pipeline for return prediction."""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Optional


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering pipeline for return prediction."""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.user_fingerprints = None
        self.product_risk_scores = None
        self.category_encoder = None
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit encoders and compute aggregated features."""

        # Store original columns for reference
        self.original_columns = X.columns.tolist()

        # 1. Fit category encoder
        if 'category' in X.columns:
            self.category_encoder = LabelEncoder()
            # Handle missing values
            categories = X['category'].fillna('unknown').astype(str)
            self.category_encoder.fit(categories)
            self.label_encoders['category'] = self.category_encoder

        # 2. Compute user return fingerprint (aggregated stats per user)
        if 'user_id' in X.columns and 'return' in X.columns and y is not None:
            # Use y if provided, otherwise use X['return']
            target = y if y is not None else X.get('return', None)
            if target is not None:
                user_stats = X.groupby('user_id').agg({
                    'return': ['mean', 'count'],
                    'time_on_page': 'mean',
                    'scroll_depth': 'mean'
                }).round(4)

                user_stats.columns = ['user_return_rate', 'user_purchase_count',
                                      'avg_time_on_page', 'avg_scroll_depth']
                self.user_fingerprints = user_stats.to_dict('index')

        # 3. Compute product risk score
        if 'product_id' in X.columns and 'return' in X.columns:
            product_risk = X.groupby('product_id')['return'].mean().to_dict()
            self.product_risk_scores = product_risk

        # 4. Identify numeric columns for scaling
        self.numeric_columns = self._get_numeric_cols(X)

        # 5. Fit scaler on numeric features
        if len(self.numeric_columns) > 0:
            # Create a temporary copy with filled NA values
            X_temp = X[self.numeric_columns].copy()
            X_temp = X_temp.fillna(0)
            self.scaler.fit(X_temp)

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features."""
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")

        X = X.copy()

        # 1. Encode categorical variables
        if 'category' in X.columns and self.category_encoder is not None:
            # Fill missing and encode
            categories = X['category'].fillna('unknown').astype(str)
            # Handle unseen categories
            unseen_mask = ~categories.isin(self.category_encoder.classes_)
            if unseen_mask.any():
                # Map unseen categories to the most common category
                categories[unseen_mask] = self.category_encoder.classes_[0]
            X['category_encoded'] = self.category_encoder.transform(categories)

        # 2. Add aggregated user fingerprint
        if self.user_fingerprints is not None and 'user_id' in X.columns:
            user_features = []
            for _, row in X.iterrows():
                user_id = row['user_id']
                if user_id in self.user_fingerprints:
                    user_features.append(self.user_fingerprints[user_id])
                else:
                    # Default values for new users
                    user_features.append({
                        'user_return_rate': 0.1,
                        'user_purchase_count': 0,
                        'avg_time_on_page': 30,
                        'avg_scroll_depth': 0.5
                    })

            user_df = pd.DataFrame(user_features)
            X['user_return_rate'] = user_df['user_return_rate'].values
            X['user_purchase_count'] = user_df['user_purchase_count'].values
            X['avg_time_on_page'] = user_df['avg_time_on_page'].values
            X['avg_scroll_depth'] = user_df['avg_scroll_depth'].values
        else:
            # Default values if no fingerprint available
            X['user_return_rate'] = 0.1
            X['user_purchase_count'] = 0
            X['avg_time_on_page'] = 30
            X['avg_scroll_depth'] = 0.5

        # 3. Add product risk score
        if self.product_risk_scores is not None and 'product_id' in X.columns:
            X['product_risk_score'] = X['product_id'].map(
                self.product_risk_scores).fillna(0.1)
        else:
            X['product_risk_score'] = 0.1

        # 4. Create interaction features
        # Engagement score (normalized time * depth)
        X['engagement_score'] = np.clip(
            (X['time_on_page'] / 100) * X['scroll_depth'], 0, 1)

        # Images per time spent
        X['images_per_time'] = X['num_images_viewed'] / (X['time_on_page'] + 1)

        # Hover intensity
        X['hover_intensity'] = X['review_hover_count'] / \
            (X['time_on_page'] + 1)

        # Add to wishlist indicator (already binary)
        X['wishlist_indicator'] = X['add_to_wishlist']

        # Cart abandonment risk (already binary)
        X['abandonment_risk'] = X['cart_abandonment']

        # Time-decay features (less time = more impulsive)
        X['impulsiveness_flag'] = (X['time_on_page'] < 20).astype(int)

        # Scroll engagement (low scroll = low interest)
        X['low_engagement_flag'] = ((X['scroll_depth'] < 0.3) & (
            X['time_on_page'] < 30)).astype(int)

        # 5. Scale numeric features
        numeric_cols_to_scale = [
            col for col in self.numeric_columns if col in X.columns]
        if len(numeric_cols_to_scale) > 0:
            X_scaled = X[numeric_cols_to_scale].fillna(0)
            X_scaled = self.scaler.transform(X_scaled)
            for i, col in enumerate(numeric_cols_to_scale):
                X[col] = X_scaled[:, i]

        # 6. Drop original identifiers and raw features (keep engineered ones)
        drop_cols = ['user_id', 'product_id', 'category', 'return']
        X = X.drop([col for col in drop_cols if col in X.columns],
                   axis=1, errors='ignore')

        return X

    def _get_numeric_cols(self, df: pd.DataFrame) -> List[str]:
        """Get list of numeric columns for scaling."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude columns that shouldn't be scaled
        exclude = ['user_id', 'product_id', 'return', 'category_encoded',
                   'wishlist_indicator', 'abandonment_risk', 'impulsiveness_flag',
                   'low_engagement_flag']

        # Also exclude binary/indicator columns
        binary_cols = [col for col in numeric_cols if df[col].nunique() <= 2]
        exclude.extend(binary_cols)

        return [col for col in numeric_cols if col not in exclude]

    def get_feature_names(self) -> List[str]:
        """Get names of features after transformation."""
        if hasattr(self, 'feature_names_'):
            return self.feature_names_
        return []

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None, **fit_params) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)
