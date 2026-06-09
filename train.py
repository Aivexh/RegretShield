"""Main training script for Regret Shield system."""

from src.config import RANDOM_SEED
from src.recommender import AlternativeRecommender
from src.intervention_engine import InterventionEngine
from src.model import ReturnPredictionModel
from src.feature_engineering import FeatureEngineer
from src.dataset_generator import SyntheticDataGenerator
import os
import json
import joblib
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sys
sys.path.append('.')


def main():
    print("=" * 60)
    print("REGRET SHIELD: Training Pipeline")
    print("=" * 60)

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # 1. Generate synthetic data
    print("\n[1/5] Generating synthetic dataset...")
    generator = SyntheticDataGenerator()
    users, products, interactions = generator.generate_full_dataset()

    print(
        f"Generated {len(users)} users, {len(products)} products, {len(interactions)} interactions")
    print(f"Return rate: {interactions['return'].mean()*100:.2f}%")

    # 2. Feature engineering
    print("\n[2/5] Engineering features...")
    feature_engineer = FeatureEngineer()

    # Merge with product categories for training
    train_data = interactions.merge(
        products[['product_id', 'category']], on='product_id', how='left')
    train_data = train_data.merge(users[['user_id'] + [f'affinity_{c}' for c in generator.categories]],
                                  on='user_id', how='left')

    # Fit and transform
    feature_engineer.fit(train_data, train_data['return'])
    X = feature_engineer.transform(train_data)
    y = train_data['return']

    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {X.columns.tolist()}")

    # 3. Train-test split
    print("\n[3/5] Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # 4. Train models
    print("\n[4/5] Training models...")
    model = ReturnPredictionModel(random_state=RANDOM_SEED)
    train_results = model.train(X_train, y_train)

    for name, metrics in train_results.items():
        print(f"  {name}: Train AUC = {metrics['train_auc']:.4f}")

    # 5. Evaluate models
    print("\n[5/5] Evaluating models...")
    eval_results = model.evaluate(X_test, y_test, threshold=0.6)

    best_model = None
    best_auc = 0
    for name, metrics in eval_results.items():
        print(f"\n  {name}:")
        print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"    Precision@0.6: {metrics['precision_at_threshold']:.4f}")
        print(f"    Recall@0.6: {metrics['recall_at_threshold']:.4f}")

        if metrics['roc_auc'] > best_auc:
            best_auc = metrics['roc_auc']
            best_model = name

    print(f"\n  Best model: {best_model} (AUC: {best_auc:.4f})")

    # 6. Feature importance - use best model instead of xgboost
    print("\n  Feature Importance (Best Model):")
    importance_df = model.get_feature_importance(best_model)
    print(importance_df.head(10).to_string(index=False))

    # 7. Business impact simulation
    print("\n  Business Impact Simulation:")
    y_pred_proba = model.predict_proba(X_test, best_model)
    intervention_engine = InterventionEngine(threshold=0.6)
    impact = intervention_engine.simulate_business_impact(
        y_test.values, y_pred_proba)

    print(
        f"    Return rate reduction: {impact['return_rate_reduction_pct']:.1f}%")
    print(f"    Interventions triggered: {impact['interventions_triggered']}")
    print(f"    Estimated savings: {impact['estimated_savings_formatted']}")

    # 8. Save artifacts
    print("\n  Saving artifacts...")
    artifacts = {
        'feature_engineer': feature_engineer,
        'model': model,
        'products': products,
        'users': users,
        'intervention_engine': intervention_engine,
        'best_model_name': best_model
    }
    joblib.dump(artifacts, 'models/regret_shield_artifacts.pkl')
    print("  Artifacts saved to models/regret_shield_artifacts.pkl")

    # Save evaluation metrics
    with open('models/evaluation_metrics.json', 'w') as f:
        json.dump(eval_results, f, indent=2)

    # Save feature importance separately
    importance_df.to_csv('models/feature_importance.csv', index=False)

    print("\n" + "=" * 60)
    print("Training complete! System ready for inference.")
    print(f"Best model: {best_model} (AUC: {best_auc:.4f})")
    print("=" * 60)

    return artifacts


if __name__ == "__main__":
    main()
