"""Synthetic dataset generator with realistic return patterns."""

import numpy as np
import pandas as pd
from typing import Tuple
from src.config import (
    N_USERS, N_PRODUCTS, N_INTERACTIONS, CATEGORIES, RANDOM_SEED
)

np.random.seed(RANDOM_SEED)


class SyntheticDataGenerator:
    """Generate realistic synthetic e-commerce data with return labels."""

    def __init__(self):
        self.n_users = N_USERS
        self.n_products = N_PRODUCTS
        self.n_interactions = N_INTERACTIONS
        self.categories = CATEGORIES

    def generate_users(self) -> pd.DataFrame:
        """Generate user profiles with behavioral traits."""
        users = pd.DataFrame({
            'user_id': range(self.n_users),
            # most low returners
            'avg_return_rate': np.random.beta(2, 5, self.n_users),
            'price_sensitivity_score': np.random.uniform(0, 1, self.n_users),
            # skewed to low
            'impulsiveness_score': np.random.beta(2, 3, self.n_users),
        })

        # Category affinity vectors (Dirichlet distribution)
        category_affinity = np.random.dirichlet(
            [1]*len(self.categories), self.n_users)
        for i, cat in enumerate(self.categories):
            users[f'affinity_{cat}'] = category_affinity[:, i]

        return users

    def generate_products(self) -> pd.DataFrame:
        """Generate product catalog with attributes."""
        products = pd.DataFrame({
            'product_id': range(self.n_products),
            'category': np.random.choice(self.categories, self.n_products),
            'price': np.random.lognormal(3, 0.8, self.n_products),
            'requires_assembly': np.random.choice([0, 1], self.n_products, p=[0.7, 0.3]),
            # mostly high ratings
            'rating_score': np.random.beta(4, 1, self.n_products),
        })

        # Add category-level return rates
        category_return_rates = {}
        for cat in self.categories:
            category_return_rates[cat] = np.random.beta(2, 10)

        products['return_rate_by_category'] = products['category'].map(
            category_return_rates)

        return products

    def generate_interactions(self, users: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
        """Generate user-product interactions with behavioral features."""
        interactions = []

        for _ in range(self.n_interactions):
            user = users.sample(1).iloc[0]
            product = products.sample(1).iloc[0]

            # Behavioral features with realistic correlations
            impulsiveness = user['impulsiveness_score']

            # Impulsive users spend less time, scroll more erratically
            time_on_page = np.random.exponential(
                30) if impulsiveness > 0.7 else np.random.exponential(60)
            scroll_depth = np.random.uniform(
                0.3, 1.0) if impulsiveness < 0.3 else np.random.uniform(0.1, 0.8)
            images_viewed = np.random.poisson(
                3) if impulsiveness > 0.5 else np.random.poisson(5)
            review_hover_count = np.random.poisson(
                2) if impulsiveness < 0.4 else np.random.poisson(0.5)
            add_to_wishlist = np.random.choice([0, 1], p=[
                                               0.9, 0.1]) if impulsiveness > 0.6 else np.random.choice([0, 1], p=[0.7, 0.3])
            cart_abandonment = np.random.choice([0, 1], p=[
                                                0.95, 0.05]) if time_on_page < 20 else np.random.choice([0, 1], p=[0.8, 0.2])

            interactions.append({
                'user_id': user['user_id'],
                'product_id': product['product_id'],
                'time_on_page': time_on_page,
                'scroll_depth': scroll_depth,
                'num_images_viewed': images_viewed,
                'review_hover_count': review_hover_count,
                'add_to_wishlist': add_to_wishlist,
                'cart_abandonment': cart_abandonment,
            })

        df_interactions = pd.DataFrame(interactions)

        # Add return labels
        df_interactions = self._generate_return_labels(
            df_interactions, users, products)

        return df_interactions

    def _generate_return_labels(self, interactions: pd.DataFrame,
                                users: pd.DataFrame,
                                products: pd.DataFrame) -> pd.DataFrame:
        """Generate realistic return labels based on probabilistic rules."""
        df = interactions.copy()

        # Merge user and product features
        df = df.merge(users, on='user_id', how='left')
        df = df.merge(products, on='product_id', how='left')

        # Calculate return probability based on multiple factors
        return_prob = np.zeros(len(df))

        # Rule 1: Impulsive users + expensive products
        expensive = df['price'] > df['price'].quantile(0.7)
        impulsive = df['impulsiveness_score'] > 0.7
        return_prob += (impulsive & expensive) * 0.4

        # Rule 2: Category mismatch (low affinity)
        for cat in self.categories:
            cat_mask = df['category'] == cat
            affinity_col = f'affinity_{cat}'
            low_affinity = df[affinity_col] < 0.1
            return_prob[cat_mask & low_affinity] += 0.3

        # Rule 3: Requires assembly + historically returns
        assembly_heavy = df['requires_assembly'] == 1
        high_returner = df['avg_return_rate'] > 0.3
        return_prob[assembly_heavy & high_returner] += 0.35

        # Rule 4: High category return rate
        return_prob += df['return_rate_by_category'] * 0.3

        # Rule 5: Behavioral signals
        low_engagement = (df['time_on_page'] < 20) & (df['scroll_depth'] < 0.3)
        return_prob[low_engagement] += 0.15

        # Add noise (5-15%)
        noise = np.random.uniform(-0.1, 0.1, len(df))
        return_prob = np.clip(return_prob + noise, 0, 1)

        # Generate binary labels
        df['return'] = (np.random.random(len(df)) < return_prob).astype(int)

        # Keep only necessary columns
        keep_cols = ['user_id', 'product_id', 'time_on_page', 'scroll_depth',
                     'num_images_viewed', 'review_hover_count', 'add_to_wishlist',
                     'cart_abandonment', 'return']

        return df[keep_cols]

    def generate_full_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate complete synthetic dataset."""
        print("Generating users...")
        users = self.generate_users()

        print("Generating products...")
        products = self.generate_products()

        print("Generating interactions...")
        interactions = self.generate_interactions(users, products)

        return users, products, interactions
