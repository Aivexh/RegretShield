
"""Configuration and constants for Regret Shield system."""

import numpy as np

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Model thresholds
RETURN_PROB_THRESHOLD = 0.6

# Dataset parameters
N_USERS = 5000
N_PRODUCTS = 1000
N_INTERACTIONS = 50000

# Categories
CATEGORIES = ['electronics', 'clothing', 'home',
              'books', 'beauty', 'sports', 'toys', 'automotive']

# Intervention messages
WARNING_MESSAGE = "⚠️ Customers with similar behavior returned this item {return_rate:.0f}% of the time."

# Recommendation weights
REC_SIMILARITY_BONUS = 0.3
REC_PRICE_PENALTY = 0.0005
