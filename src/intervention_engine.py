# src/intervention_engine.py
"""Intervention engine for warning and blocking purchases."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict


class InterventionEngine:
    """Decide whether to warn/block purchase and generate messages."""

    def __init__(self, threshold: float = 0.6, enable_auto_block: bool = True,
                 block_threshold: float = 0.85):
        """
        Initialize intervention engine.

        Args:
            threshold: Threshold for issuing warnings (default: 0.6)
            enable_auto_block: Whether to automatically block extremely high-risk purchases
            block_threshold: Threshold for auto-block (default: 0.85)
        """
        self.threshold = threshold
        self.enable_auto_block = enable_auto_block
        self.block_threshold = block_threshold

        # Statistics tracking
        self.intervention_stats = {
            'total_predictions': 0,
            'warnings_issued': 0,
            'blocks': 0,
            'allows': 0,
            'avg_return_risk_warned': 0,
            'avg_return_risk_allowed': 0,
            'warnings_by_category': defaultdict(int),
            'blocks_by_category': defaultdict(int),
            'intervention_history': []
        }

    def evaluate_purchase(self, return_probability: float,
                          product_info: Optional[Dict] = None,
                          user_info: Optional[Dict] = None,
                          interaction_features: Optional[Dict] = None) -> Dict:
        """
        Evaluate if purchase should be warned or blocked.

        Args:
            return_probability: Predicted return probability (0-1)
            product_info: Dictionary with product information (category, price, etc.)
            user_info: Dictionary with user information
            interaction_features: Dictionary with interaction features

        Returns:
            Dictionary with decision, message, and metadata
        """
        self.intervention_stats['total_predictions'] += 1

        decision = 'allow'
        message = None
        intervention_reason = []

        # Get product category for tracking
        category = product_info.get(
            'category', 'unknown') if product_info else 'unknown'

        # Decision logic
        if return_probability > self.threshold:
            decision = 'warn'
            self.intervention_stats['warnings_issued'] += 1
            self.intervention_stats['warnings_by_category'][category] += 1

            # Update average risk for warned purchases
            prev_avg = self.intervention_stats['avg_return_risk_warned']
            warned_count = self.intervention_stats['warnings_issued']
            self.intervention_stats['avg_return_risk_warned'] = (
                (prev_avg * (warned_count - 1) + return_probability) / warned_count
            )

            # Generate warning message
            return_rate_pct = return_probability * 100
            message = self._generate_warning_message(
                return_probability, product_info, user_info)
            intervention_reason.append(
                f"Return risk exceeds {self.threshold*100:.0f}% threshold")

            # Auto-block for extremely high risk
            if self.enable_auto_block and return_probability > self.block_threshold:
                decision = 'block'
                self.intervention_stats['blocks'] += 1
                self.intervention_stats['blocks_by_category'][category] += 1
                message += " 🔒 This purchase has been temporarily blocked due to very high return risk."
                intervention_reason.append(
                    f"Return risk exceeds auto-block threshold of {self.block_threshold*100:.0f}%")
        else:
            self.intervention_stats['allows'] += 1
            # Update average risk for allowed purchases
            prev_avg = self.intervention_stats['avg_return_risk_allowed']
            allowed_count = self.intervention_stats['allows']
            self.intervention_stats['avg_return_risk_allowed'] = (
                (prev_avg * (allowed_count - 1) +
                 return_probability) / allowed_count
            )

        # Log intervention
        intervention_record = {
            'timestamp': datetime.now().isoformat(),
            'return_probability': return_probability,
            'decision': decision,
            'category': category,
            'reasons': intervention_reason,
            'product_id': product_info.get('product_id') if product_info else None,
            'user_id': user_info.get('user_id') if user_info else None
        }
        self.intervention_stats['intervention_history'].append(
            intervention_record)

        # Keep only last 1000 records to prevent memory issues
        if len(self.intervention_stats['intervention_history']) > 1000:
            self.intervention_stats['intervention_history'] = self.intervention_stats['intervention_history'][-1000:]

        return {
            'decision': decision,
            'warning_message': message,
            'return_probability': return_probability,
            'threshold_used': self.threshold,
            'auto_blocked': decision == 'block',
            'intervention_reasons': intervention_reason,
            'timestamp': intervention_record['timestamp']
        }

    def _generate_warning_message(self, return_probability: float,
                                  product_info: Optional[Dict] = None,
                                  user_info: Optional[Dict] = None) -> str:
        """Generate personalized warning message."""
        return_rate_pct = return_probability * 100

        # Basic message
        message = f"⚠️ Customers with similar behavior returned this item {return_rate_pct:.0f}% of the time."

        # Add personalized advice based on product features
        if product_info:
            if product_info.get('requires_assembly', False):
                message += " This product requires assembly, which is a common return reason."

            price = product_info.get('price', 0)
            if price > 200:
                message += " Consider the high price point before purchasing."

        # Add user-specific advice
        if user_info:
            user_return_rate = user_info.get('avg_return_rate', 0)
            if user_return_rate > 0.3:
                message += " Given your return history, you might want to reconsider."

        return message

    def simulate_business_impact(self, baseline_returns: np.ndarray,
                                 predicted_returns: np.ndarray,
                                 intervention_effectiveness: float = 0.6,
                                 avg_return_cost: float = 50.0) -> Dict:
        """
        Simulate business impact of interventions.

        Args:
            baseline_returns: Actual return labels (0/1)
            predicted_returns: Predicted return probabilities
            intervention_effectiveness: How effective warnings are at preventing returns (0-1)
            avg_return_cost: Average cost of processing a return

        Returns:
            Dictionary with business impact metrics
        """
        baseline_returns = np.array(baseline_returns)
        predicted_returns = np.array(predicted_returns)

        # Identify high-risk transactions
        high_risk_mask = predicted_returns > self.threshold
        extreme_risk_mask = predicted_returns > self.block_threshold

        # Count interventions
        warnings_issued = high_risk_mask.sum()
        blocks_issued = extreme_risk_mask.sum() if self.enable_auto_block else 0

        # Without intervention, all returns happen
        total_returns_baseline = baseline_returns.sum()
        total_transactions = len(baseline_returns)
        baseline_return_rate = total_returns_baseline / total_transactions

        # With intervention, some returns are prevented
        # Assumption: Effectiveness applies only to warned purchases that would have been returned
        preventable_returns = high_risk_mask & (baseline_returns == 1)
        prevented_returns = preventable_returns.sum() * intervention_effectiveness

        # Auto-blocked purchases prevent 100% of returns
        if self.enable_auto_block:
            blocked_returns = extreme_risk_mask & (baseline_returns == 1)
            prevented_returns += blocked_returns.sum()

        total_returns_with_intervention = total_returns_baseline - prevented_returns
        return_rate_with_intervention = total_returns_with_intervention / total_transactions

        # Calculate impact
        absolute_reduction = total_returns_baseline - total_returns_with_intervention
        relative_reduction_pct = (
            absolute_reduction / total_returns_baseline) * 100 if total_returns_baseline > 0 else 0

        # Financial impact
        cost_savings = prevented_returns * avg_return_cost

        # Customer impact (assuming prevented returns = happier customers)
        unhappy_customers_prevented = prevented_returns * \
            0.7  # 70% of returns lead to unhappy customers

        return {
            'total_transactions': int(total_transactions),
            'baseline_return_rate': float(baseline_return_rate * 100),
            'return_rate_with_intervention': float(return_rate_with_intervention * 100),
            'return_rate_reduction_pct': float(relative_reduction_pct),
            'absolute_return_reduction': int(absolute_reduction),
            'prevented_returns': int(prevented_returns),
            'interventions_triggered': int(warnings_issued),
            'blocks_issued': int(blocks_issued),
            'intervention_rate_pct': float((warnings_issued / total_transactions) * 100),
            'estimated_cost_savings': float(cost_savings),
            'estimated_savings_formatted': f"${cost_savings:,.2f}",
            'unhappy_customers_prevented': int(unhappy_customers_prevented),
            'intervention_effectiveness_assumed': intervention_effectiveness,
            'avg_return_cost_assumed': avg_return_cost
        }

    def get_risk_tier(self, return_probability: float) -> str:
        """Get risk tier for a given probability."""
        if return_probability >= 0.8:
            return "Critical"
        elif return_probability >= 0.6:
            return "High"
        elif return_probability >= 0.4:
            return "Medium"
        elif return_probability >= 0.2:
            return "Low"
        else:
            return "Very Low"

    def get_stats(self) -> Dict:
        """Get intervention statistics."""
        stats = self.intervention_stats.copy()

        # Convert defaultdict to regular dict for serialization
        stats['warnings_by_category'] = dict(stats['warnings_by_category'])
        stats['blocks_by_category'] = dict(stats['blocks_by_category'])

        # Calculate additional metrics
        if stats['total_predictions'] > 0:
            stats['warning_rate_pct'] = (
                stats['warnings_issued'] / stats['total_predictions']) * 100
            stats['block_rate_pct'] = (
                stats['blocks'] / stats['total_predictions']) * 100
            stats['allow_rate_pct'] = (
                stats['allows'] / stats['total_predictions']) * 100
        else:
            stats['warning_rate_pct'] = 0
            stats['block_rate_pct'] = 0
            stats['allow_rate_pct'] = 0

        # Risk reduction potential
        if stats['avg_return_risk_warned'] > 0 and stats['avg_return_risk_allowed'] > 0:
            stats['risk_reduction_potential'] = (
                (stats['avg_return_risk_warned'] - stats['avg_return_risk_allowed']) /
                stats['avg_return_risk_warned'] * 100
            )
        else:
            stats['risk_reduction_potential'] = 0

        # Last 5 interventions
        stats['recent_interventions'] = stats['intervention_history'][-5:]

        return stats

    def reset_stats(self):
        """Reset all statistics."""
        self.intervention_stats = {
            'total_predictions': 0,
            'warnings_issued': 0,
            'blocks': 0,
            'allows': 0,
            'avg_return_risk_warned': 0,
            'avg_return_risk_allowed': 0,
            'warnings_by_category': defaultdict(int),
            'blocks_by_category': defaultdict(int),
            'intervention_history': []
        }

    def update_threshold(self, new_threshold: float):
        """Update the warning threshold dynamically."""
        if 0 <= new_threshold <= 1:
            self.threshold = new_threshold
            return True
        return False

    def should_intervene(self, return_probability: float,
                         user_history: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Quick check if intervention is needed.

        Returns:
            Tuple of (should_intervene, reason)
        """
        if return_probability > self.threshold:
            reason = f"Return probability ({return_probability:.2%}) exceeds threshold ({self.threshold:.0%})"

            # Additional checks based on user history
            if user_history and user_history.get('recent_returns', 0) > 2:
                reason += f" and user has {user_history['recent_returns']} recent returns"

            return True, reason

        return False, "Risk level acceptable"
