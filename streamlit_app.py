# streamlit_app.py
"""Streamlit dashboard for Regret Shield."""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Regret Shield Dashboard", layout="wide")

st.title("🛡️ Regret Shield Dashboard")
st.markdown("Pre-purchase return prediction and intervention system")

# Sidebar
st.sidebar.header("Configuration")
API_URL = st.sidebar.text_input("API URL", "http://localhost:8000")
THRESHOLD = st.sidebar.slider("Intervention Threshold", 0.3, 0.9, 0.6, 0.05)

# Load sample data


@st.cache_data
def load_sample_data():
    """Load sample products and users."""
    import joblib
    artifacts = joblib.load('models/regret_shield_artifacts.pkl')
    return artifacts['products'], artifacts['users']


products_df, users_df = load_sample_data()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 System Performance")

    # Load evaluation metrics
    try:
        with open('models/evaluation_metrics.json', 'r') as f:
            metrics = json.load(f)

        model_names = list(metrics.keys())
        aucs = [metrics[m]['roc_auc'] for m in model_names]

        fig = px.bar(x=model_names, y=aucs, title="ROC-AUC by Model",
                     labels={'x': 'Model', 'y': 'ROC-AUC'})
        st.plotly_chart(fig, use_container_width=True)

        # Calibration curve
        fig2 = go.Figure()
        for model_name in model_names:
            cal = metrics[model_name]['calibration']
            fig2.add_trace(go.Scatter(x=cal['prob_pred'], y=cal['prob_true'],
                                      mode='lines+markers', name=model_name))
        fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                  name='Perfect Calibration', line=dict(dash='dash')))
        fig2.update_layout(title="Calibration Curves", xaxis_title="Predicted Probability",
                           yaxis_title="Actual Fraction")
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not load metrics: {e}")

with col2:
    st.subheader("🎯 Test Prediction")

    # User selection
    user_id = st.selectbox("Select User", users_df['user_id'].head(100))
    user_data = users_df[users_df['user_id'] == user_id].iloc[0]

    # Product selection
    product_id = st.selectbox(
        "Select Product", products_df['product_id'].head(50))
    product_data = products_df[products_df['product_id'] == product_id].iloc[0]

    # Interaction features
    st.markdown("**Behavioral Signals**")
    time_on_page = st.slider("Time on page (seconds)", 0, 300, 45)
    scroll_depth = st.slider("Scroll depth", 0.0, 1.0, 0.6)
    images_viewed = st.slider("Images viewed", 0, 20, 4)
    review_hover = st.slider("Review hovers", 0, 20, 3)

    if st.button("Predict Return Risk", type="primary"):
        # Call API
        payload = {
            "user_id": int(user_id),
            "product_id": int(product_id),
            "time_on_page": time_on_page,
            "scroll_depth": scroll_depth,
            "num_images_viewed": images_viewed,
            "review_hover_count": review_hover,
            "add_to_wishlist": 0,
            "cart_abandonment": 0
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            result = response.json()

            # Display results
            st.markdown("---")
            st.subheader("🔮 Prediction Result")

            # Risk gauge
            risk = result['return_probability']
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk * 100,
                title={'text': "Return Risk (%)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if risk > THRESHOLD else "green"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgreen"},
                        {'range': [60, 85], 'color': "yellow"},
                        {'range': [85, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': THRESHOLD * 100
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

            # Decision
            if result['decision'] == 'allow':
                st.success("✅ Decision: ALLOW purchase")
            elif result['decision'] == 'warn':
                st.warning(
                    f"⚠️ Decision: WARNING - {result['warning_message']}")
            else:
                st.error(f"🚫 Decision: BLOCKED - {result['warning_message']}")

            # Risk factors
            if result.get('risk_factors'):
                st.subheader("📋 Key Risk Factors")
                risk_factors = result['risk_factors'].get(
                    'top_risk_factors', [])
                for factor in risk_factors[:5]:
                    st.write(
                        f"• {factor['feature']}: {factor['shap_value']:.3f}")

            # Alternatives
            if result.get('alternatives'):
                st.subheader("🔄 Recommended Alternatives")
                alt_df = pd.DataFrame(result['alternatives'])
                alt_df['return_risk_pct'] = alt_df['return_risk'] * 100
                st.dataframe(
                    alt_df[['product_id', 'category', 'price', 'return_risk_pct', 'rating']])

        except Exception as e:
            st.error(f"API call failed: {e}")

# Business impact simulation
st.subheader("💼 Business Impact Simulation")

# Simulate based on threshold
actual_returns = np.random.beta(2, 8, 1000)  # Simulated
predicted_returns = actual_returns + np.random.normal(0, 0.1, 1000)
predicted_returns = np.clip(predicted_returns, 0, 1)

high_risk = predicted_returns > THRESHOLD
intervention_effectiveness = st.slider(
    "Intervention Effectiveness", 0.0, 1.0, 0.6)

prevented = (high_risk * (actual_returns > 0.5)).sum() * \
    intervention_effectiveness
total_returns = (actual_returns > 0.5).sum()
reduction = prevented / total_returns if total_returns > 0 else 0

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Return Rate Reduction", f"{reduction*100:.1f}%")
with col2:
    st.metric("Interventions/Hour", f"{int(high_risk.sum() / 10)}")
with col3:
    st.metric("Estimated Monthly Savings", f"${int(reduction * 10000):,}")

# Product risk heatmap
st.subheader("🔥 Product Risk Heatmap")
category_risk = products_df.groupby(
    'category')['return_rate_by_category'].mean().reset_index()
fig = px.treemap(category_risk, path=['category'], values='return_rate_by_category',
                 title="Return Risk by Category", color='return_rate_by_category',
                 color_continuous_scale='Reds')
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown(
    "🚀 **Regret Shield v1.0** - Machine Learning System for Return Prediction")
