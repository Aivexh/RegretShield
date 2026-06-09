##JHBFCKJCHFHJFHDF
# 🛡️ Regret Shield

## AI-Powered Purchase Regret Prediction System

Regret Shield is a machine learning-powered return prevention platform that predicts whether a customer is likely to return a product before completing a purchase.

By analyzing user behavior, product characteristics, and interaction patterns, the system identifies high-risk purchases and triggers proactive interventions such as warnings and alternative recommendations.

---

## 📌 Overview

The platform is designed for:

* E-commerce platforms
* Online marketplaces
* Customer support systems
* Recommendation engines
* Checkout optimization workflows

---

## 🎯 Key Features

* User behavior analysis
* Product risk scoring
* Feature engineering pipeline
* Machine learning-based return prediction
* FastAPI REST API
* Streamlit dashboard
* Alternative product recommendation engine
* Business impact simulation
* Real-time intervention logic

---

## 🏗️ System Architecture

```text
User Behavior + Product Data
            ↓
     Feature Engineering
            ↓
     ML Prediction Layer
            ↓
    Intervention Engine
            ↓
      API + Dashboard
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/Aivexh/RegretShield.git
cd RegretShield
```

### Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🤖 Train the Model

```bash
python train.py
```

Example output:

```text
============================================================
REGRET SHIELD: Training Pipeline
============================================================

Generated 5000 users
Generated 1000 products
Generated 50000 interactions

Best model: logistic_regression
ROC-AUC: 0.7583

Training complete!
```

---

## ▶️ Run API Server

```bash
python api.py
```

API available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

## 📊 Launch Dashboard

```bash
streamlit run streamlit_app.py
```

Dashboard available at:

```text
http://localhost:8501
```

---

## 📡 API Endpoints

| Method | Endpoint | Description                |
| ------ | -------- | -------------------------- |
| GET    | /        | Root API                   |
| GET    | /health  | Health check               |
| POST   | /predict | Predict return probability |
| GET    | /stats   | System statistics          |
| GET    | /docs    | Swagger documentation      |

---

## Example Prediction Request

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "user_id": 123,
  "product_id": 456,
  "time_on_page": 45.5,
  "scroll_depth": 0.7,
  "num_images_viewed": 4,
  "review_hover_count": 3,
  "add_to_wishlist": 0,
  "cart_abandonment": 0
}'
```

---

## Example Response

```json
{
  "return_probability": 0.24,
  "decision": "allow",
  "warning_message": null,
  "alternatives": [
    {
      "product_id": 789,
      "category": "electronics",
      "price": 49.99,
      "return_risk": 0.18,
      "rating": 4.5
    }
  ],
  "threshold_used": 0.6
}
```

---

## 🤖 Machine Learning Models

| Model               | ROC-AUC | Precision |
| ------------------- | ------- | --------- |
| Logistic Regression | 0.758   | 0.700     |
| Gradient Boosting   | 0.758   | 0.710     |
| Random Forest       | 0.736   | 0.673     |

### Best Model

**Logistic Regression**

Performance:

* ROC-AUC: 0.7583
* Precision: 0.700
* Recall: 0.163

---

## 📈 Feature Importance

Top predictive features:

| Feature             | Importance |
| ------------------- | ---------- |
| user_return_rate    | 5.44       |
| product_risk_score  | 5.03       |
| low_engagement_flag | 0.40       |
| avg_scroll_depth    | 0.27       |
| engagement_score    | 0.25       |

---

## 🚨 Intervention Rules

| Risk Level | Action                 |
| ---------- | ---------------------- |
| > 85%      | Block Purchase         |
| 60% - 85%  | Warning + Alternatives |
| < 60%      | Allow Purchase         |

---

## 📈 Business Impact

Simulation Results (50,000 Transactions)

| Metric                         | Value   |
| ------------------------------ | ------- |
| Baseline Return Rate           | 26.1%   |
| Return Rate After Intervention | 23.2%   |
| Return Rate Reduction          | 11.3%   |
| Interventions Triggered        | 609     |
| Estimated Savings              | $14,780 |

---

## 📁 Project Structure

```text
regret-shield/
├── src/
│   ├── config.py
│   ├── dataset_generator.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── intervention_engine.py
│   └── recommender.py
│
├── train.py
├── api.py
├── streamlit_app.py
├── models/
├── requirements.txt
└── README.md
```

---

## 🧪 Testing

Health Check:

```bash
curl http://localhost:8000/health
```

Prediction Test:

```bash
curl -X POST "http://localhost:8000/predict"
```

---

## 🔧 Common Commands

Start Training:

```bash
python train.py
```

Run API:

```bash
python api.py
```

Run Dashboard:

```bash
streamlit run streamlit_app.py
```

---

* XGBoost integration
* Real-time checkout intervention
* Explainable AI predictions
* A/B testing framework
* Online learning pipeline
* Production deployment support

---

## 📄 License

MIT License

---

## 👤 Author

**Monika Yadav**

GitHub: https://github.com/Aivexh
