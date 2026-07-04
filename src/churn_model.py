"""
churn_model.py
-----------------
Predicts which customers are likely to churn using behavioral features
only (Frequency, Monetary) — deliberately excluding Recency-derived
features, since Recency is used to define the churn label itself.
Including it would leak the answer into the input (see README for details).

Definition used here: a customer is labeled "churned" if their Recency
(days since last purchase) exceeds 90 days as of the snapshot date.

Usage:
    python src/churn_model.py
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

RFM_PATH = Path("data/processed/customer_rfm_segments.csv")
CHURN_OUTPUT_PATH = Path("data/processed/customer_churn_predictions.csv")

CHURN_THRESHOLD_DAYS = 90


def label_churn(rfm: pd.DataFrame, threshold_days: int = CHURN_THRESHOLD_DAYS) -> pd.DataFrame:
    rfm = rfm.copy()
    rfm["Churned"] = (rfm["Recency"] > threshold_days).astype(int)
    return rfm


def train_churn_model(rfm: pd.DataFrame):
    # Deliberately excludes R_Score / Recency — see module docstring.
    features = ["Frequency", "Monetary", "F_Score", "M_Score"]
    X = rfm[features]
    y = rfm["Churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Classification report (test set):")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")

    return model, features


def score_all_customers(rfm: pd.DataFrame, model, features: list) -> pd.DataFrame:
    rfm = rfm.copy()
    rfm["ChurnProbability"] = model.predict_proba(rfm[features])[:, 1]
    return rfm


def main():
    rfm = pd.read_csv(RFM_PATH)
    rfm = label_churn(rfm)

    print(f"Churn rate (Recency > {CHURN_THRESHOLD_DAYS} days): "
          f"{rfm['Churned'].mean():.1%}")

    model, features = train_churn_model(rfm)
    scored = score_all_customers(rfm, model, features)

    # Revenue at risk = spend from customers not yet "churned" but with
    # high predicted churn probability — these are worth a retention
    # campaign, since already-churned customers are a sunk cost.
    at_risk = scored[(scored["Churned"] == 0) & (scored["ChurnProbability"] > 0.5)]
    revenue_at_risk = at_risk["Monetary"].sum()
    print(f"\nActive customers flagged as high churn risk: {len(at_risk)}")
    print(f"Revenue at risk from these customers: £{revenue_at_risk:,.2f}")

    CHURN_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(CHURN_OUTPUT_PATH, index=False)
    print(f"\nSaved churn predictions to {CHURN_OUTPUT_PATH}")


if __name__ == "__main__":
    main()