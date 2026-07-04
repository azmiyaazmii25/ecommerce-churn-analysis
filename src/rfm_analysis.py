"""
rfm_analysis.py
-----------------
Builds RFM (Recency, Frequency, Monetary) scores per customer and
segments them using K-Means clustering on log-transformed features.

Usage:
    python src/rfm_analysis.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

PROCESSED_PATH = Path("data/processed/transactions_clean.csv")
RFM_OUTPUT_PATH = Path("data/processed/customer_rfm_segments.csv")

SEGMENT_LABELS = {
    0: "Champions",
    1: "At Risk",
    2: "Lost",
    3: "New / Promising",
}


def build_rfm_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary value per customer."""
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
    ).reset_index()

    return rfm


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Add 1-4 quartile scores for each RFM dimension (4 = best)."""
    rfm = rfm.copy()
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
    return rfm


def cluster_customers(rfm: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> pd.DataFrame:
    """K-Means cluster customers on scaled, log-transformed RFM values."""
    rfm = rfm.copy()

    # Some customers can have net Monetary <= 0 if returns outweighed
    # purchases in this window. log1p is undefined for values < -1, so
    # clip at 0 before transforming (their spend is effectively negligible).
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    features["Frequency"] = np.log1p(features["Frequency"].clip(lower=0))
    features["Monetary"] = np.log1p(features["Monetary"].clip(lower=0))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(scaled)

    cluster_rank = (
        rfm.groupby("Cluster")["Monetary"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    rank_to_label = {cluster_id: SEGMENT_LABELS[i] for i, cluster_id in enumerate(cluster_rank)}
    rfm["Segment"] = rfm["Cluster"].map(rank_to_label)

    return rfm


def main():
    df = pd.read_csv(PROCESSED_PATH)

    rfm = build_rfm_table(df)
    rfm = score_rfm(rfm)
    rfm = cluster_customers(rfm)

    print(rfm["Segment"].value_counts())
    print("\nAverage RFM by segment:")
    print(rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(1))

    RFM_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rfm.to_csv(RFM_OUTPUT_PATH, index=False)
    print(f"\nSaved RFM segments to {RFM_OUTPUT_PATH}")


if __name__ == "__main__":
    main()