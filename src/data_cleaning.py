import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/online_retail_II.xlsx")
PROCESSED_PATH = Path("data/processed/transactions_clean.csv")


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load and concatenate both sheets of the raw workbook."""
    sheets = pd.read_excel(path, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    df["IsCancelled"] = df["Invoice"].astype(str).str.startswith("C")

    df = df.dropna(subset=["Customer ID"])
    df["Customer ID"] = df["Customer ID"].astype(int)

    non_product_codes = ["POST", "D", "DOT", "M", "BANK CHARGES", "PADS", "CRUK"]
    df = df[~df["StockCode"].astype(str).isin(non_product_codes)]

    df = df[(df["Price"] > 0) & (df["Quantity"] != 0)]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["Revenue"] = df["Quantity"] * df["Price"]

    df = df.rename(columns={"Customer ID": "CustomerID"})

    return df.reset_index(drop=True)


def main():
    print(f"Loading raw data from {RAW_PATH} ...")
    raw = load_raw()
    print(f"Raw shape: {raw.shape}")

    clean = clean_transactions(raw)
    print(f"Clean shape: {clean.shape}")
    print(f"Unique customers: {clean['CustomerID'].nunique()}")
    print(f"Date range: {clean['InvoiceDate'].min()} to {clean['InvoiceDate'].max()}")

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved cleaned data to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()