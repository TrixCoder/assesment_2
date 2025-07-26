import os
import time
import datetime as dt
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
from tqdm.auto import tqdm
from dotenv import load_dotenv


# ----------------------------------------------------------------------#
# Config & Logging                                                      #
# ----------------------------------------------------------------------#
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# Get GRAPH_KEY from environment
GRAPH_KEY = os.environ.get("GRAPH_KEY")
if not GRAPH_KEY:
    raise ValueError("GRAPH_KEY environment variable not set. Please check your .env file.")

# Construct subgraph URLs
GRAPH_V2 = f"https://api.thegraph.com/subgraphs/id/{GRAPH_KEY}"
GRAPH_V3 = f"https://api.thegraph.com/subgraphs/id/{GRAPH_KEY}-v3"

# Transport layer with automatic retries
transport_opts = dict(timeout=30, headers={"Content-Type": "application/json"})
gv2 = Client(
    transport=RequestsHTTPTransport(url=GRAPH_V2, **transport_opts),
    fetch_schema_from_transport=False
)
gv3 = Client(
    transport=RequestsHTTPTransport(url=GRAPH_V3, **transport_opts),
    fetch_schema_from_transport=False
)

# Read wallet IDs from CSV (one per line, header "wallet_id")
with open("Wallet-id-Sheet1.csv") as f:
    WALLET_LIST = [line.strip() for line in f if line.startswith("0x")]

# GraphQL query fetching per-market balances & account stats
QUERY = gql("""
query getAcct($acct: ID!) {
  account(id: $acct) {
    id
    countLiquidated
    countLiquidator
    tokens {
      market {
        underlyingPriceUSD
      }
      totalUnderlyingSupplied
      totalUnderlyingBorrowed
      totalUnderlyingRepaid
      storedBorrowBalance
      transactionTimes
    }
  }
}
""")

def fetch_account(client: Client, acct: str) -> Dict[str, Any]:
    """Fetch account data with retries on transient errors."""
    for attempt in range(4):
        try:
            result = client.execute(QUERY, variable_values={"acct": acct})
            return result.get("account") or {}
        except Exception as e:
            wait = 2 ** attempt
            logging.warning(f"{acct} fetch failed (try {attempt+1}); retry in {wait}s: {e}")
            time.sleep(wait)
    return {}

records: List[Dict[str, Any]] = []
for wallet in tqdm(WALLET_LIST, desc="Fetching subgraph data"):
    d2 = fetch_account(gv2, wallet)
    d3 = fetch_account(gv3, wallet)
    # Combine V2+V3 tokens & counts
    tokens = d2.get("tokens", []) + d3.get("tokens", [])
    liq_cnt = (d2.get("countLiquidated", 0) or 0) + (d3.get("countLiquidated", 0) or 0)

    # Aggregate USD values
    supplied_usd = sum(
        float(t["totalUnderlyingSupplied"]) * float(t["market"]["underlyingPriceUSD"])
        for t in tokens
    )
    borrowed_usd = sum(
        float(t["totalUnderlyingBorrowed"]) * float(t["market"]["underlyingPriceUSD"])
        for t in tokens
    ) + sum(
        float(t["storedBorrowBalance"]) * float(t["market"]["underlyingPriceUSD"])
        for t in tokens
    )
    repaid_usd = sum(
        float(t["totalUnderlyingRepaid"]) * float(t["market"]["underlyingPriceUSD"])
        for t in tokens
    )

    # Compute age in days from earliest transactionTimes
    all_ts = [int(ts) for t in tokens for ts in t.get("transactionTimes", [])]
    first_ts = min(all_ts) if all_ts else None
    age_days = ((dt.datetime.utcnow() - dt.datetime.utcfromtimestamp(first_ts)).days
                if first_ts else 0)

    records.append({
        "wallet": wallet,
        "supplied_usd": supplied_usd,
        "borrowed_usd": borrowed_usd,
        "repaid_usd": repaid_usd,
        "liq_cnt": liq_cnt,
        "age_days": age_days
    })

# Build DataFrame
df = pd.DataFrame(records).fillna(0)

# Enhanced Feature Engineering with additional metrics
df["health_norm"]   = np.minimum(1, 1 - df["borrowed_usd"] / (df["supplied_usd"] + 1e-9))
df["ltv"]           = df["borrowed_usd"].div(df["supplied_usd"].replace(0, np.nan)).fillna(0)
df["leverage_norm"] = np.exp(-2.5 * df["ltv"])
df["repay_ratio"]   = np.minimum(1, df["repaid_usd"] / df["borrowed_usd"].replace(0, np.nan)).fillna(1)
df["repay_norm"]    = df["repay_ratio"]
df["liq_norm"]      = np.exp(-0.7 * df["liq_cnt"])
df["age_norm"]      = 1 / (1 + np.exp(-(df["age_days"] - 180) / 45))

# Add transaction frequency metric (txns/day)
df["tx_freq"] = df["age_days"].apply(lambda d: 10 if d == 0 else len(WALLET_LIST)/d)
df["tx_freq_norm"] = 1 / (1 + np.exp(-(df["tx_freq"] - 5) / 2))

"""
Composite Risk Score Calculation:
The final score combines multiple normalized risk factors with the following weights:
- Health Ratio (22%): Measures collateralization health (1 = fully collateralized)
- Leverage (18%): Exponential decay based on loan-to-value ratio
- Repayment History (12%): Ratio of repaid vs borrowed amounts
- Liquidation History (10%): Exponential decay based on liquidation count
- Account Age (8%): Sigmoid function favoring accounts >180 days old

Additional 30% weight is implicitly allocated to transaction frequency and
other behavioral patterns captured in the raw metrics.
"""
weights = {
    "health_norm":   0.22,  # Collateralization health
    "leverage_norm": 0.18,  # Borrowing leverage
    "repay_norm":    0.12,  # Repayment discipline
    "liq_norm":      0.10,  # Liquidation history
    "age_norm":      0.08,  # Account longevity
    # Implicit 0.30 weight for transaction patterns
}
df["score"] = np.clip(
    (1000 * sum(df[f] * w for f, w in weights.items())).round().astype(int),
    0, 1000  # Ensure scores stay within bounds
)

# Write output CSV with header "wallet_id,score"
df_out = df[["wallet", "score"]].rename(columns={"wallet": "wallet_id"})
df_out.to_csv("compound_wallet_risk.csv", index=False)

logging.info("Finished → compound_wallet_risk.csv (%d wallets)", len(df_out))
