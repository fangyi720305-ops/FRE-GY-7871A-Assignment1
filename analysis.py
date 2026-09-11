## part 1
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.cwd()

universe = pd.read_csv(
    ROOT / "data/universe/universe.csv",
    dtype={"cik": str}
)

meta = pd.read_csv(
    ROOT / "data/interim/filings_meta.csv",
    dtype={"cik": str},
    parse_dates=["filing_date", "report_date", "acceptance_datetime"]
)

shares = pd.read_csv(
    ROOT / "data/prices/shares.csv",
    dtype={"cik": str}
)

prices = pd.read_csv(
    ROOT / "data/prices/prices.csv",
    index_col=0,
    parse_dates=True
)

volume = pd.read_csv(
    ROOT / "data/prices/volume.csv",
    index_col=0,
    parse_dates=True
)

## part 2
print("=== UNIVERSE ===")
print("Universe rows:", len(universe))
print(universe["status"].value_counts(dropna=False))
print("Domestic filers:", (universe["status"] == "domestic_filer").sum())
print()

print("=== FILINGS ===")
print("Total filings:", len(meta))
print("Unique firms:", meta["ticker"].nunique())
print("Unique accessions:", meta["accession"].nunique())
print("Duplicate accessions:", meta["accession"].duplicated().sum())
print()
print(meta["form"].value_counts())
print()
print(pd.crosstab(meta["filing_date"].dt.year, meta["form"]))
print()

print("Filing date range:",
      meta["filing_date"].min(),
      "to",
      meta["filing_date"].max())

print("Missing acceptance datetime:",
      meta["acceptance_datetime"].isna().sum())

print("Missing report date:",
      meta["report_date"].isna().sum())

print("Nonpositive word counts:",
      (meta["n_words"] <= 0).sum())

print("Word-count summary:")
print(meta.groupby("form")["n_words"].describe().round(1))
print()

print("=== TEXT FILES ===")
text_exists = meta["text_path"].map(
    lambda p: (ROOT / Path(p)).exists()
)
print("Text files found:", text_exists.sum())
print("Missing text files:", (~text_exists).sum())
print()

print("=== MARKET DATA ===")
print("Prices shape:", prices.shape)
print("Volume shape:", volume.shape)
print("Price dates:", prices.index.min(), "to", prices.index.max())

required_market = set(meta["ticker"]) | {"SPY", "ARKK", "^VIX"}
missing_price_tickers = sorted(
    t for t in required_market
    if t not in prices.columns or prices[t].notna().sum() == 0
)

print("Tickers without prices:", missing_price_tickers)
print()

share_match = meta["accession"].isin(shares["accession"])
print("Filings matched to share count:", share_match.sum())
print("Filings missing share count:", (~share_match).sum())
print("Share-count match rate:", f"{share_match.mean():.2%}")


# part 3
import re

print("\n=== DUPLICATE ACCESSIONS ===")

dup_rows = meta[
    meta["accession"].duplicated(keep=False)
].sort_values(["accession", "ticker"])

print(
    dup_rows[
        [
            "accession",
            "ticker",
            "cik",
            "company",
            "form",
            "filing_date",
            "report_date",
            "text_path",
        ]
    ].to_string(index=False)
)

print("\nDuplicate structure:")
dup_summary = (
    dup_rows.groupby("accession")
    .agg(
        rows=("accession", "size"),
        tickers=("ticker", lambda x: "|".join(sorted(set(x)))),
        n_tickers=("ticker", "nunique"),
        n_ciks=("cik", "nunique"),
        n_text_paths=("text_path", "nunique"),
    )
)

print(dup_summary.to_string())


print("\n=== ARK TICKERS NOT IN UNIVERSE.CSV ===")

raw = pd.read_csv(ROOT / "data/universe/ark_holdings_raw.csv")


def clean_ticker(raw_ticker):
    ticker = str(raw_ticker).strip().upper().split(" ")[0]

    if not ticker or ticker == "NAN":
        return None

    if "/" in ticker:
        return None

    if re.fullmatch(r"\d+", ticker):
        return None

    return ticker


raw["clean_ticker"] = raw["ticker"].map(clean_ticker)

clean_tickers = set(raw["clean_ticker"].dropna())
universe_tickers = set(universe["ticker"])

not_in_universe = sorted(clean_tickers - universe_tickers)

print("Raw rows:", len(raw))
print("Unique cleaned tickers:", len(clean_tickers))
print("Tickers represented in universe.csv:", len(universe_tickers))
print("Not represented in universe.csv:", len(not_in_universe))
print(not_in_universe)


print("\n=== NO 10-X FILERS ===")

no_10x = universe.loc[
    universe["status"] == "no_10x_filings",
    ["ticker", "ark_name", "sec_name", "cik", "n_10k", "n_10q"]
]

print(no_10x.to_string(index=False))


## part 4
print("\n=== ALPHABET SHARE-CLASS SELECTION ===")

alphabet_tickers = ["GOOG", "GOOGL"]

alphabet_dollar_volume = (
    prices[alphabet_tickers] * volume[alphabet_tickers]
).mean().sort_values(ascending=False)

print("Average daily dollar volume:")
print(alphabet_dollar_volume)

alphabet_primary = alphabet_dollar_volume.idxmax()

print("\nSelected Alphabet ticker:", alphabet_primary)

# part 5 meta_clean
duplicate_accessions = meta.loc[
    meta["accession"].duplicated(keep=False),
    "accession"
].unique()

meta_clean = meta.loc[
    ~meta["accession"].isin(duplicate_accessions)
    | (meta["ticker"] == alphabet_primary)
].copy()

meta_clean = (
    meta_clean
    .sort_values(["filing_date", "ticker", "accession"])
    .reset_index(drop=True)
)

# CIK is the appropriate firm identifier.
meta_clean["firm_id"] = meta_clean["cik"].str.zfill(10)

assert meta_clean["accession"].is_unique
assert len(meta_clean) == 1682
assert meta_clean["firm_id"].nunique() == 92
assert set(meta_clean["form"]) == {"10-K", "10-Q"}

print("\n=== CLEAN FILING SAMPLE ===")
print("Unique filings:", len(meta_clean))
print("Unique issuers:", meta_clean["firm_id"].nunique())
print("Unique tickers:", meta_clean["ticker"].nunique())
print()

print(meta_clean["form"].value_counts())
print()

print(pd.crosstab(
    meta_clean["filing_date"].dt.year,
    meta_clean["form"]
))

clean_share_match = meta_clean["accession"].isin(shares["accession"])

print("\nShare-count matches:", clean_share_match.sum())
print("Missing share counts:", (~clean_share_match).sum())
print("Match rate:", f"{clean_share_match.mean():.2%}")

## part 6 checking Loughran-McDonald Dictionary
from src.lexicons import load_master_dictionary, load_all

print("\n=== LOUGHRAN-MCDONALD DICTIONARY ===")

master = load_master_dictionary()
starter_word_lists = load_all()

print("Dictionary shape:", master.shape)
print("Columns:")
print(master.columns.tolist())

categories = ["Negative", "Uncertainty"]

for category in categories:
    flags = pd.to_numeric(
        master[category],
        errors="coerce"
    ).fillna(0)

    print(f"\n{category}:")
    print("  Positive flags:", (flags > 0).sum())
    print("  Negative/removed flags:", (flags < 0).sum())
    print("  All nonzero flags:", (flags != 0).sum())
    print("  Starter load_all() size:",
          len(starter_word_lists[category]))

    if (flags < 0).any():
        removed_words = master.loc[
            flags < 0,
            ["Word", category]
        ]

        print("  Removed words:")
        print(removed_words.to_string(index=False))
## part 7
word_lists = load_all()

negative_words = word_lists["Negative"]
uncertainty_words = word_lists["Uncertainty"]

overlap_words = negative_words & uncertainty_words
analysis_vocabulary = negative_words | uncertainty_words

print("\n=== ASSIGNMENT WORD LISTS ===")
print("Negative words:", len(negative_words))
print("Uncertainty words:", len(uncertainty_words))
print("Overlap:", len(overlap_words))
print("Combined vocabulary:", len(analysis_vocabulary))
print("Overlapping words:")
print(sorted(overlap_words))

assert len(negative_words) == 2355
assert len(uncertainty_words) == 297