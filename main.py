import re
import math
import pandas as pd
from collections import defaultdict
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")


# -----------------------------
# 1. Hardcoded S&P 500 tickers
# -----------------------------
SP500_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","BRK.B","LLY","TSLA",
    "UNH","JPM","XOM","V","PG","AVGO","MA","HD","CVX","MRK","ABBV","PEP",
    "COST","KO","ADBE","CRM","WMT","MCD","BAC","AMD","NFLX","TMO","ACN",
    "LIN","ABT","ORCL","DHR","CMCSA","CSCO","WFC","TXN","VZ","INTU","QCOM",
    "PM","AMGN","RTX","NEE","HON","LOW","UNP","UPS","IBM","SPGI","INTC",
    "GS","ISRG","PLD","BLK","MDT","SBUX","ADP","CVS","CAT","DE","NOW",
    "GE","ELV","AMAT","CI","SYK","TJX","BA","SCHW","LMT","MMC","MO","PGR",
    "CB","AXP","GILD","TGT","BKNG","ZTS","MDLZ","FIS","C","ADSK","SO","CL",
    "ETN","DUK","BSX","REGN","HCA","NOC","VRTX","AON","EOG","APD","ITW",
    "SLB","PNC","ICE","CSX","KLAC","SNPS","FDX","EMR","USB","EW","GM",
    "FCX","MAR","PSA","MPC","ORLY","CDNS","NSC","PH","ROP","WM","AZO",
    "FTNT","AIG","CME","MSI","BDX","SHW","NXPI","GD","COF","HUM","APTV",
    "PSX","PAYX","TRV","ALL","EL","KMB","MS","PCAR","HLT","MNST",
    "ROST","CTAS","YUM","CHTR","AEP","SRE","DLR","PRU","VLO","SPG","O",
    "D","KMI","PEG","MET","ED","F","KR","IDXX","FAST","CTSH","AMP","OTIS",
    "ROK","DD","CMG","BIIB","VRSK","GWW","KHC","AFL","DFS","TEL","HES",
    "HPQ","WELL","EA","MTB","STZ","DAL","EXC","ON","XEL","ECL","WEC",
    "FANG","ARE","SBAC","ANET","AVB","VICI","OKE","PXD","FITB","EFX",
    "TSCO","DOW","CPRT","WTW","NUE","DTE","GLW","EXR","VMC","BR","RMD",
    "RSG","WBD","UAL","CCL","WY","HAL","LHX","TT","IQV","CHD","KEYS",
    "MLM","XYL","RF","PCG","TTWO","FTV","ETSY","BKR","CARR","EBAY","HIG",
    "LEN","IR","GRMN","AWK","PPL","MKC","DHI","PAYC","MTCH","DOV","NTRS",
    "SWKS","FE","CMS","WST","TSN","IFF","ZBH","CTVA","INVH","JBHT","EXPD",
    "WRB","AES","STE","MOH","TRMB","HOLX","TDY","VTR","ATO","UDR","COO",
    "J","STT","CNP","NRG","ESS","WAT","PKG","DRI","BBY","CLX","LH","AEE",
    "DG","CINF","MAA","RJF","EQR","ZBRA","LUV","OMC","ALGN","NVR","FDS",
    "FOX","FOXA","K","IP","KIM","REG","HRL","PNR","CBOE","AOS","TECH",
    "WHR","EMN","EXPE","AIZ","HST","BEN","SEE","FMC","BIO","CTLT","NWS",
    "NWSA","RL","CPB","LW","MGM","GNRC","TER","POOL","XRAY","AKAM","APA",
    "MOS","IVZ","TAP","DXC","CE","UHS","HSIC","CZR","ENPH","FRT","PEAK",
    "ALB","HII","NDSN","QRVO","TFX","JKHY","KEY","LNT","NCLH","WRK","BWA",
    "GL","RE","MKTX","GPC","PFG","NI","ALLE","A","CAH","PNW","IPG","HWM",
    "MSCI","ODFL","ETR","LVS","LYV","ROL","MRO","FLT","AMCR","ZION","KMX",
    "RCL","NOV","HPE","CAG","ULTA","DLTR","MHK","TPR","VFC","TXT","PWR",
    "CPT","MAS","AAP","NWL","INCY","LEG","WYNN","AAL","ALK","OGN","PARA",
    "DISH","EVRG"
]

SP500_TICKERS = sorted(set(SP500_TICKERS))


# ------------------------------------------------------
# 2. Main class: inverted index + sentiment aggregation
# ------------------------------------------------------
class WSBStockSentimentIndex:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_excel(file_path)

        # Make sure expected columns exist
        expected = {"title", "score", "id", "url", "comms_num", "created", "body", "timestamp"}
        missing = expected - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")

        self.df = self.df.copy()

        # Combine title + body into one search field
        self.df["title"] = self.df["title"].fillna("").astype(str)
        self.df["body"] = self.df["body"].fillna("").astype(str)
        self.df["full_text"] = (self.df["title"] + " " + self.df["body"]).str.strip()

        # Optional: normalize weird encoding artifacts
        self.df["full_text"] = self.df["full_text"].str.replace("‚Äôt", "n't", regex=False)

        self.sia = SentimentIntensityAnalyzer()
        self._extend_lexicon()

        # Build regex once per ticker
        self.ticker_patterns = self._build_ticker_patterns()

        # Inverted index: ticker -> list of row indices
        self.inverted_index = defaultdict(list)

        # Build index
        self._build_inverted_index()

    # -----------------------
    # Sentiment customization
    # -----------------------
    def _extend_lexicon(self):
        extra = {
            "moon": 3.0,
            "mooning": 3.2,
            "bullish": 2.8,
            "bearish": -2.8,
            "buy": 1.5,
            "sell": -1.5,
            "calls": 2.0,
            "puts": -2.0,
            "rocket": 2.5,
            "rockets": 2.8,
            "tendies": 2.5,
            "bagholder": -2.5,
            "bagholding": -2.7,
            "undervalued": 2.2,
            "overvalued": -2.2,
            "pump": 1.2,
            "dump": -2.5,
            "crash": -3.0,
            "rip": -2.0,
            "hold": 1.0,
            "hodl": 1.8,
            "diamond hands": 2.5,
            "paper hands": -2.2,
            "squeeze": 2.0,
            "short squeeze": 3.0,
            "green": 1.5,
            "red": -1.5,
            "profit": 2.0,
            "profits": 2.0,
            "loss": -2.0,
            "losses": -2.2,
            "bull": 2.0,
            "bear": -2.0
        }
        self.sia.lexicon.update(extra)

    # --------------------------
    # Build ticker regex patterns
    # --------------------------
    def _build_ticker_patterns(self):
        patterns = {}

        for ticker in SP500_TICKERS:
            escaped = re.escape(ticker)

            # Let BRK.B also match BRK-B in text
            if "." in ticker:
                escaped_alt = re.escape(ticker.replace(".", "-"))
                pattern = re.compile(
                    rf"(?<![A-Za-z0-9])\$?(?:{escaped}|{escaped_alt})(?![A-Za-z0-9])",
                    re.IGNORECASE
                )
            else:
                pattern = re.compile(
                    rf"(?<![A-Za-z0-9])\$?{escaped}(?![A-Za-z0-9])",
                    re.IGNORECASE
                )

            patterns[ticker] = pattern

        return patterns

    # --------------------------
    # Build inverted index
    # --------------------------
    def _build_inverted_index(self):
        for idx, text in self.df["full_text"].items():
            if not isinstance(text, str) or not text.strip():
                continue

            upper_text = text.upper()

            for ticker in SP500_TICKERS:
                if self.ticker_patterns[ticker].search(upper_text):
                    self.inverted_index[ticker].append(idx)

    # --------------------------
    # Normalize ticker input
    # --------------------------
    def normalize_ticker(self, ticker):
        ticker = ticker.strip().upper().replace("$", "")
        ticker = ticker.replace("-", ".")
        return ticker

    # --------------------------
    # Retrieve posts/comments
    # --------------------------
    def retrieve(self, ticker):
        ticker = self.normalize_ticker(ticker)

        if ticker not in self.inverted_index:
            return self.df.iloc[0:0].copy()

        row_ids = self.inverted_index[ticker]
        return self.df.loc[row_ids].copy()

    # --------------------------
    # Extract local context
    # --------------------------
    def extract_context(self, text, ticker, window=1):
        """
        Split into sentences and keep the sentence containing the ticker,
        plus nearby sentences.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        ticker = self.normalize_ticker(ticker)
        pattern = self.ticker_patterns.get(ticker)
        if pattern is None:
            return text

        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences:
            return text

        hit_positions = []
        for i, sent in enumerate(sentences):
            if pattern.search(sent.upper()):
                hit_positions.append(i)

        if not hit_positions:
            return text

        selected = []
        for pos in hit_positions:
            start = max(0, pos - window)
            end = min(len(sentences), pos + window + 1)
            selected.extend(sentences[start:end])

        # dedupe while preserving order
        seen = set()
        deduped = []
        for s in selected:
            if s not in seen:
                deduped.append(s)
                seen.add(s)

        return " ".join(deduped).strip()

    # --------------------------
    # Sentiment score for text
    # --------------------------
    def score_text(self, text):
        if not isinstance(text, str) or not text.strip():
            return 0.0

        text = re.sub(r"\s+", " ", text).strip()

        vader = self.sia.polarity_scores(text)["compound"]
        rocket_bonus = text.count("🚀") * 0.08
        chart_up_bonus = text.count("📈") * 0.08
        chart_down_bonus = text.count("📉") * -0.08

        score = vader + rocket_bonus + chart_up_bonus + chart_down_bonus
        return max(-1.0, min(1.0, score))

    # --------------------------
    # Weight rows for aggregation
    # --------------------------
    def row_weight(self, row):
        score_weight = 1.0
        comment_weight = 1.0

        try:
            reddit_score = max(float(row.get("score", 0)), 0.0)
            score_weight = 1.0 + math.log1p(reddit_score)
        except Exception:
            pass

        try:
            comms = max(float(row.get("comms_num", 0)), 0.0)
            comment_weight = 1.0 + 0.25 * math.log1p(comms)
        except Exception:
            pass

        # Slightly downweight rows that are just "Comment" titles if you want
        title_text = str(row.get("title", "")).strip().lower()
        post_type_weight = 0.9 if title_text == "comment" else 1.0

        return score_weight * comment_weight * post_type_weight

    # --------------------------
    # Main sentiment analysis
    # --------------------------
    def analyze_ticker(self, ticker, top_k=5):
        ticker = self.normalize_ticker(ticker)
        results = self.retrieve(ticker)

        if results.empty:
            return {
                "ticker": ticker,
                "num_matches": 0,
                "avg_sentiment": None,
                "weighted_avg_sentiment": None,
                "label": "no data",
                "top_positive": [],
                "top_negative": []
            }

        # Use context around the ticker instead of whole post when possible
        results["context"] = results["full_text"].apply(lambda x: self.extract_context(x, ticker))
        results["sentiment"] = results["context"].apply(self.score_text)
        results["weight"] = results.apply(self.row_weight, axis=1)
        results["weighted_sentiment"] = results["sentiment"] * results["weight"]

        avg_sentiment = results["sentiment"].mean()
        weighted_avg = results["weighted_sentiment"].sum() / results["weight"].sum()

        if weighted_avg >= 0.2:
            label = "bullish"
        elif weighted_avg <= -0.2:
            label = "bearish"
        else:
            label = "neutral/mixed"

        top_positive = (
            results.sort_values("sentiment", ascending=False)
            [["id", "title", "body", "url", "score", "comms_num", "timestamp", "context", "sentiment"]]
            .head(top_k)
            .to_dict(orient="records")
        )

        top_negative = (
            results.sort_values("sentiment", ascending=True)
            [["id", "title", "body", "url", "score", "comms_num", "timestamp", "context", "sentiment"]]
            .head(top_k)
            .to_dict(orient="records")
        )

        return {
            "ticker": ticker,
            "num_matches": int(len(results)),
            "avg_sentiment": float(avg_sentiment),
            "weighted_avg_sentiment": float(weighted_avg),
            "label": label,
            "top_positive": top_positive,
            "top_negative": top_negative
        }

    # --------------------------
    # Optional: inspect index size
    # --------------------------
    def ticker_stats(self):
        rows = []
        for ticker in sorted(self.inverted_index.keys()):
            rows.append({
                "ticker": ticker,
                "num_docs": len(self.inverted_index[ticker])
            })
        return pd.DataFrame(rows).sort_values("num_docs", ascending=False)

    # --------------------------
    # Optional: search many tickers
    # --------------------------
    def rank_all_tickers_by_sentiment(self, min_mentions=5):
        summaries = []

        for ticker in SP500_TICKERS:
            if ticker not in self.inverted_index or len(self.inverted_index[ticker]) < min_mentions:
                continue

            summary = self.analyze_ticker(ticker, top_k=3)
            summaries.append({
                "ticker": summary["ticker"],
                "num_matches": summary["num_matches"],
                "weighted_avg_sentiment": summary["weighted_avg_sentiment"],
                "label": summary["label"]
            })

        if not summaries:
            return pd.DataFrame(columns=["ticker", "num_matches", "weighted_avg_sentiment", "label"])

        return pd.DataFrame(summaries).sort_values(
            ["weighted_avg_sentiment", "num_matches"],
            ascending=[False, False]
        )


# -----------------------------------
# 3. Example usage on your Excel file
# -----------------------------------
file_path = "Book1.xlsx"   # change if needed

indexer = WSBStockSentimentIndex(file_path)

# Example: analyze one ticker
summary = indexer.analyze_ticker("AAPL", top_k=3)

print("Ticker:", summary["ticker"])
print("Matches:", summary["num_matches"])
print("Average sentiment:", summary["avg_sentiment"])
print("Weighted average sentiment:", summary["weighted_avg_sentiment"])
print("Label:", summary["label"])

print("\nTop positive examples:")
for row in summary["top_positive"]:
    print(f"- sentiment={row['sentiment']:.3f} | title={row['title']} | context={row['context']}")

print("\nTop negative examples:")
for row in summary["top_negative"]:
    print(f"- sentiment={row['sentiment']:.3f} | title={row['title']} | context={row['context']}")


# Example: see which tickers appear most
stats_df = indexer.ticker_stats()
print("\nMost-mentioned tickers:")
print(stats_df.head(20))


# Example: rank all tickers by sentiment
ranked_df = indexer.rank_all_tickers_by_sentiment(min_mentions=3)
print("\nTop bullish tickers:")
print(ranked_df.head(10))

print("\nTop bearish tickers:")
print(ranked_df.tail(10))