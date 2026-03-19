import re
import math
from collections import defaultdict

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as st

nltk.download("vader_lexicon")


# ============================================================
# 1. Hardcoded ticker vocabulary
#    Ambiguous tickers intentionally excluded.
# ============================================================
TICKERS = sorted(set([
    "AAPL","ABBV","ABT","ACN","ADBE","ADI","ADM","ADP","ADSK","AEE","AEP","AES",
    "AFL","AIG","AIZ","AJG","AKAM","ALB","ALGN","ALLE","AMAT","AMCR","AMD",
    "AMGN","AMP","AMT","AMZN","ANET","AON","AOS","APA","APD","APH","APTV",
    "ARE","ATO","AVB","AVGO","AVY","AWK","AXON","AXP","AZO","BA","BAC","BALL",
    "BAX","BBY","BDX","BEN","BF.B","BIIB","BK","BKNG","BLK","BMY","BR","BRK.B",
    "BRO","BSX","BWA","BX","BXP","CAG","CAH","CARR","CAT","CB","CBOE","CBRE",
    "CCI","CCL","CDNS","CDW","CE","CEG","CF","CFG","CHD","CHRW","CHTR","CI",
    "CINF","CL","CLX","CMA","CMCSA","CME","CMG","CMI","CMS","CNC","CNP","COF",
    "COO","COP","COR","COST","CPAY","CPB","CPRT","CPT","CRL","CRM","CSCO","CSGP",
    "CSX","CTAS","CTRA","CTSH","CTVA","CVS","CVX","CZR","DAL","DD","DE","DECK",
    "DFS","DG","DGX","DHI","DHR","DIS","DLR","DLTR","DOV","DOW","DPZ","DRI",
    "DUK","DVA","DVN","DXCM","EA","EBAY","ECL","ED","EFX","EG","EIX","EL","ELV",
    "EMN","EMR","ENPH","EOG","EPAM","EQIX","EQR","EQT","ERIE","ES","ESS","ETN",
    "ETR","EVRG","EW","EXC","EXPD","EXPE","EXR","FANG","FAST","FCX","FDX","FE",
    "FFIV","FI","FICO","FIS","FITB","FMC","FOX","FOXA","FRT","FTNT","FTV","GD",
    "GE","GEHC","GEN","GEV","GILD","GIS","GL","GLW","GM","GNRC","GOOG","GOOGL",
    "GPC","GPN","GRMN","GS","GWW","HAL","HAS","HCA","HD","HES","HIG","HII","HLT",
    "HOLX","HON","HPE","HPQ","HRL","HSIC","HST","HSY","HUBB","HUM","HWM","IBM",
    "ICE","IDXX","IFF","INCY","INTC","INTU","INVH","IP","IPG","IQV","IR","IRM",
    "ISRG","ITW","IVZ","JBHT","JBL","JKHY","JNJ","JPM","KDP","KEYS","KHC","KIM",
    "KKR","KLAC","KMB","KMI","KMX","KO","KR","KVUE","LDOS","LEN","LH","LHX","LIN",
    "LKQ","LLY","LMT","LNT","LOW","LRCX","LULU","LUV","LVS","LW","LYB","LYV",
    "MA","MAA","MAR","MAS","MCD","MCHP","MCK","MCO","MDLZ","MDT","MET","META",
    "MGM","MHK","MKC","MKTX","MLM","MMC","MMM","MNST","MO","MOH","MOS","MPC",
    "MRK","MRNA","MRO","MS","MSCI","MSFT","MTB","MTCH","MU","NCLH","NDAQ","NEE",
    "NEM","NFLX","NI","NKE","NOC","NOW","NRG","NSC","NTAP","NTRS","NUE","NVDA",
    "NVR","NWS","NWSA","NXPI","ODFL","OKE","OMC","ON","ORCL","ORLY","OTIS","OXY",
    "PANW","PARA","PAYC","PAYX","PCAR","PCG","PEG","PEP","PFE","PFG","PG","PGR",
    "PH","PHM","PKG","PLD","PM","PNC","PNR","PNW","PODD","POOL","PPG","PPL","PRU",
    "PSA","PSX","PTC","PWR","PYPL","QCOM","QRVO","RCL","REG","REGN","RF","RJF",
    "RL","RMD","ROK","ROL","ROP","ROST","RSG","RTX","RVTY","SBAC","SBUX","SCHW",
    "SEDG","SHW","SJM","SLB","SMCI","SNA","SNPS","SO","SOLV","SPG","SPGI","SRE",
    "STE","STT","STX","STZ","SW","SWK","SWKS","SYF","SYK","SYY","TAP","TDG",
    "TDY","TECH","TEL","TER","TFC","TFX","TGT","TJX","TMO","TMUS","TPL","TPR",
    "TRGP","TRMB","TROW","TRV","TSCO","TSLA","TSN","TT","TTWO","TXN","TXT","TYL",
    "UAL","UBER","UDR","UHS","ULTA","UNH","UNP","UPS","URI","USB","V","VICI","VLO",
    "VLTO","VMC","VRSK","VRSN","VRTX","VST","VTR","VTRS","VZ","WAB","WAT","WBA",
    "WBD","WDC","WEC","WELL","WFC","WM","WMB","WMT","WRB","WRK","WSM","WTW","WY",
    "WYNN","XEL","XOM","XYL","YUM","ZBH","ZBRA","ZTS"
]))


# ============================================================
# 2. Main engine
# ============================================================
class StockSentimentIndexer:
    def __init__(
        self,
        df: pd.DataFrame,
        title_col: str = "title",
        body_col: str = "body",
        score_col: str = "score",
        comments_col: str = "comms_num",
        timestamp_col: str = "timestamp",
        tickers=None,
    ):
        self.df = df.copy()
        self.title_col = title_col
        self.body_col = body_col
        self.score_col = score_col
        self.comments_col = comments_col
        self.timestamp_col = timestamp_col
        self.tickers = sorted(set(tickers if tickers is not None else TICKERS))

        for col in [title_col, body_col]:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        self.df[title_col] = self.df[title_col].fillna("").astype(str)
        self.df[body_col] = self.df[body_col].fillna("").astype(str)
        self.df["document"] = (
            self.df[title_col].str.strip() + " " + self.df[body_col].str.strip()
        ).str.strip()

        if score_col in self.df.columns:
            self.df[score_col] = pd.to_numeric(self.df[score_col], errors="coerce").fillna(0)
        if comments_col in self.df.columns:
            self.df[comments_col] = pd.to_numeric(self.df[comments_col], errors="coerce").fillna(0)

        self.sia = SentimentIntensityAnalyzer()
        self._extend_finance_lexicon()

        self.patterns = self._build_ticker_patterns()
        self.inverted_index = defaultdict(list)

    def _extend_finance_lexicon(self):
        finance_words = {
            "bullish": 2.8,
            "bearish": -2.8,
            "moon": 3.0,
            "mooning": 3.1,
            "rocket": 2.4,
            "rockets": 2.6,
            "buy": 1.5,
            "sell": -1.5,
            "calls": 1.8,
            "puts": -1.8,
            "undervalued": 2.2,
            "overvalued": -2.2,
            "bagholder": -2.5,
            "bagholding": -2.7,
            "tendies": 2.4,
            "hodl": 1.8,
            "hold": 1.0,
            "short squeeze": 3.0,
            "squeeze": 2.0,
            "crash": -3.0,
            "dump": -2.4,
            "pump": 1.3,
            "green": 1.2,
            "red": -1.2,
            "profit": 2.0,
            "profits": 2.0,
            "loss": -2.0,
            "losses": -2.2,
            "diamond hands": 2.5,
            "paper hands": -2.2,
            "beat earnings": 2.0,
            "missed earnings": -2.0,
            "guidance raise": 1.8,
            "guidance cut": -1.8,
        }
        self.sia.lexicon.update(finance_words)

    def _build_ticker_patterns(self):
        patterns = {}

        for ticker in self.tickers:
            escaped = re.escape(ticker)

            if "." in ticker:
                alt = re.escape(ticker.replace(".", "-"))
                pattern_body = rf"(?:{escaped}|{alt})"
            else:
                pattern_body = escaped

            patterns[ticker] = re.compile(
                rf"(?<![A-Za-z0-9])\$?{pattern_body}(?![A-Za-z0-9])",
                re.IGNORECASE,
            )

        return patterns

    def build_inverted_index(self):
        self.inverted_index = defaultdict(list)

        for row_idx, text in self.df["document"].items():
            if not isinstance(text, str) or not text.strip():
                continue

            text_upper = text.upper()
            for ticker, pattern in self.patterns.items():
                if pattern.search(text_upper):
                    self.inverted_index[ticker].append(row_idx)

        return self.inverted_index

    def get_posting_list(self, ticker: str):
        ticker = ticker.strip().upper().replace("$", "").replace("-", ".")
        return self.inverted_index.get(ticker, [])

    def extract_context(self, text: str, ticker: str, window: int = 1):
        if not isinstance(text, str) or not text.strip():
            return ""

        ticker = ticker.strip().upper().replace("$", "").replace("-", ".")
        pattern = self.patterns.get(ticker)
        if pattern is None:
            return text

        sentences = re.split(r"(?<=[.!?])\s+", text)
        hits = []

        for i, sent in enumerate(sentences):
            if pattern.search(sent.upper()):
                hits.append(i)

        if not hits:
            return text

        selected = []
        for i in hits:
            start = max(0, i - window)
            end = min(len(sentences), i + window + 1)
            selected.extend(sentences[start:end])

        seen = set()
        out = []
        for s in selected:
            s_clean = s.strip()
            if s_clean and s_clean not in seen:
                out.append(s_clean)
                seen.add(s_clean)

        return " ".join(out)

    def score_text(self, text: str):
        if not isinstance(text, str) or not text.strip():
            return 0.0

        text = re.sub(r"\s+", " ", text).strip()
        score = self.sia.polarity_scores(text)["compound"]
        score += text.count("🚀") * 0.08
        score += text.count("📈") * 0.06
        score -= text.count("📉") * 0.06

        return max(-1.0, min(1.0, score))

    def row_weight(self, row):
        weight = 1.0

        if self.score_col in row and pd.notna(row[self.score_col]):
            weight *= 1.0 + math.log1p(max(float(row[self.score_col]), 0.0))

        if self.comments_col in row and pd.notna(row[self.comments_col]):
            weight *= 1.0 + 0.2 * math.log1p(max(float(row[self.comments_col]), 0.0))

        return weight

    def analyze_ticker(self, ticker: str, context_window: int = 1, top_k: int = 5):
        ticker = ticker.strip().upper().replace("$", "").replace("-", ".")
        doc_ids = self.get_posting_list(ticker)

        if not doc_ids:
            return {
                "ticker": ticker,
                "mentions": 0,
                "avg_sentiment": None,
                "weighted_sentiment": None,
                "label": "no data",
                "top_positive_examples": [],
                "top_negative_examples": []
            }

        subset = self.df.loc[doc_ids].copy()
        subset["context"] = subset["document"].apply(
            lambda x: self.extract_context(x, ticker, window=context_window)
        )
        subset["sentiment"] = subset["context"].apply(self.score_text)
        subset["weight"] = subset.apply(self.row_weight, axis=1)
        subset["weighted_component"] = subset["sentiment"] * subset["weight"]

        avg_sentiment = subset["sentiment"].mean()
        weighted_sentiment = subset["weighted_component"].sum() / subset["weight"].sum()

        if weighted_sentiment >= 0.2:
            label = "bullish"
        elif weighted_sentiment <= -0.2:
            label = "bearish"
        else:
            label = "neutral/mixed"

        keep_cols = [c for c in [self.title_col, self.body_col, "context", "sentiment", self.score_col, self.comments_col, self.timestamp_col] if c in subset.columns]

        top_positive = (
            subset.sort_values("sentiment", ascending=False)[keep_cols]
            .head(top_k)
            .to_dict(orient="records")
        )

        top_negative = (
            subset.sort_values("sentiment", ascending=True)[keep_cols]
            .head(top_k)
            .to_dict(orient="records")
        )

        return {
            "ticker": ticker,
            "mentions": int(len(subset)),
            "avg_sentiment": float(avg_sentiment),
            "weighted_sentiment": float(weighted_sentiment),
            "label": label,
            "top_positive_examples": top_positive,
            "top_negative_examples": top_negative
        }

    def rank_all_tickers(self, min_mentions: int = 1, context_window: int = 1):
        rows = []

        for ticker in self.tickers:
            result = self.analyze_ticker(ticker, context_window=context_window, top_k=3)
            if result["mentions"] >= min_mentions:
                rows.append({
                    "ticker": result["ticker"],
                    "mentions": result["mentions"],
                    "avg_sentiment": result["avg_sentiment"],
                    "weighted_sentiment": result["weighted_sentiment"],
                    "label": result["label"]
                })

        ranked = pd.DataFrame(rows)

        if ranked.empty:
            return ranked

        return ranked.sort_values(
            by=["weighted_sentiment", "mentions"],
            ascending=[False, False]
        ).reset_index(drop=True)


# ============================================================
# 3. Streamlit UI
# ============================================================
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Please upload a CSV or XLSX file.")


@st.cache_resource
def build_engine(df):
    engine = StockSentimentIndexer(
        df,
        title_col="title",
        body_col="body",
        score_col="score",
        comments_col="comms_num",
        timestamp_col="timestamp",
        tickers=TICKERS,
    )
    engine.build_inverted_index()
    return engine


st.set_page_config(page_title="StockPulse", layout="wide")

st.title("📈 StockPulse")
st.caption("Sentiment ranking for S&P 500 stocks using WallStreetBets-style post data")

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    context_window = st.slider("Context window", min_value=0, max_value=3, value=1)
    min_mentions = st.number_input("Minimum mentions for full ranking", min_value=1, value=3, step=1)

if uploaded_file is None:
    st.info("Upload a CSV or XLSX dataset to begin.")
    st.stop()

df = load_data(uploaded_file)
engine = build_engine(df)

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Analyze a Stock")
user_prompt = st.text_input("Enter a stock ticker", placeholder="Example: AAPL")

if st.button("Get Sentiment"):
    if not user_prompt.strip():
        st.warning("Please enter a ticker.")
    else:
        result = engine.analyze_ticker(user_prompt, context_window=context_window, top_k=5)

        st.markdown(f"### Result for {result['ticker']}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mentions", result["mentions"])
        col2.metric("Average Sentiment", "N/A" if result["avg_sentiment"] is None else f"{result['avg_sentiment']:.3f}")
        col3.metric("Weighted Sentiment", "N/A" if result["weighted_sentiment"] is None else f"{result['weighted_sentiment']:.3f}")
        col4.metric("Label", result["label"])

        st.markdown("#### Top Positive Examples")
        if result["top_positive_examples"]:
            for ex in result["top_positive_examples"]:
                with st.expander(f"Sentiment: {ex['sentiment']:.3f}"):
                    if "title" in ex:
                        st.write(f"**Title:** {ex['title']}")
                    if "body" in ex:
                        st.write(f"**Body:** {ex['body']}")
                    if "context" in ex:
                        st.write(f"**Context:** {ex['context']}")
        else:
            st.write("No positive examples found.")

        st.markdown("#### Top Negative Examples")
        if result["top_negative_examples"]:
            for ex in result["top_negative_examples"]:
                with st.expander(f"Sentiment: {ex['sentiment']:.3f}"):
                    if "title" in ex:
                        st.write(f"**Title:** {ex['title']}")
                    if "body" in ex:
                        st.write(f"**Body:** {ex['body']}")
                    if "context" in ex:
                        st.write(f"**Context:** {ex['context']}")
        else:
            st.write("No negative examples found.")

st.subheader("Rank All Stocks")
if st.button("Generate Full Ranking"):
    ranked_df = engine.rank_all_tickers(min_mentions=min_mentions, context_window=context_window)

    if ranked_df.empty:
        st.warning("No stocks met the minimum mention threshold.")
    else:
        st.dataframe(ranked_df, use_container_width=True)

        st.markdown("#### Top 10 Most Bullish")
        st.dataframe(ranked_df.head(10), use_container_width=True)

        st.markdown("#### Top 10 Most Bearish")
        st.dataframe(ranked_df.tail(10), use_container_width=True)