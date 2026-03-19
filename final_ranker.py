import math
import re
import pandas as pd
import streamlit as st


class StockPulseSentiment:
    def __init__(self, df, alias_map):
        self.df = df.copy()
        self.alias_map = {k.upper(): v for k, v in alias_map.items()}

        # Required text columns
        if "title" not in self.df.columns:
            self.df["title"] = ""
        if "body" not in self.df.columns:
            self.df["body"] = ""

        self.df["title"] = self.df["title"].fillna("").astype(str)
        self.df["body"] = self.df["body"].fillna("").astype(str)
        self.df["score"] = pd.to_numeric(self.df.get("score", 0), errors="coerce").fillna(0)
        self.df["comms_num"] = pd.to_numeric(self.df.get("comms_num", 0), errors="coerce").fillna(0)

        if "url" not in self.df.columns:
            self.df["url"] = ""

        self.df["combined_text"] = (self.df["title"] + " " + self.df["body"]).str.strip()

        self.word_scores = {
            "bull": 1.4,
            "bullish": 2.0,
            "buy": 1.3,
            "long": 0.9,
            "calls": 1.6,
            "call": 1.3,
            "undervalued": 1.7,
            "beat": 1.5,
            "beats": 1.5,
            "green": 0.8,
            "rally": 1.3,
            "rip": 1.1,
            "moon": 2.0,
            "mooning": 2.0,
            "rocket": 1.8,
            "rockets": 1.8,
            "squeeze": 1.4,
            "hold": 0.4,
            "hodl": 0.8,
            "bear": -1.4,
            "bearish": -2.0,
            "sell": -1.3,
            "short": -1.0,
            "puts": -1.6,
            "put": -1.3,
            "overvalued": -1.7,
            "miss": -1.5,
            "missed": -1.5,
            "red": -0.8,
            "dump": -1.6,
            "crash": -2.0,
            "crashing": -2.0,
            "bagholder": -1.7,
            "bankrupt": -2.4,
            "bankruptcy": -2.4,
            "fraud": -2.2,
            "downgrade": -1.7,
            "plunge": -1.8,
            "tank": -1.8,
            "tanking": -1.8,
        }

        self.phrase_scores = {
            "buy the dip": 2.2,
            "short squeeze": 2.4,
            "to the moon": 2.6,
            "beats earnings": 2.1,
            "beat earnings": 2.1,
            "price target raised": 1.8,
            "going to zero": -2.8,
            "miss earnings": -2.1,
            "missed earnings": -2.1,
            "price target cut": -1.8,
            "dead cat bounce": -1.8,
            "sell the rip": -1.4,
        }

        self.negations = {"not", "no", "never", "isnt", "wasnt", "dont", "doesnt", "cant", "wont"}

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = text.replace("🚀", " rocket ")
        text = text.replace("🟢", " green ")
        text = text.replace("🔴", " red ")
        text = re.sub(r"[^a-z0-9$!\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _ticker_patterns(self, ticker):
        aliases = self.alias_map.get(ticker.upper(), [ticker.upper()])
        patterns = []

        for alias in aliases:
            alias = alias.strip()
            if not alias:
                continue

            if alias.startswith("$"):
                raw = alias[1:]
                patterns.append(re.compile(rf"(?<![A-Z])\$?{re.escape(raw)}(?![A-Z])", re.IGNORECASE))
            elif alias.isupper() and len(alias) <= 5:
                patterns.append(re.compile(rf"(?<![A-Z])\$?{re.escape(alias)}(?![A-Z])", re.IGNORECASE))
            else:
                patterns.append(re.compile(rf"\b{re.escape(alias.lower())}\b", re.IGNORECASE))

        return patterns

    def _matches_ticker(self, row, ticker):
        patterns = self._ticker_patterns(ticker)
        for pat in patterns:
            if pat.search(row["title"]) or pat.search(row["body"]):
                return True
        return False

    def _ticker_relevance(self, row, ticker):
        patterns = self._ticker_patterns(ticker)
        title_hits = 0
        body_hits = 0

        for pat in patterns:
            if pat.search(row["title"]):
                title_hits += 1
            if pat.search(row["body"]):
                body_hits += 1

        if title_hits == 0 and body_hits == 0:
            return 0.0

        relevance = 1.0
        if title_hits > 0:
            relevance += 0.8
        if body_hits > 0:
            relevance += 0.4
        relevance += 0.15 * min(title_hits + body_hits, 3)

        return relevance

    def _text_sentiment(self, text):
        text = self._clean_text(text)
        score = 0.0

        for phrase, value in self.phrase_scores.items():
            if phrase in text:
                score += value

        tokens = text.split()

        for i, tok in enumerate(tokens):
            if tok not in self.word_scores:
                continue

            val = self.word_scores[tok]
            window_start = max(0, i - 3)
            window = tokens[window_start:i]

            if any(w in self.negations for w in window):
                val *= -0.8

            score += val

        exclam_bonus = min(text.count("!"), 4) * 0.08
        score *= (1.0 + exclam_bonus)

        return math.tanh(score / 4.0)

    def _engagement_weight(self, row):
        score_part = 0.18 * math.log1p(max(row["score"], 0))
        comment_part = 0.12 * math.log1p(max(row["comms_num"], 0))
        return 1.0 + score_part + comment_part

    def analyze_ticker(self, ticker, top_k=5):
        ticker = ticker.upper()

        if ticker not in self.alias_map:
            return {
                "ticker": ticker,
                "label": "Unknown ticker",
                "score_0_to_100": None,
                "avg_sentiment": None,
                "posts_used": 0,
                "top_posts": []
            }

        candidates = self.df[self.df.apply(lambda r: self._matches_ticker(r, ticker), axis=1)].copy()

        if candidates.empty:
            return {
                "ticker": ticker,
                "label": "No data",
                "score_0_to_100": None,
                "avg_sentiment": None,
                "posts_used": 0,
                "top_posts": []
            }

        candidates["ticker_relevance"] = candidates.apply(lambda r: self._ticker_relevance(r, ticker), axis=1)
        candidates["text_sentiment"] = candidates["combined_text"].apply(self._text_sentiment)
        candidates["engagement_weight"] = candidates.apply(self._engagement_weight, axis=1)

        candidates["final_weight"] = candidates["ticker_relevance"] * candidates["engagement_weight"]
        candidates["contribution"] = candidates["text_sentiment"] * candidates["final_weight"]

        total_weight = candidates["final_weight"].sum()
        avg_sentiment = candidates["contribution"].sum() / total_weight if total_weight > 0 else 0.0

        if avg_sentiment > 0.15:
            label = "Bullish"
        elif avg_sentiment < -0.15:
            label = "Bearish"
        else:
            label = "Neutral"

        score_0_to_100 = round(50 + 50 * avg_sentiment, 1)

        candidates["display_rank"] = (
            0.55 * candidates["ticker_relevance"] +
            0.30 * candidates["engagement_weight"] +
            0.15 * candidates["text_sentiment"].abs()
        )

        top_posts = candidates.sort_values("display_rank", ascending=False).head(top_k)

        return {
            "ticker": ticker,
            "label": label,
            "score_0_to_100": score_0_to_100,
            "avg_sentiment": round(avg_sentiment, 4),
            "posts_used": int(len(candidates)),
            "top_posts": top_posts[
                ["title", "body", "score", "comms_num", "url", "text_sentiment", "ticker_relevance"]
            ].to_dict(orient="records")
        }

    def rank_all_tickers(self, top_k=25):
        rows = []

        for ticker in self.alias_map.keys():
            result = self.analyze_ticker(ticker, top_k=3)
            if result["posts_used"] > 0 and result["score_0_to_100"] is not None:
                rows.append({
                    "ticker": result["ticker"],
                    "label": result["label"],
                    "score_0_to_100": result["score_0_to_100"],
                    "avg_sentiment": result["avg_sentiment"],
                    "posts_used": result["posts_used"],
                })

        if not rows:
            return pd.DataFrame()

        ranked = pd.DataFrame(rows).sort_values(
            by=["score_0_to_100", "posts_used"],
            ascending=[False, False]
        ).reset_index(drop=True)

        return ranked.head(top_k)


DEFAULT_ALIAS_MAP = {
    "AAPL": ["AAPL", "$AAPL", "Apple"],
    "TSLA": ["TSLA", "$TSLA", "Tesla"],
    "NVDA": ["NVDA", "$NVDA", "Nvidia"],
    "AMD": ["AMD", "$AMD", "Advanced Micro Devices"],
    "AMZN": ["AMZN", "$AMZN", "Amazon"],
    "META": ["META", "$META", "Meta", "Facebook"],
    "MSFT": ["MSFT", "$MSFT", "Microsoft"],
    "GME": ["GME", "$GME", "Gamestop", "GameStop"],
    "SPY": ["SPY", "$SPY", "S&P 500", "sp500"],
}


@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)


@st.cache_resource
def build_model(df):
    return StockPulseSentiment(df, DEFAULT_ALIAS_MAP)


st.set_page_config(page_title="StockPulse", layout="wide")
st.title("📈 StockPulse")
st.caption("Sentiment scoring for stock discussion posts")

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    ticker = st.selectbox("Choose a ticker", options=list(DEFAULT_ALIAS_MAP.keys()))
    top_k = st.slider("Top posts to show", min_value=1, max_value=10, value=5)

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

df = load_data(uploaded_file)
model = build_model(df)

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

result = model.analyze_ticker(ticker, top_k=top_k)

st.subheader(f"Analysis for {result['ticker']}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Label", result["label"])
c2.metric("Score (0-100)", "N/A" if result["score_0_to_100"] is None else result["score_0_to_100"])
c3.metric("Average Sentiment", "N/A" if result["avg_sentiment"] is None else result["avg_sentiment"])
c4.metric("Posts Used", result["posts_used"])

st.markdown("### Top Matching Posts")
if result["top_posts"]:
    for i, post in enumerate(result["top_posts"], start=1):
        title = post["title"] if post["title"] else f"Post {i}"
        with st.expander(f"{i}. {title}"):
            st.write(f"**Body:** {post['body']}")
            st.write(f"**Sentiment:** {post['text_sentiment']:.3f}")
            st.write(f"**Ticker relevance:** {post['ticker_relevance']:.3f}")
            st.write(f"**Reddit score:** {post['score']}")
            st.write(f"**Comments:** {post['comms_num']}")
            if post["url"]:
                st.write(f"**URL:** {post['url']}")
else:
    st.write("No posts found for this ticker.")

st.subheader("Overall Ranking")
if st.button("Generate Ranking"):
    ranked_df = model.rank_all_tickers(top_k=len(DEFAULT_ALIAS_MAP))
    if ranked_df.empty:
        st.warning("No ranked results available.")
    else:
        st.dataframe(ranked_df, use_container_width=True)