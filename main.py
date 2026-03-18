import re
import math
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")


class StockSentimentAnalyzer:
    def __init__(self, df, text_col="text", score_col=None, date_col=None):
        """
        df: pandas DataFrame containing Reddit data
        text_col: column with post/comment text
        score_col: optional column with upvotes / score
        date_col: optional column with timestamp
        """
        self.df = df.copy()
        self.text_col = text_col
        self.score_col = score_col
        self.date_col = date_col
        self.sia = SentimentIntensityAnalyzer()

        # Add some WSB-style lexicon tweaks
        self._add_wsb_lexicon()

    def _add_wsb_lexicon(self):
        """
        Extend VADER lexicon for finance / WallStreetBets slang.
        """
        new_words = {
            "moon": 3.0,
            "mooning": 3.2,
            "bullish": 2.8,
            "bearish": -2.8,
            "buy": 1.8,
            "sell": -1.8,
            "calls": 2.0,
            "puts": -2.0,
            "pump": 1.5,
            "dump": -2.5,
            "rocket": 2.5,
            "rockets": 2.8,
            "undervalued": 2.4,
            "overvalued": -2.4,
            "bagholder": -2.5,
            "bagholding": -2.7,
            "tendies": 2.5,
            "rip": -2.0,
            "crash": -3.0,
            "squeeze": 2.0,
            "short squeeze": 3.0,
            "long": 1.5,
            "short": -1.5,
            "profit": 2.2,
            "profits": 2.2,
            "loss": -2.2,
            "losses": -2.4,
            "green": 1.7,
            "red": -1.7,
            "diamond hands": 2.5,
            "paper hands": -2.2,
            "hold": 1.0,
            "hodl": 1.8,
            "rug": -3.0,
            "rugged": -3.2,
            "bull": 2.0,
            "bear": -2.0,
        }
        self.sia.lexicon.update(new_words)

    def _normalize_ticker(self, ticker):
        """
        Normalize ticker input like '$aapl' -> 'AAPL'
        """
        return ticker.strip().upper().replace("$", "")

    def _ticker_pattern(self, ticker):
        """
        Build regex to match ticker as a standalone token.
        Matches:
        - AAPL
        - $AAPL
        but avoids partial matches inside bigger words.
        """
        ticker = re.escape(self._normalize_ticker(ticker))
        return re.compile(rf"(?<![A-Za-z0-9])\$?{ticker}(?![A-Za-z0-9])", re.IGNORECASE)

    def _contains_ticker(self, text, ticker_regex):
        if not isinstance(text, str):
            return False
        return bool(ticker_regex.search(text))

    def _clean_text(self, text):
        """
        Light cleaning only. Keep emojis/punctuation because VADER uses them.
        """
        if not isinstance(text, str):
            return ""
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _score_text(self, text):
        """
        Return VADER compound score in [-1, 1]
        """
        text = self._clean_text(text)
        if not text:
            return 0.0

        # Extra WSB heuristics
        rocket_bonus = text.count("🚀") * 0.08
        down_bonus = text.count("📉") * -0.08
        up_bonus = text.count("📈") * 0.08

        vader_score = self.sia.polarity_scores(text)["compound"]
        final_score = vader_score + rocket_bonus + down_bonus + up_bonus

        # clamp to [-1,1]
        final_score = max(-1.0, min(1.0, final_score))
        return final_score

    def _weight_row(self, row):
        """
        Weight each post/comment.
        Default: 1
        If score_col exists: use log(1 + upvotes)
        """
        if self.score_col is None or self.score_col not in row or pd.isna(row[self.score_col]):
            return 1.0

        try:
            score = max(float(row[self.score_col]), 0.0)
            return 1.0 + math.log1p(score)
        except Exception:
            return 1.0

    def find_mentions(self, ticker):
        """
        Return all rows mentioning the ticker.
        """
        ticker = self._normalize_ticker(ticker)
        regex = self._ticker_pattern(ticker)

        mask = self.df[self.text_col].apply(lambda x: self._contains_ticker(x, regex))
        mentions = self.df.loc[mask].copy()

        if mentions.empty:
            return mentions

        mentions["clean_text"] = mentions[self.text_col].apply(self._clean_text)
        mentions["sentiment"] = mentions["clean_text"].apply(self._score_text)
        mentions["weight"] = mentions.apply(self._weight_row, axis=1)
        mentions["weighted_sentiment"] = mentions["sentiment"] * mentions["weight"]

        return mentions

    def summarize_sentiment(self, ticker):
        """
        Main function:
        input ticker -> output sentiment summary dict
        """
        ticker = self._normalize_ticker(ticker)
        mentions = self.find_mentions(ticker)

        if mentions.empty:
            return {
                "ticker": ticker,
                "num_mentions": 0,
                "average_sentiment": None,
                "weighted_average_sentiment": None,
                "sentiment_label": "no data",
                "top_positive_examples": [],
                "top_negative_examples": [],
            }

        avg_sentiment = mentions["sentiment"].mean()
        weighted_avg = mentions["weighted_sentiment"].sum() / mentions["weight"].sum()

        if weighted_avg >= 0.2:
            label = "bullish"
        elif weighted_avg <= -0.2:
            label = "bearish"
        else:
            label = "neutral/mixed"

        # strongest examples
        top_positive = (
            mentions.sort_values("sentiment", ascending=False)[[self.text_col, "sentiment"]]
            .head(3)
            .to_dict(orient="records")
        )

        top_negative = (
            mentions.sort_values("sentiment", ascending=True)[[self.text_col, "sentiment"]]
            .head(3)
            .to_dict(orient="records")
        )

        return {
            "ticker": ticker,
            "num_mentions": int(len(mentions)),
            "average_sentiment": float(avg_sentiment),
            "weighted_average_sentiment": float(weighted_avg),
            "sentiment_label": label,
            "top_positive_examples": top_positive,
            "top_negative_examples": top_negative,
        }