from textblob import TextBlob

class SentimentAnalyzer:
    def analyze_sentiment(self, message):
        """
        Analyzes the sentiment of the message and returns a polarity score.
        Polarity ranges from -1 (negative) to 1 (positive).
        """
        try:
            analysis = TextBlob(message)
            return analysis.sentiment.polarity
        except Exception as e:
            # Handle errors gracefully
            return f"Error analyzing sentiment: {str(e)}"
