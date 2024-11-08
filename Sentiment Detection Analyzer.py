#Step 1 : Downloading And Importing Liberies

from textblob import TextBlob

#Step 2 : Analyzing

def analyze_sentiment(text):
  blob = TextBlob(text)
  polarity = blob.sentiment.polarity
  subjectivity = blob.sentiment.subjectivity
  if polarity > 0:
    sentiment = "Positive"
  elif polarity < 0:
    sentiment = "Negative"
  else:
    sentiment = "Neutral"
    return sentiment, polarity, subjectivity


#Step 3 : OutPut
def main():
    print("Welcome to the Sentiment Analyzer!")
    while True:
        text = input("\nEnter text to analyze (or type 'exit' to quit): ")
        if text.lower() == 'exit':
            break
        sentiment, polarity, subjectivity = analyze_sentiment(text)
        print(f"\nSentiment: {sentiment}")
        print(f"Polarity Score: {polarity}")
        print(f"Subjectivity Score: {subjectivity}")
        print("-" * 50)
if __name__ == "__main__":
  main()