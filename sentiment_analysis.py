#step 2: sentiment analysis
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer() #initialise sentiment analyzer
processed_data = pd.read_csv('/dataset/cleaned_wikileaks_parsed.csv') #change to your cleaned data file

#create get_sentiment function
def get_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score > 0.05: #apply threshold to determine sentiment type
        sentiment = "positive"
    elif score < -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return score, sentiment

#apply get_sentiment function & create the 2 new cols
processed_data[['sentiment_score', 'sentiment_type']] = processed_data['cleaned_Text'].apply(lambda x: pd.Series(get_sentiment(x)))

output_file_path = '/dataset/finalanalysed_wikileaks_parsed.csv'
processed_data.to_csv(output_file_path, index=False)

print("sentiment analysis scores recorded")
