#step 2: sentiment analysis
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer() #initialise sentiment analyzer
processed_data = pd.read_csv('/Users/vrie4/Downloads/nodate5clean_wikileaks_parsed.csv') #change to your cleaned data file

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

#apply get_sentiment function
processed_data['sentiment'] = processed_data['cleaned_Text'].apply(get_sentiment)

output_file_path = '/Users/vrie4/Downloads/analysed_compound_wikileaks_parsed.csv'
processed_data.to_csv(output_file_path, index=False)

print("sentiment analysis scores recorded")