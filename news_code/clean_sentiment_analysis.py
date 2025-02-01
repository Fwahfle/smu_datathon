import pandas as pd
import string
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

file_path = "dataset/news_excerpts_parsed.xlsx" # Change this to your actual file path
df = pd.read_excel(file_path, sheet_name="Sheet1")

sia = SentimentIntensityAnalyzer()

def clean_text(text): 
# Function to clean text by removing punctuation and converting to lowercase
    if isinstance(text, str):
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.lower()
    return ""

df["Cleaned_Text"] = df["Text"].astype(str).apply(clean_text)

# Extract domain name from the "Link" column to classify articles by source
df["Source"] = df["Link"].apply(lambda x: x.split('/')[2] if pd.notnull(x) 
                                else "Unknown")

# Function to get sentiment score and label using VADER
def get_sentiment(text):
    sentiment_score = sia.polarity_scores(text)["compound"] 
    # Classify sentiment based on the compound score
    if sentiment_score > 0.05:
        sentiment_label = "Positive"
    elif sentiment_score < -0.05:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"
    return sentiment_score, sentiment_label

df["Sentiment_Score"], df["Sentiment_Label"] = zip(*df["Cleaned_Text"].apply(get_sentiment))

output_file = "dataset/cleaned_sentiment_data.xlsx"
df.to_excel(output_file, index=False)

pd.set_option('display.max_rows', 50)

print(df)  # Display the entire DataFrame or up to 50 rows

print(f"\n Data processing complete! The cleaned file is saved as '{output_file}'.")