import pandas as pd
import string
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Load the dataset from the Excel file
file_path = "C:/Users/Admin/smu_datathon-1/dataset/news_excerpts_parsed.xlsx"  # Change this to your actual file path
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to clean text by removing punctuation and converting to lowercase
def clean_text(text):
    if isinstance(text, str):  # Ensure input is a string
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        return text.lower()  # Convert text to lowercase
    return ""

# Apply text cleaning function to the "Text" column
df["Cleaned_Text"] = df["Text"].astype(str).apply(clean_text)

# Extract domain name from the "Link" column to classify articles by source
df["Source"] = df["Link"].apply(lambda x: x.split('/')[2] if pd.notnull(x) 
                                else "Unknown")

# Function to get sentiment score and label using VADER
def get_sentiment(text):
    sentiment_score = sia.polarity_scores(text)["compound"]  # Get VADER compound score
    # Classify sentiment based on the compound score
    if sentiment_score > 0.05:
        sentiment_label = "Positive"
    elif sentiment_score < -0.05:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"
    return sentiment_score, sentiment_label

# Apply sentiment analysis function to each row
df["Sentiment_Score"], df["Sentiment_Label"] = zip(*df["Cleaned_Text"].apply(get_sentiment))

# Save the cleaned and analyzed data to a new Excel file
output_file = "cleaned_sentiment_data.xlsx"
df.to_excel(output_file, index=False)

# Set pandas to display more rows (e.g., 50 rows)
pd.set_option('display.max_rows', 50)

# Print a preview of the processed data
print(df)  # Display the entire DataFrame or up to 50 rows

print(f"\n✅ Data processing complete! The cleaned file is saved as '{output_file}'.")