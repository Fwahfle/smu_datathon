import pandas as pd
from textblob import TextBlob


file_path = "dataset/news_excerpts_parsed.xlsx" # Write your own filepath here
df = pd.read_excel(file_path, sheet_name="Sheet1")

# We first define keywords to analyze
keywords = [
    "money laundering", "data privacy", "fraud", "corruption", "bribery", 
    "cybersecurity", "data breach", "financial crime", "tax evasion", "whistleblower",
    "insider trading", "market manipulation", "sanctions", "terrorist financing", "identity theft",
    "regulatory compliance", "risk management", "AML", "KYC", "forensic accounting",
    "financial misconduct", "embezzlement", "corporate fraud", "consumer protection", "ethics violations",
    "shell companies", "offshore accounts", "Ponzi scheme", "phishing attacks", "malware",
    "securities fraud", "money transfer scams", "investment fraud", "financial scandals", "misleading advertising",
    "fake news", "antitrust violations", "bank fraud", "wire fraud", "intellectual property theft",
    "insider leaks", "business ethics", "regulatory breaches", "white-collar crime", "data leaks",
    "hacking", "scam", "forgery", "deception", "unethical behavior"
]
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def analyze_keywords(df, keywords):
    results = []
    for index, row in df.iterrows():
        text = str(row['Text']).lower()
        sentiment = get_sentiment(text)
        for keyword in keywords:
            if keyword in text:
                results.append({
                    "Keyword": keyword,
                    "Sentiment": sentiment,
                    "Excerpt": row['Text'][:200],  # First 200 characters for context
                    "Article Link": row['Link']  # Include article link for Tableau visualization
                })
    return pd.DataFrame(results)

# Run the analysis
results_df = analyze_keywords(df, keywords)

# Save results to an Excel file for Tableau
output_path = "dataset/keyword_analysis_results.xlsx"
results_df.to_excel(output_path, index=False)

print(f"Analysis complete. Results saved to {output_path}")
