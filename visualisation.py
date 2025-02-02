#step 3: visualisation (word cloud based on sentiment)
import pandas as pd
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt

data = pd.read_csv('/dataset/finalanalysed_wikileaks_parsed.csv')  #update to your file path if needed

text_col = data['cleaned_Text'] #get all text
positive_text = data[data['sentiment_type'] =='positive']['cleaned_Text'] #get positive text
negative_text = data[data['sentiment_type'] =='negative']['cleaned_Text'] #get negative text
neutral_text = data[data['sentiment_type'] =='neutral']['cleaned_Text'] #get neutral text

positive_combined_text = ' '.join(positive_text) #combine positive text
negative_combined_text = ' '.join(negative_text) #combine negative text
neutral_combined_text = ' '.join(neutral_text) #combine neutral text

def make_wordcloud(text, sentiment_type): #function to make word cloud
    wordcloud = WordCloud(width=700, height=350, max_font_size=100, max_words=100, background_color='white').generate(text)
    plt.figsize=(10, 5)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word cloud of words that have {sentiment_type} sentiment')
    plt.show()

make_wordcloud(positive_combined_text, 'positive') #positive word cloud
make_wordcloud(negative_combined_text, 'negative') #negative word cloud
make_wordcloud(neutral_combined_text, 'neutral') #neutral word cloud
