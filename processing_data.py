#step 1: preprocessing the data

import spacy
import pandas as pd
import re

model = spacy.load('en_core_web_sm')

def clean_data(text):

    text = text.strip() #remove leading/trailing spaces
    text = re.sub(r'(?<=\w)\s*/\s*(?=\w)', ' ', text) #specific case for '/': if means 'or', replace with space instead of completely removing if alphabetical character is before & aft / 
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) #remove all non-alphanumeric char
    text = re.sub(r'\s+', ' ', text) #remove extra spaces
    
    data = model(text) #tokenise text
    
    cleaned_tokens = [token.lemma_.lower() for token in data if not token.is_stop and not token.is_punct] #remove stopwords, lowercase, and lemmatize
    cleaned_data = ' '.join(cleaned_tokens) #join the cleaned tokens back into a single string

    return cleaned_data

input_file_path = '/Users/vrie4/Downloads/wikileaks_parsedcsv.csv' #load dataset (replace with your file path)
data = pd.read_csv(input_file_path)

data['cleaned_Text'] = data['Text'].apply(clean_data) #new col name

output_file_path = '/Users/vrie4/Downloads/cleaned_wikileaks_parsed.csv'
data.to_csv(output_file_path, index=False)

print("data processing done")