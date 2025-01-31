import pandas as pd
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the Excel file
file_path = "C:/Users/Admin/smu_datathon/cleaned_sentiment_data.xlsx"  # Update with actual file path
df = pd.read_excel(file_path)

# Function to extract entity relationships
def extract_entity_relationships(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    relations = []

    for token in doc:
        # Extract subject-verb-object triples
        if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
            subject = token.text
            verb = token.head.text
            obj = [child.text for child in token.head.children if child.dep_ in ("dobj", "attr")]
            if obj:
                relations.append((subject, verb, obj[0]))

    return entities, relations

# Apply extraction function to the "Cleaned_Text" column
df["Entities"], df["Relationships"] = zip(*df["Cleaned_Text"].apply(extract_entity_relationships))

# Save the extracted relationships to a new file
output_file = "extracted_entity_relationships.xlsx"
df.to_excel(output_file, index=False)

# Display preview
print(df[["Cleaned_Text", "Entities", "Relationships"]].head())

print(f"\n✅ Entity relationship extraction complete! 
      Data saved as '{output_file}'.")
