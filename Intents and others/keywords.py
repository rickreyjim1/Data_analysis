import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re

# Load data
keywords_df = pd.read_excel('kws.xlsx')
intent_df = pd.read_excel('kw-intent.xlsx')

print(f"Keywords:\n {keywords_df.columns}")
print(f"Intents:\n {intent_df.columns}")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))

def preprocess(text):
    # Convert to string
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'\W', ' ', text)
    # Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Substitute multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # Lemmatize and remove stopwords
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stopwords]
    return ' '.join(text)

# Apply preprocess function to a list of words
def preprocess_list(word_list):
    return [preprocess(word) for word in word_list]



# Keep a copy of the original keywords
keywords_df['Original Keyword'] = keywords_df['Keyword']

# Preprocess keywords
keywords_df['Keyword'] = keywords_df['Keyword'].apply(preprocess)

# Split the "Keyword Modifiers" column into lists of words
intent_df["Keyword Modifiers"] = intent_df["Keyword Modifiers"].str.split(", ")

# Preprocess each word in the "Keyword Modifiers" column
intent_df["Keyword Modifiers"] = intent_df["Keyword Modifiers"].apply(preprocess_list)

# Save the processed data to Excel files
keywords_df.to_excel('preprocessed_kws.xlsx', index=False)
intent_df.to_excel('preprocessed_kw-intent.xlsx', index=False)

# Load the preprocessed data
keywords_df = pd.read_excel('preprocessed_kws.xlsx')
intent_df = pd.read_excel('preprocessed_kw-intent.xlsx')

# Convert the Keyword Modifiers column into lists of words

# Since the "Keyword Modifiers" column has been preprocessed and saved as a string, we need to evaluate the string to get the list back
intent_df["Keyword Modifiers"] = intent_df["Keyword Modifiers"].apply(eval)

# Initialize a new column in keywords_df to store the intents
keywords_df["Intent"] = "Uncategorized"

# Loop through each keyword
for i, keyword in keywords_df["Keyword"].items():
    # Initialize a list to store the intents of the current keyword
    intents = []
    
    # Convert keyword to string and split it into individual words
    keyword_words = set(str(keyword).split())
    
    # Loop through each intent
    for _, row in intent_df.iterrows():
        # Check if any word from the current intent appears in the keyword_words set
        if any(word in keyword_words for word in row["Keyword Modifiers"]):
            # If it does, append the intent to the list
            intents.append(row["Search Intent Type"])
    
    # If the list of intents is not empty, assign it to the Intent column of the current keyword
    if intents:
        keywords_df.at[i, "Intent"] = ", ".join(intents)

# Save the results to a new Excel file
keywords_df[['Original Keyword', 'Intent']].to_excel('keywords_with_intents.xlsx', index=False)