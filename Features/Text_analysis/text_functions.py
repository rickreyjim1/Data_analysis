#Import Required Libraries

import re
import nltk
#from nltk.corpus import stopwords
# nltk.download('punkt_tab')   # Download resources if needed
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

#---------------------------------------- Data preparation ----------------------------------------#
#---------------------------------------- Data cleaning ----------------------------------------#
#stop_words = list(stopwords.words('english'))  # Get stopwords for English
lemmatizer = WordNetLemmatizer()

#---------------------------------------- Function to obtain the common words ----------------------------------------#
def get_common_words(col):
    """Get the most common words
    from Data"""
    cnt = Counter()
    for text in col.values:
        for word in text.split():
            cnt[word] += 1
    return cnt.most_common()

#---------------------------------------- Functions to remove stop/frequent/rare words and lemmatize ----------------------------------------#
def stem_sentence(sentence):
    """Convert Data into Words & 
    Apply Lemmatization on words to find meaningful words"""
    token_words = word_tokenize(sentence)
    token_words = [word for word in token_words if word not in stop_words]
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(lemmatizer.lemmatize(word))
    return " ".join(stem_sentence)

def remove_freqwords(sentence):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(sentence).split() if word not in FREQWORDS])

def remove_rarewords(text):
    """custom function to remove the rare words"""
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])

#---------------------------------------- Function to remove calendar words ----------------------------------------#
def remove_calendar(text):
    """Remove date, time, names, and metadata from work notes, keeping only the main text."""
    if not isinstance(text, str):
        return text  # Return as is if it's not a string

    # Step 1: Remove date and time (e.g., '2024-08-09 15:58:23 - ')
    text = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - ', '', text)

    # Step 2: Remove names and roles (e.g., 'Marta Switala [Capgemini] (Work notes)')
    text = re.sub(r'[A-Za-z\s]+$$\w+$$\s$[^)]+$', '', text)

    # Step 3: Remove extra whitespace and newlines
    text = text.strip()  # Remove leading/trailing spaces
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with a single newline

    return text

#---------------------------------------- Function to remove greetings and preprocess text ----------------------------------------#
def remove_greetings(text):
    """Remove greetings from data"""
    greetings = ['good afternoon', 'good morning', 'good evening', 'good night', 'hi', 'hi team']
    for tags in greetings:
        return re.sub(tags, '', text)
    
def process_text(text):
    """
    Cleans and preprocesses the given text, removing extra symbols, handling contractions,
    and applying other cleaning techniques.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned and preprocessed text.
    """
    
    # Convert to lowercase
    if isinstance(text, str):
        return text.lower()
    else:
        return text  # Return the original value if it's not a string
    
    # Remove extra spaces and newlines
    #text = re.sub(r'\s+', ' ', text)

    # Handle common abbreviations and contractions
    text = text.replace("pw", "password")
    text = text.replace("e.g.", "for example")
    text = text.replace("i.e.", "that is")
    text = text.replace("etc.", "and so on")
    text = text.replace("can't", "cannot")
    text = text.replace("won't", "will not")
    text = text.replace("isn't", "is not")
    text = text.replace("aren't", "are not")
    text = text.replace("didn't", "did not")
    text = text.replace("doesn't", "does not")
    text = text.replace("don't", "do not")
    text = text.replace("hadn't", "had not")
    text = text.replace("hasn't", "has not")
    text = text.replace("haven't", "have not")
    text = text.replace("isn't", "is not")
    text = text.replace("wasn't", "was not")
    text = text.replace("weren't", "were not")
    text = text.replace("won't", "will not")
    text = text.replace("wouldn't", "would not")

    # Handle possessive apostrophes
    text = re.sub(r"([A-Za-z]+)'s", r"\1 is", text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers and dates
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\d{4}-\d{2}-\d{2}', '', text)

    # Remove URLs and email addresses
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)

    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace("‼", "")  # Remove exclamation marks
    text = text.replace("/m", "")  # Remove "/m"
    text = text.replace("\h", "")  # Remove "\h"
    text = text.replace("\z", "")  # Remove "\z"
    text = text.replace("☺", "")  # Remove smiley faces
    text = text.replace("¶", "")  # Remove paragraph marks
    text = text.replace("§", "")  # Remove section marks

    # Remove stop words (consider using a stop word list)
    # stop_words = set(stopwords.words('english'))
    # text = ' '.join([word for word in text.split() if word not in stop_words])

    # Remove short words
    text = re.sub(r'\b\w{1,2}\b', '', text)

    # Remove leading and trailing whitespace
    text = text.strip()

    return text

# Define a function to count words in a text
def calculate_word_count(df, columns):
    """
    Calculates the sum of word counts for specified columns and adds a new column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to calculate word counts.

    Returns:
        pd.DataFrame: The DataFrame with the new "word_count" column.
    """

    # Split text into words and count the length of each split
    for col in columns:
        df[col + '_word_count'] = df[col].astype(str).str.split().str.len()

    # Sum the word counts for the specified columns
    df['word_count'] = df[[col + '_word_count' for col in columns]].sum(axis=1)

    # Drop the temporary word count columns
    df.drop([col + '_word_count' for col in columns], axis=1, inplace=True)

    return df    