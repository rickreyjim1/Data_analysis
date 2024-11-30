import pandas as pd
from nltk import ngrams
from collections import Counter
import nltk

# Ensure you have the NLTK word tokenizer
#nltk.download('punkt')

# Load your Excel file
df = pd.read_excel('FCC_10222024.xlsx')#, sheet_name="Data")

# Tokenize the text in the 'Short Description' column
tokens = nltk.word_tokenize(' '.join(df['Short description'].astype(str)))

# Find bigrams, trigrams, and tetragrams
bigrams = ngrams(tokens, 2)
trigrams = ngrams(tokens, 3)
tetragrams = ngrams(tokens, 4)
pentagrams = ngrams(tokens, 5)
hexagrams = ngrams(tokens, 6)

# Count frequencies
bigram_freq = Counter(bigrams)
trigram_freq = Counter(trigrams)
tetragram_freq = Counter(tetragrams)
pentagram_freq = Counter(pentagrams)
hexagram_freq = Counter(hexagrams)

# Convert results to DataFrames
bigram_df = pd.DataFrame(bigram_freq.most_common(50), columns=['Bigram', 'Frequency'])
trigram_df = pd.DataFrame(trigram_freq.most_common(50), columns=['Trigram', 'Frequency'])
tetragram_df = pd.DataFrame(tetragram_freq.most_common(50), columns=['Tetragram', 'Frequency'])
pentagram_df = pd.DataFrame(pentagram_freq.most_common(50), columns=['Pentagram', 'Frequency'])
hexagram_df = pd.DataFrame(hexagram_freq.most_common(50), columns=['Hexagram', 'Frequency'])


# Write to Excel file
with pd.ExcelWriter('FCC_10222024-ngram_frequencies.xlsx') as writer:
    bigram_df.to_excel(writer, sheet_name='Bigrams', index=False)
    trigram_df.to_excel(writer, sheet_name='Trigrams', index=False)
    tetragram_df.to_excel(writer, sheet_name='Tetragrams', index=False)
    pentagram_df.to_excel(writer, sheet_name='Pentagrams', index=False)
    hexagram_df.to_excel(writer, sheet_name='Hexagrams', index=False)

print("N-grams saved to 'ngram_frequencies.xlsx'")
