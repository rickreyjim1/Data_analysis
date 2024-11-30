#Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import nltk
from nltk.corpus import stopwords
#nltk.download('punkt_tab')   
from string import punctuation
from nltk.tokenize import word_tokenize
import re
import string
from nltk.stem import WordNetLemmatizer
#from stop_words import get_stop_words
#import truecase
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import time
import os
from collections import Counter
import calendar
import networkx as nx
import community
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from tqdm import trange
import community.community_louvain

#importing dataset functions
from features.dataset_functions import create_folder
from features.dataset_functions import select_excel_file
from features.dataset_functions import reading_dataset
from features.dataset_functions import shape_dataframe
from features.dataset_functions import format_column_names
from features.dataset_functions import missing_values_table
from features.dataset_functions import handle_missing_values
#from features.dataset_functions import select_columns_for_analysis
from features.dataset_functions import get_top_values
from features.dataset_functions import extract_kmdb_string

#importing plot functions
from features.plots import plot_top_values
from features.plots import plot_word_count_distribution
from features.plots import plot_and_save_time_related_data

#importing text functions
from features.text_functions import stem_sentence
from features.text_functions import get_common_words
from features.text_functions import remove_freqwords
from features.text_functions import remove_rarewords
from features.text_functions import remove_calender_text
from features.text_functions import remove_greetings
from features.text_functions import process_text
from features.text_functions import calculate_word_count

#importing time functions
from features.time_functions import extract_date_time_features

#---------------------------------------- Start of the analysis ----------------------------------------#
start_time = time.time()
print(f"Start time -> \n{start_time}\n")

#---------------------------------------- Creation of folders for outputs ----------------------------------------#

# Define the initial folder path
folder_path_input = os.getcwd()

# Input the name of the new folder
new_folder = input("Please input the name of the folder to be created (where the outputs will be sent) -> ")

# Select the Excel file from the input folder path
selected_file = select_excel_file(folder_path_input)

# Combine the input folder path with the new folder name
outputs_folder_preliminary = os.path.join(folder_path_input, new_folder)

# Create the preliminary folder
create_folder(outputs_folder_preliminary)

if selected_file:
    print(f"Selected Excel file -> {selected_file}\n")
    
    # Extract the root of the selected file to create the final folder name
    file_root, file_extension = os.path.splitext(selected_file)  # Get the filename without extension
    
    # Combine the preliminary folder with the file root to create the final output folder
    outputs_folder = os.path.join(outputs_folder_preliminary, file_root)
    
    # Create the final output folder
    create_folder(outputs_folder)
        
else:
    print("No valid file selected.")

df_data = reading_dataset(selected_file)
df_data = format_column_names(df_data)
missing_values = missing_values_table(df_data)
#print(missing_values.head(60))
#Drop the columns having more than 20% missing values
drop_indexes = missing_values[missing_values['% of Total Values'] > 80].index.tolist()
print(f"Columns to be dropped -> \n{drop_indexes}\n")