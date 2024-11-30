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
import docx
from datetime import datetime

