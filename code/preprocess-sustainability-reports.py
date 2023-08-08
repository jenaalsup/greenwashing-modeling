from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import os
from pathlib import Path
import numpy as np
import pandas as pd
import modin.pandas as md
import file_operations as fop
import re # for removing digits using regex

# constants
ROOT_DIR        = "./reports" # raw data
INDIR           = "processed_reports/"
RAW_DATA_PREFIX = "processed_reports/" # save the processed data

# output relative paths
CORPUS_FILEPATH_PREFIX = "corpus/"
GENSIM_CORPUS_FILEPATH = "corpus.obj"
COUNTVECTOR_FILEPATH = "countvec.obj"

def remove_extra_chars(text): # digits, punctuation, special characters, keep spaces
    pattern = r'[^a-zA-Z\s]'
    return re.sub(pattern, '', text)

def remove_digits(text):
    return re.sub(r'\d', '', text)

# Apply the function to the text_column


def basic_clean(df):
    df['text'] = df['text'].astype('str')
    df = df.drop_duplicates(keep="first")
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].apply(remove_extra_chars)
    return df

porter_stemmer = PorterStemmer()
def stem_sentence(sentence):
    tokens = sentence.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

inDir = os.path.join(ROOT_DIR, INDIR)
dl = sorted(fop.get_files_in_dir(ROOT_DIR)) # put files in chronological order
print(dl)

stop_words  = (stopwords.words('english'))
added_words = ["will","has","by","for","hi","hey","are","as","i","we","our","ours","ourselves","use",
               "you","your","yours","that","f","e","s","t","c","n","u","v","l","p","d","b","g","k","m","x","y","z",
               "be","with","is","was","been","not","they","way","and","to","do","go","on","have","from",
               "at","but","or","an","if","all","so","it","thing","put","well","take","see","","can't","can",
               "got","cant","could","him","his","this","had","he","her","she","hers","their","they're","things",
               "go","going","let","would","make","like","come","us"]
stop_words= list(np.append(stop_words,added_words))

# tokenizes data
countvec = CountVectorizer( stop_words = stop_words,
                            lowercase = True,
                            ngram_range = (1, 2), # allow for bigrams
                            max_df = 10000, # remove words with > 10,000 occurrences
                            min_df = 20)# remove words with < 20 occurrencees

# countvec.vocabulary_
# name of company, year, type of company - add columns to dataframe, rows at document level
df = pd.DataFrame()
for i, f in enumerate(dl):
    path_in = os.path.join(ROOT_DIR,f)
    with open(path_in, encoding='cp1252') as file:
        info = [file.read()]

    df_temp = pd.DataFrame([info], columns=['text'])
    df_temp = df_temp[df_temp['text'].notnull()] # remove empty speech
    df_temp['text'] = df_temp['text'].apply(stem_sentence) # stem the textes
    df_temp = basic_clean(df_temp) # remove duplicates and punctuation - TODO: giving a warning
    mask = df_temp['text'].str.len() > 15 # more than 15 characters
    df_temp = df_temp.loc[mask]
    df = pd.concat([df, df_temp], axis=0)

print(df.head())
df.to_csv(os.path.join(ROOT_DIR, RAW_DATA_PREFIX + 'processed-data.csv'), index=False) 