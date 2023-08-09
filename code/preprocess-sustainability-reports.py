from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import os
from pathlib import Path
import numpy as np
import pandas as pd
import modin.pandas as md
import file_operations as fop
import re

# constants
ROOT_DIR        = "./reports" # raw data
INDIR           = "processed_reports/"
RAW_DATA_PREFIX = "processed_reports/" # save the processed data
ENERGY_TICKERS = ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PXD", "PSX", "VLO", "OXY", "WMB", "HES", "LNG", "KMI", "DVN"]
CLEAN_ENERGY_TICKERS = ["FSLR", "ENPH", "SEDG", "ED", "PLUG", "ORA", "SHLS", "RUN", "ARRY", "AGR", "NOVA", "CWEN", "GPRE", "SPWR", "FCEL"]

# output relative paths
CORPUS_FILEPATH_PREFIX = "corpus/"
GENSIM_CORPUS_FILEPATH = "corpus.obj"
COUNTVECTOR_FILEPATH = "countvec.obj"

def remove_extra_chars(text): # digits, punctuation, special characters, keep spaces
    pattern = r'[^a-zA-Z\s]'
    return re.sub(pattern, '', text)

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
dl = sorted(fop.get_files_in_dir(ROOT_DIR)) # put files in chronological order by name

stop_words  = (stopwords.words('english'))
added_words = ["will","has","by","for","hi","hey","are","as","i","we","our","ours","ourselves","use",
               "you","your","yours","that","f","e","s","t","c","n","u","v","l","p","d","b","g","k","m","x","y","z",
               "be","with","is","was","been","not","they","way","and","to","do","go","on","have","from",
               "at","but","or","an","if","all","so","it","thing","put","well","take","see","","can't","can",
               "got","cant","could","him","his","this","had","he","her","she","hers","their","they're","things",
               "go","going","let","would","make","like","come","us"]
stop_words= list(np.append(stop_words,added_words))

# tokenizes data
vectorizer = CountVectorizer(stop_words = stop_words,
                            lowercase = True,
                            ngram_range = (1, 2), # allow for bigrams
                            max_df = 100000, # remove words with > 10,000 occurrences
                            min_df = 1)# remove words with < 20 occurrencees

df = pd.DataFrame()
column_names = ['company-type', 'company-ticker', 'year', 'part', 'text'] # add 'company-type', later
for i, f in enumerate(dl):
    # open file
    path_in = os.path.join(ROOT_DIR,f)
    try:
        with open(path_in, encoding='cp1252') as file:
            info = [file.read()]
    except: # UnicodeDecodeError
        print(f"Failed to read {path_in} with cp1252 encoding.")

    # add metadata
    filename_parts = f.split("-")
    company_type = 'N/A'
    if (filename_parts[0].upper() in ENERGY_TICKERS):
        company_type = 'energy'
    if (filename_parts[0].upper() in CLEAN_ENERGY_TICKERS):
        company_type = 'clean-energy'
    data = {'company-type': company_type,
            'company-ticker': filename_parts[0],
            'year': filename_parts[1],
            'part': filename_parts[2][:-4],
            'text': info,}
    
    # create dataframe
    df_temp = pd.DataFrame(data, columns=column_names)

    # pre-process speech
    df_temp = df_temp[df_temp['text'].notnull()] # remove empty speech
    df_temp['text'] = df_temp['text'].apply(stem_sentence) # stem the text
    df_temp = basic_clean(df_temp) # remove duplicates and punctuation
    mask = df_temp['text'].str.len() > 15 # more than 15 characters
    df_temp = df_temp.loc[mask]

    # concatenate this dataframe to the global result
    df = pd.concat([df, df_temp], axis=0)

df.to_csv(os.path.join(ROOT_DIR, RAW_DATA_PREFIX + 'processed-data.csv'), index=False) 
