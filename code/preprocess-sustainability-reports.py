from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import os
from pathlib import Path
import numpy as np
import pandas as pd
import modin.pandas as md
import file_operations as fop

# constants
ROOT_DIR        = "./reports" # raw data
INDIR           = "processed_reports/"
RAW_DATA_PREFIX = "processed_reports/" # save the processed data

# output relative paths
CORPUS_FILEPATH_PREFIX = "corpus/"
GENSIM_CORPUS_FILEPATH = "corpus.obj"
COUNTVECTOR_FILEPATH = "countvec.obj"

def basic_clean(df):
    df['speech'] = df['speech'].astype('str')
    df = df.drop_duplicates(keep="first")
    df['speech'] = df['speech'].str.lower() # lowercase
    df['speech'] = df['speech'].str.replace(r'[^\w\s\d]+', '') # remove digits and punctuation
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

for i, f in enumerate(dl):
    path_in = os.path.join(ROOT_DIR,f) # was inDir before
    print(path_in)
    df = pd.read_csv(path_in, encoding='cp1252', sep=" ", header=None, names = ['speech_id','speech'], usecols=['speech_id','speech']) # add columns
    df = df[df['speech'].notnull()] # remove empty speeches
    df['speech'] = df['speech'].apply(stem_sentence) # stem the speeches
    #mempool = cp.get_default_memory_pool()
    #mempool.free_all_blocks()
    df = basic_clean(df) # remove duplicates and punctuation - TODO: giving a warning
    mask = df['speech'].str.len() > 15 # more than 15 characters
    df   = df.loc[mask]
    print(df.head())
    df.to_csv(os.path.join(ROOT_DIR, RAW_DATA_PREFIX + f), index=False) 



