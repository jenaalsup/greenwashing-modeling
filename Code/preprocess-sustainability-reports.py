# TODO: reorganize imports
import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import gensim
import file_operations as fop
import os

# constants
ROOT_DIR        = "/reports" # raw data
INDIR           = "processed_reports/"
RAW_DATA_PREFIX = "processed_reports/" # save the processed data

# output relative paths
CORPUS_FILEPATH_PREFIX = "corpus/"
GENSIM_CORPUS_FILEPATH = "corpus.obj"
COUNTVECTOR_FILEPATH = "countvec.obj"

def basic_clean(df):
    df['speech'] = df['speech'].astype('str')
    df = df.drop_duplicates(keep="first")
    df['speech'] = df['speech'].str.lower()
    df['speech'] = df['speech'].str.replace(r'[^\w\s\d]+', '')
    return df

porter_stemmer = PorterStemmer()
def stem_sentence(sentence):
    tokens = sentence.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

stop_words  = (stopwords.words('english'))
#added_words = ["sustainability", "environment"] # TODO: tweak if necessary, go through danny's and keep important stuff
#stop_words= list(np.append(stop_words,added_words))

countvec = CountVectorizer( stop_words = stop_words,
                            lowercase = True,
                            ngram_range = (1, 2), # allow for bigrams
                            max_df = 10000, # remove words with > 10,000 occurrences
                            min_df = 20)# remove words with < 20 occurrencees

inDir = os.path.join(ROOT_DIR, INDIR)
# TODO: not working - get_files_in_dir() method not found
dl = sorted(fop.get_files_in_dir(inDir)) # put files in chronological order
print("Done. Split files located at: {}.\n".format(inDir))