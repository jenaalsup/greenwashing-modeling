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
ROOT_DIR        = "/Reports" # raw data
INDIR           = "processed_reports/"
RAW_DATA_PREFIX = "processed_reports/" # save the processed data

# output relative paths, TODO: what does this mean?
X_MAT_FILEPATH_PREFIX = "x_mat/"
X_FILEPATH = "X_full.obj"
X_DF_FILEPATH = "X_df.obj"
X_LST_FILEPATH = "X_lst.obj"
CORPUS_FILEPATH_PREFIX = "corpus/"
GENSIM_CORPUS_FILEPATH = "corpus.obj"
COUNTVECTOR_FILEPATH = "countvec.obj"
TOP_SENTS_FILEPATH = "top_sents.obj"
JST_FILEPATH = "JST.obj"
VOCAB_FILEPATH = "vocab.csv"
EXISTING_VOCAB_FILEPATH = "vocab.obj"
TOPIC_FILEPATH_PREFIX   = 'predicted_topics/'
DOCUMENT_TOPIC_FILEPATH = 'dtm.csv'
COHERENCE_FILEPATH = 'coherence.obj'
DOCUMENT_TOPIC_FILEPATH_TOT = 'dtm_df.csv'
OUT_ID_DATA_PREFIX = 'ids/' 
TOP_WORDS_FILEPATH ='top_words.csv'

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
added_words = ["sustainability", "environment"] # TODO: add other custom stopwords
stop_words= list(np.append(stop_words,added_words))

countvec = CountVectorizer( stop_words = stop_words,
                            lowercase = True,
                            ngram_range = (1, 2), # allow for bigrams
                            #preprocessor = custom_preprocessor, TODO: what does this do?
                            max_df = 10000, # remove words with > 10,000 occurrences
                            min_df = 20)# remove words with < 20 occurrencees

inDir = os.path.join(ROOT_DIR, INDIR)
# TODO: not working - get_files_in_dir() method not found
dl = sorted(fop.get_files_in_dir(inDir)) # put files in chronological order
print("Done. Split files located at: {}.\n".format(inDir))