from platform import win32_edition
import numpy as np
import cupy as cp
import scipy
import os
from os.path import exists, isfile, join
from pathlib import Path
import sys
import shutil
import gc
import math
import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import json
from math import floor
np.set_printoptions(precision=9)

# Import stopwords
import nltk
from nltk import word_tokenize
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.util import everygrams

# Import TensorLy
import tensorly as tl
import cudf
from cudf import Series
from cuml.feature_extraction.text import CountVectorizer # replace cuml with sklearn equivalents 
#from cuml.preprocessing.text.stem import PorterStemmer
from nltk.stem import PorterStemmer
import cupyx 

#Insert Plotly
import pandas as pd
import modin.pandas as md
import time
import pickle






# Import utility functions from other files
import file_operations as fop



# Constants

ROOT_DIR        = "/home/debanks/Dropbox/CongSpeechData/hein-bound/" # raw data
INDIR           = "processed_speech/"
RAW_DATA_PREFIX = "processed_speech/" # save the processed data


# Output Relative paths -- do not change
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

# Device settings
backend="cupy"
tl.set_backend(backend)
device = 'cuda'
porter_stemmer = PorterStemmer()


def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)




def partial_fit(self , data):
    if(hasattr(self , 'vocabulary_')):
        vocab = self.vocabulary_ # series
    else:
        vocab = Series()
    self.fit(data)
    vocab = vocab.append(self.vocabulary_)
    self.vocabulary_ = vocab.unique()

def tune_filesplit_size_on_IPCA_batch_size(IPCA_batchsize):
    return None



RAW_DATA_PREFIX = "processed_speech/"

# Device settings
backend="cupy"
tl.set_backend(backend)
device = 'cuda'
porter_stemmer = PorterStemmer()

def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


def basic_clean(df):
    df['speech'] = df['speech'].astype('str')
    df = df.drop_duplicates(keep="first")
    df['speech'] = df['speech'].str.lower()
    df['speech'] = df['speech'].str.replace(r'[^\w\s\d]+', '')
    return df


inDir = os.path.join(ROOT_DIR, INDIR)

dl = sorted(fop.get_files_in_dir(inDir)) # we sort so that they are ordered in chronological order
print("Done. Split files located at: {}.\n".format(inDir))


for i, f in enumerate(dl):
    path_in      = os.path.join(inDir,f)
    print(path_in)
    df = md.read_csv(path_in, sep="|",encoding='cp1252',on_bad_lines='skip' ,engine="python" ,names = ['speech_id','speech'],usecols=['speech_id','speech'])
    df = df[df['speech'].notnull()] # remove empty speeches
    df['speech'] = df['speech'].apply(stem_sentences) # stem the speeches
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    df   = basic_clean(df) # remove duplicates and punctuation
    mask = df['speech'].str.len() > 15 #more than 15 characters
    df   = df.loc[mask]
    print(df.head())
    df.to_csv(os.path.join(ROOT_DIR, RAW_DATA_PREFIX + f), index=False) 
    
for i, f in enumerate(dl):
    source_path = os.path.join(ROOT_DIR, RAW_DATA_PREFIX + f)
    for j,chunk in enumerate(pd.read_csv(source_path, chunksize=200000)):
        nrows = chunk.shape[0]
        if nrows==200000:
            chunk.to_csv(os.path.join(ROOT_DIR, RAW_DATA_PREFIX + Path(f).stem+'_'+str(j)+'.txt'), index=False) 
        if j>0:
            chunk.to_csv(os.path.join(ROOT_DIR, RAW_DATA_PREFIX + Path(f).stem+'_'+str(j)+'.txt'), index=False) 



# declare the stop words 

stop_words  = (stopwords.words('english'))
added_words = ["thread","im","say","will","has","upon","law","shall","secretari","senatorselect","representativeselect","speaker","by","for","hi","hey","hah","thank","watch","doe",
               "said","talk","congrats","congratulations","are","as","i", "time","abus","year","mani","h","r","seven","georg",
               "me", "my", "myself", "we", "our", "ours", "ourselves", "use","look","movement","new","york","jersey",
               "you", "your", "yours","he","her","him","she","hers","that","harass","whi","feel","say","gt","f","e","s","t","c","n","u","v","l","p","d","b","g","k","m","x","y","z",
               "be","with","their","they're","is","was","been","not","they","way","thi","rt","i","we","and","kentucki","michigan",
               "to","for","do","go","sir","1","2","motion","recognize","mr","gentleman","gentlewoman","gentlemen","recogniz","p","h",
               "amendment","would","on","have","ha","from","at","but","or","an","if","thi","all","about","so","nebraska","utah","senat",
               "it","have",  "one","think",   "thing","bring","put","well","take","exactli","tell","7","question","previous",
               "good","day","work", "latest","today","becaus","peopl","via","see","old","ani","covid-19","-","president","presid",
               "call", "wouldnt","wow", "learned","hi","-","", "things" ,"thing","can't","can","right","got","show",
               "cant","will","go","going","let","would","could","him","his","think","thi","ha","onli","back",
               "lets","let's","say","says","know","talk","talked","talks","dont","think","watch","right","yea","yes","state","jame",
               "said","something","this","was","has","had","abc","rt","ha","haha","hat","even","happen","resolut","rule","nae","nay",
               "something","wont","people","make","want","went","goes","people","had","also","ye","still","must",
               "person","like","come","from","yet","able","wa","yah","yeh","yeah","onli","ask","give","read",
               "need",  "get", "amp","amp&","yr","yrs","&amp;","amp","advanced","advancing","advanc","vote","aye","chair","made",
               "shirt", "vs","iâ€™m","|","amendment","family","get","adam","hear","feder","de","la","los","20","21","22","23","24","25","26","27",
               "28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","43","44","45","60","70","80","1970","1980","1990","2000","1975",    
               'el', 'para', 'en', 'que',"lo","0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","1800","1801","1900","1901",
               "2000","2001","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","adopt","adopted","adopting","adoption",
               "amend","back","","","service","work","around","alway","november","august","january","years","year","month","day","week","weekend",
               "happen","ive","hall","nation","work","service","this","discuss","community","learn","congressional","amendment","speaker","say",
               "said","talk","congrats","pelosi","gop","congratulations","are","as","i", "me", "my", "myself", "we", "our", "ours", "ourselves", 
               "you", "your", "yours","he","her","him","she","hers","that","be","with","their","they're","is","was","been","not","they","it","have",
               "will","has","by","for","madam","Speaker","mister","gentleman","gentlewoman","lady","voinovich","kayla","111th","115th","114th","rodgers",      
               "clerk" ,    "honor" ,   "address"   , "100","1000","10000","100000","16","17","18","20","may","june","july","august","september","october",
               "house" , "start"   ,"amend","bill",   "114th"    ,   "congress"  ,  "april","january","february","march","november","december",   
               "one",   "thing"    ,"bring","put", "north","give","keep","pa","even","texa","year","join","well","address","take","exactli","tell","good",
               "call",  "learned"    ,   "legislator","things" ,"things","can't","can","cant","will","go","going","let","ad","actual","actually","would",
               "lets","let's","say","says","know","talk","talked","talks","lady","honorable","dont","think","said","something",
               "something","wont","people","make","want","went","goes","congressmen","people","person","like","come","from",
               "need","us"]

# set stop words and countvectorizer method
stop_words= list(np.append(stop_words,added_words))
CountVectorizer.partial_fit = partial_fit



def custom_preprocessor(doc):
    return doc

countvec = CountVectorizer( stop_words = stop_words, #stop_words, # works
                            lowercase = True,#True, # works
                            ngram_range = (1, 2), #(1,2), ## allow for bigrams
                            preprocessor = custom_preprocessor,
                            max_df = 500000, #100000, # limit this to 10,000 ## 500000 for 8M
                            min_df = 20)# 2000) ## limit this to 20 ## 2500 for 8M