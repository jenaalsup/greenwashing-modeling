from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import os
from pathlib import Path
import numpy as np
import pandas as pd
import modin.pandas as md
import file-operations as fop

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
    df['speech'] = df['speech'].str.lower()
    df['speech'] = df['speech'].str.replace(r'[^\w\s\d]+', '') # TODO: what does this do?
    return df

porter_stemmer = PorterStemmer()
def stem_sentence(sentence):
    tokens = sentence.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

inDir = os.path.join(ROOT_DIR, INDIR)
dl = sorted(fop.get_files_in_dir(ROOT_DIR)) # put files in chronological order
print(dl)

for i, f in enumerate(dl):
    path_in = os.path.join(inDir,f)
    print(path_in)
    # TODO: can read_csv work with txt files? - the following line causes an error
    #df = md.read_csv(path_in, sep="|",encoding='cp1252',on_bad_lines='skip' ,engine="python" ,names = ['speech_id','speech'],usecols=['speech_id','speech'])

# TODO: what does this for loop do?
for i, f in enumerate(dl):
    source_path = os.path.join(ROOT_DIR, RAW_DATA_PREFIX + f)
    for j,chunk in enumerate(pd.read_csv(source_path, chunksize=200000)):
        nrows = chunk.shape[0]
        if nrows==200000:
            chunk.to_csv(os.path.join(ROOT_DIR, RAW_DATA_PREFIX + Path(f).stem+'_'+str(j)+'.txt'), index=False) 
        if j>0:
            chunk.to_csv(os.path.join(ROOT_DIR, RAW_DATA_PREFIX + Path(f).stem+'_'+str(j)+'.txt'), index=False) 


# TODO: where is the rest of the code being used?

stop_words  = (stopwords.words('english'))
added_words = ["will","has","by","for","hi","hey","are","as","i","we","our","ours","ourselves","use",
               "you","your","yours","that","f","e","s","t","c","n","u","v","l","p","d","b","g","k","m","x","y","z",
               "be","with","is","was","been","not","they","way","and","to","do","go","on","have","from",
               "at","but","or","an","if","all","so","it","thing","put","well","take","see","","can't","can",
               "got","cant","could","him","his","this","had","he","her","she","hers","their","they're","things",
               "go","going","let","would","make","like","come","us"]
stop_words= list(np.append(stop_words,added_words))

countvec = CountVectorizer( stop_words = stop_words,
                            lowercase = True,
                            ngram_range = (1, 2), # allow for bigrams
                            max_df = 10000, # remove words with > 10,000 occurrences
                            min_df = 20)# remove words with < 20 occurrencees



