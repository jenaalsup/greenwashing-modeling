import os
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import file_operations as fop
import re

# constants
ROOT_DIR = "./reports"
PROCESSED_DIR = "processed_reports/" 
ENERGY_TICKERS = ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PXD", "PSX", "VLO", "OXY", "WMB", "HES", "LNG", "KMI", "DVN"]
CLEAN_ENERGY_TICKERS = ["FSLR", "ENPH", "SEDG", "ED", "PLUG", "ORA", "SHLS", "RUN", "ARRY", "AGR", "NOVA", "CWEN", "GPRE", "SPWR", "FCEL"]

# output relative paths
#CORPUS_FILEPATH_PREFIX = "corpus/"
#GENSIM_CORPUS_FILEPATH = "corpus.obj"
#COUNTVECTOR_FILEPATH = "countvec.obj"

def remove_extra_chars(text): # remove digits, punctuation, special characters, keep spaces
    pattern = r'[^a-zA-Z\s]'
    cleaned_string = re.sub(pattern, '', text)
    consolidated_spaces_string = re.sub(r'\s+', ' ', cleaned_string)
    return consolidated_spaces_string

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

 #/opt/homebrew/Cellar/gcc/13.1.0


df = pd.DataFrame()
column_names = ['company-type', 'company-ticker', 'year', 'part', 'text']
dl = sorted(fop.get_files_in_dir(ROOT_DIR)) # sort files chronologically
for i, f in enumerate(dl):
    # open file:
    path_in = os.path.join(ROOT_DIR,f)
    try:
        with open(path_in, encoding='cp1252') as file:
            info = [file.read()]
    except: # UnicodeDecodeError
        print(f"Failed to read {path_in} with cp1252 encoding.")
    # TODO: fix encodings

    # get metadata:
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
    # TODO: Add metadata for available subheadings
    
    # create dataframe:
    df_temp = pd.DataFrame(data, columns=column_names)

    # pre-process speech:
    df_temp = df_temp[df_temp['text'].notnull()] # remove empty speech
    df_temp['text'] = df_temp['text'].apply(stem_sentence) # stem the text
    df_temp = basic_clean(df_temp) # remove duplicates and punctuation
    mask = df_temp['text'].str.len() > 15 # more than 15 characters
    df_temp = df_temp.loc[mask]

    # concatenate this dataframe to the global result:
    df = pd.concat([df, df_temp], axis=0)

df.to_csv(os.path.join(ROOT_DIR, PROCESSED_DIR + 'processed-data.csv'), index=False) 


# TODO: anything after this should go in the modeling R script


# assmeble stop words:
stop_words  = (stopwords.words('english'))
added_words = ["will","has","by","for","hi","hey","are","as","i","we","our","ours","ourselves","use",
               "you","your","yours","that","f","e","s","t","c","n","u","v","l","p","d","b","g","k","m","x","y","z",
               "be","with","is","was","been","not","they","way","and","to","do","go","on","have","from",
               "at","but","or","an","if","all","so","it","thing","put","well","take","see","","can't","can",
               "got","cant","could","him","his","this","had","he","her","she","hers","their","they're","things",
               "go","going","let","would","make","like","come","us"]
stop_words= list(np.append(stop_words,added_words))





# tokenize data:
vectorizer = CountVectorizer(stop_words = stop_words,
                            lowercase = True,
                            ngram_range = (1, 2), # allow for bigrams
                            max_df = 0.9, # TODO: should this be a decimal or integer? remove words with > 1000 occurrences, > 90 % of docs
                            min_df = 0.1)# TODO: remove words with < 10 occurrencees, < 10 % of docs

# apply vectorizer:
X = vectorizer.fit_transform(df['text'])
X_transformed = X.toarray()
word_count = X_transformed.sum(axis=0)
vocab = vectorizer.get_feature_names_out()
word_count_dict = dict(zip(vocab, word_count))
feature_names = vectorizer.get_feature_names_out() # get vocabulary
vectorized_df = pd.DataFrame(X_transformed, columns=feature_names) # create dataframe

# TODO: used to determine max_df and min_df
# sort dictionary descending
sorted_word_count = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)
top_n = 30
for word, count in sorted_word_count[:top_n]:
    print(f"{word}: {count}")

# sort dictionary ascending
sorted_word_count = sorted(word_count_dict.items(), key=lambda x: x[1])
top_n = 10 
for word, count in sorted_word_count[:top_n]:
    print(f"{word}: {count}")

# TODO: get metrics from sec filings