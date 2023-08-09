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

# name of company, year, type of company - add columns to dataframe, rows at document level
df = pd.DataFrame()
column_names = ['text', 'company-ticker', 'year', 'part'] # add 'company-type', later
for i, f in enumerate(dl):
    # open file
    path_in = os.path.join(ROOT_DIR,f)
    with open(path_in, encoding='cp1252') as file:
        info = [file.read()]
    
    # create dataframe
    filename_parts = f.split("-") # for metadata
    data = {'text': info,
            'company-ticker': filename_parts[0],
            'year': filename_parts[1],
            'part': filename_parts[2]}
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



# determine max_df and min_df values:

# fit and transform text data
X = vectorizer.fit_transform(df['text'])
# get the count of each word
word_count = X.toarray().sum(axis=0)
vocab = vectorizer.get_feature_names_out()
# store word counts in dictionary
word_count_dict = dict(zip(vocab, word_count))

# sort dictionary descending
sorted_word_count = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)
top_n = 10
for word, count in sorted_word_count[:top_n]:
    print(f"{word}: {count}")

# sort dictionary ascending
sorted_word_count = sorted(word_count_dict.items(), key=lambda x: x[1])
top_n = 10 
for word, count in sorted_word_count[:top_n]:
    print(f"{word}: {count}")


X = vectorizer.fit_transform(df['text'])
# Get feature names (vocabulary) and transformed data
feature_names = vectorizer.get_feature_names_out()
X_transformed = X.toarray()
# Create a DataFrame for the vectorized data
vectorized_df = pd.DataFrame(X_transformed, columns=feature_names)
vectorized_df.to_csv(os.path.join(ROOT_DIR, RAW_DATA_PREFIX + 'processed-data-with-vectorizer.csv'), index=False) 