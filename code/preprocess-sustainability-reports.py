import os
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import file_operations as fop
import re
import statistics # for vectorizer - delete later

# constants
ROOT_DIR        = "./reports" # raw data
PROCESSED_DIR = "processed_reports/" 
ENERGY_TICKERS = ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PXD", "PSX", "VLO", "OXY", "WMB", "HES", "LNG", "KMI", "DVN"]
CLEAN_ENERGY_TICKERS = ["FSLR", "ENPH", "SEDG", "ED", "PLUG", "ORA", "SHLS", "RUN", "ARRY", "AGR", "NOVA", "CWEN", "GPRE", "SPWR", "FCEL"]

# output relative paths
#CORPUS_FILEPATH_PREFIX = "corpus/"
#GENSIM_CORPUS_FILEPATH = "corpus.obj"
#COUNTVECTOR_FILEPATH = "countvec.obj"

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

inDir = os.path.join(ROOT_DIR, PROCESSED_DIR)
dl = sorted(fop.get_files_in_dir(ROOT_DIR)) # sort files chronologically

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

df.to_csv(os.path.join(ROOT_DIR, PROCESSED_DIR + 'processed-data.csv'), index=False) 

# assmeble stop words
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
                            max_df = 1000, # remove words with > 10,000 occurrences
                            min_df = 10)# remove words with < 20 occurrencees


# determine max_df and min_df values:

# fit and transform text data
X = vectorizer.fit_transform(df['text'])
# get the count of each word
word_count = X.toarray().sum(axis=0)
vocab = vectorizer.get_feature_names_out()
# store word counts in dictionary
word_count_dict = dict(zip(vocab, word_count))

# TODO: check length of word_count_dict and find statistics on word frequency

# sort dictionary descending
sorted_word_count = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)

top_n = 30
for word, count in sorted_word_count[:top_n]:
    print(f"{word}: {count}")
print("total unique words: ", len(sorted_word_count))

# Extract frequencies from the list
frequencies = [frequency for word, frequency in sorted_word_count]
print("total words: ", sum(frequencies))


# Convert frequencies to numpy array
frequencies_array = np.array(frequencies, dtype=np.int64)

# Calculate statistics
mode = statistics.mode(frequencies)
median = np.median(frequencies_array)
mean = np.mean(frequencies_array)
std_dev = np.std(frequencies_array)

# Print statistics
print("Mode:", mode)
print("Median:", median)
print("Mean:", mean)
print("Standard Deviation:", std_dev)

# sort dictionary ascending
#sorted_word_count = sorted(word_count_dict.items(), key=lambda x: x[1])
#top_n = 10 
#for word, count in sorted_word_count[:top_n]:
    #print(f"{word}: {count}")


X = vectorizer.fit_transform(df['text'])
# Get feature names (vocabulary) and transformed data
feature_names = vectorizer.get_feature_names_out()
X_transformed = X.toarray()
# Create a DataFrame for the vectorized data
vectorized_df = pd.DataFrame(X_transformed, columns=feature_names)
#vectorized_df.to_csv(os.path.join(ROOT_DIR, PROCESSED_DIR + 'processed-data-with-vectorizer.csv'), index=False) 



'''
# OUTLIER CALCULATION
import numpy as np

# Extract frequencies from the list
frequencies = [frequency for _, frequency in sorted_word_count]

# Calculate the mean and standard deviation
mean = np.mean(frequencies)
std_dev = np.std(frequencies)

# Set the z-score threshold (e.g., 2 standard deviations)
z_score_threshold = 2

# Calculate the z-scores for each frequency
z_scores = [(frequency - mean) / std_dev for frequency in frequencies]

# Identify outliers based on z-scores
outliers = [word for (word, frequency), z_score in zip(sorted_word_count, z_scores) if abs(z_score) > z_score_threshold]


# Print the frequencies of outliers
for word, frequency in sorted_word_count:
    if word in outliers:
        print(f"Outlier: {word}, Frequency: {frequency}")
'''

