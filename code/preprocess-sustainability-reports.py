import os
import pandas as pd
from nltk.stem import PorterStemmer
import file_operations as fop
import re

# constants
ROOT_DIR = "./reports"
PROCESSED_DIR = "processed_reports/" 
ENERGY_TICKERS = ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PXD", "PSX", "VLO", "OXY", "WMB", "HES", "LNG", "KMI", "DVN"]
CLEAN_ENERGY_TICKERS = ["FSLR", "ENPH", "SEDG", "ED", "PLUG", "ORA", "SHLS", "RUN", "ARRY", "AGR", "NOVA", "CWEN", "GPRE", "SPWR", "FCEL"]

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

df = pd.DataFrame()
column_names = ['doc_id', 'company_type', 'company_ticker', 'year', 'part', 'text']
dl = sorted(fop.get_files_in_dir(ROOT_DIR)) # sort files chronologically
for i, f in enumerate(dl):
    # open file:
    path_in = os.path.join(ROOT_DIR,f)
    try:
        with open(path_in, encoding='cp1252') as file:
            info = [file.read()]
    except: # potential UnicodeDecodeError
        print(f"Failed to read {path_in} with cp1252 encoding.")

    # get metadata:
    filename_parts = f.split("-")
    company_type = 'N/A'
    if (filename_parts[0].upper() in ENERGY_TICKERS):
        company_type = 'energy'
    if (filename_parts[0].upper() in CLEAN_ENERGY_TICKERS):
        company_type = 'clean-energy'
    data = {'doc_id': f,
            'company_type': company_type,
            'company_ticker': filename_parts[0],
            'year': filename_parts[1],
            'part': filename_parts[2][:-4],
            'text': info,}
    
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