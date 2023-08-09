

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