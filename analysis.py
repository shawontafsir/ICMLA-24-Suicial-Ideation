import math
import re
import string

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import Phrases, CoherenceModel
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess
from nltk import WordNetLemmatizer
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim
from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score

import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# ---------------------- Judgemental Analysis -----------------------------
df_judgemental = pd.read_csv('documents/judgmental_sample.csv')
df_judgemental["Label"] = df_judgemental.apply(lambda row: 1 if row["Label"] == "suicidal" else 0, axis=1)

df_judgemental_khalid = pd.read_csv('documents/judgmental_sample_khalid.csv')
df = pd.merge(df_judgemental, df_judgemental_khalid, on="Text", how="inner")
print(cohen_kappa_score(df["Label_x"], df["Label_y"]))

df_judgemental_jamil = pd.read_csv('documents/judgmental_sample_jamil.csv')
df = df_judgemental.merge(df_judgemental_jamil, on="Text", how="inner")
# Set the original label for confused annotated labels
df["Label_y"] = df.apply(lambda row: row["Label_y"] if pd.isna(row["Remarks"]) else row["Label_x"], axis=1)
print(cohen_kappa_score(df["Label_x"], df["Label_y"]))


# Merge both train and test set for analysis
df_train = pd.read_csv('documents/suicidality_labeled_train.csv').dropna()
df_test = pd.read_csv('documents/suicidality_labeled_test.csv').dropna()
df = pd.concat([df_train, df_test])

# # ----------------- Stratified Sample of the Dataset for Judgemental Analysis ------------------
# # Define the sampling criteria, e.g., equal representation of class labels
# sample_size = 0.01  # 1% of the dataset
# stratified_sample = df.groupby('Label', group_keys=False).apply(lambda x: x.sample(min(len(x), int(sample_size * len(df)))))
# stratified_sample["Text"] = stratified_sample.apply(lambda row: f"{row['Title']} {row['Content']}", axis=1)
# stratified_sample = stratified_sample[["Text", "Label"]].reset_index(drop=True)
#
# # Save the stratified sample for expert review
# stratified_sample.to_csv('./documents/judgmental_sample.csv', index=True)


df_suicidal = df[df["Label"] == "suicidal"]
df_non_suicidal = df[df["Label"] == "non-suicidal"]

# ------------ Total Hashtag ----------------
total_hashtag = sum(str(row.loc['Title'] + " " + row['Content']).count('#') for _, row in df_suicidal.iterrows())
print(total_hashtag/len(df_suicidal))
total_hashtag = sum(str(row.loc['Title'] + " " + row['Content']).count('#') for _, row in df_non_suicidal.iterrows())
print(total_hashtag/len(df_non_suicidal))

# ----------- Posts with URL ----------------
posts_with_url = sum(1 if 'http' in str(row.loc['Title'] + " " + row['Content']).lower() else 0 for _, row in df_suicidal.iterrows())
print(posts_with_url/len(df_suicidal))
posts_with_url = sum(1 if 'http' in str(row.loc['Title'] + " " + row['Content']).lower() else 0 for _, row in df_non_suicidal.iterrows())
print(posts_with_url/len(df_non_suicidal))

# ----------- Posts Length ------------------
total_posts_length = sum(len(str(row['Content']).lower()) for _, row in df_suicidal.iterrows())
print(total_posts_length/len(df_suicidal))
total_posts_length = sum(len(str(row['Content']).lower()) for _, row in df_non_suicidal.iterrows())
print(total_posts_length/len(df_non_suicidal))


def preprocess(row):
    text = str(row['Title'] + " " + row['Content']).lower()

    # Remove special characters and digits
    text = re.sub(r'\W+', ' ', text)
    # Tokenize and remove stop words
    tokens = [word for word in simple_preprocess(text) if word not in stop_words]

    # Remove punctuation and special characters
    table = str.maketrans('', '', string.punctuation)
    tokens = [token.translate(table) for token in tokens]

    return [WordNetLemmatizer().lemmatize(token) for token in tokens]


# ----------- Parts-of-Speech -----------------------
# Define the POS tags for verbs, nouns, pronouns, and adjectives
verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
pronoun_tags = {'PRP', 'PRP$', 'WP', 'WP$'}
adjective_tags = {'JJ', 'JJR', 'JJS'}

# suicidal
pos_tag_count = {}
for _, row in df_suicidal.iterrows():
    tokens = nltk.word_tokenize(row["Content"])
    pos_tags = nltk.pos_tag(tokens)
    for _, pos in pos_tags:
        if pos in verb_tags:
            pos_tag_count["verb"] = pos_tag_count.get("verb", 0) + 1
        elif pos in noun_tags:
            pos_tag_count["noun"] = pos_tag_count.get("noun", 0) + 1
        elif pos in pronoun_tags:
            pos_tag_count["pronoun"] = pos_tag_count.get("pronoun", 0) + 1
        elif pos in adjective_tags:
            pos_tag_count["adjective"] = pos_tag_count.get("adjective", 0) + 1

print("Suicidal POS tag count: ", pos_tag_count)
print("Avg suicidal pos tag count: ", [(k, v/len(df_suicidal)) for k, v in pos_tag_count.items()])


# non-suicidal
pos_tag_count = {}
for _, row in df_non_suicidal.iterrows():
    tokens = nltk.word_tokenize(row["Content"])
    pos_tags = nltk.pos_tag(tokens)
    for _, pos in pos_tags:
        if pos in verb_tags:
            pos_tag_count["verb"] = pos_tag_count.get("verb", 0) + 1
        elif pos in noun_tags:
            pos_tag_count["noun"] = pos_tag_count.get("noun", 0) + 1
        elif pos in pronoun_tags:
            pos_tag_count["pronoun"] = pos_tag_count.get("pronoun", 0) + 1
        elif pos in adjective_tags:
            pos_tag_count["adjective"] = pos_tag_count.get("adjective", 0) + 1

print("Non-Suicidal POS tag count: ", pos_tag_count)
print("Avg non-suicidal pos tag count: ", [(k, v/len(df_non_suicidal)) for k, v in pos_tag_count.items()])


# ------------ Tokens Count --------------
# non-suicidal
docs = [preprocess(row) for _, row in df_non_suicidal.iterrows()]
total_tokens = sum(len(doc) for doc in docs)
print(total_tokens/len(df_non_suicidal))

# suicidal
docs = [preprocess(row) for _, row in df_suicidal.iterrows()]
total_tokens = sum(len(doc) for doc in docs)
print(total_tokens/len(df_suicidal))


# ------------- Phrasal Extraction ---------------
# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(docs, min_count=20, threshold=2)
trigram = Phrases(bigram[docs], threshold=2)


# Fit the models
bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


# Form bigrams
bigram_docs = make_bigrams(docs)

# Form trigrams (which also includes bigrams)
trigram_docs = make_trigrams(docs)


# ---------------- Topic Modeling -----------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([' '.join(tokens) for tokens in trigram_docs])

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Get TF-IDF scores for each word in each document
tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)

# Create a DataFrame with words and their TF-IDF scores
tfidf_df = pd.DataFrame({'term': feature_names, 'tfidf': tfidf_scores})

# Sort the DataFrame by TF-IDF scores in descending order
tfidf_df = tfidf_df.sort_values(by='tfidf', ascending=False)

# Display the top 20 important words by TF-IDF score
print(tfidf_df.head(20))

topic_number = 8
lda = LatentDirichletAllocation(n_components=topic_number, random_state=42)
lda.fit(tfidf_matrix)

# # Compute Topic Coherence Score
# # Create a dictionary representation of the documents.
# dictionary = Dictionary(trigram_docs)
#
# # Filter out words that occur less than 20 documents, or more than 50% of the documents.
# dictionary.filter_extremes(no_below=20, no_above=0.5)
#
# # Bag-of-words representation of the documents.
# corpus = [dictionary.doc2bow(doc) for doc in trigram_docs]
#
# # Extract the topics and terms from the scikit-learn LDA model
# topics_terms = lda.components_
#
# # Convert the LDA model output to the format expected by gensim
# gensim_topics = []
# for topic_idx, topic in enumerate(topics_terms):
#     topic_terms = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-10 - 1:-1]]
#     gensim_topics.append(topic_terms)
#
# coherence_model = CoherenceModel(
#     topics=gensim_topics,
#     texts=trigram_docs,
#     dictionary=dictionary,
#     coherence='c_v'
# )
# coherence_lda = coherence_model.get_coherence()
#
# print("Coherence Score: ", coherence_lda)


def print_topics(top_n=20):
    for idx, topic in enumerate(lda.components_):
        print(f"Topic {idx}:")
        print([vectorizer.get_feature_names_out()[ind] for ind in topic.argsort()[:-top_n - 1:-1]])


print_topics()


# ------------------- Display topic wise related words as word clouds -------------------
topic_wise_all_words = [
    ' '.join([vectorizer.get_feature_names_out()[ind] for ind in topic.argsort()[:-20 - 1:-1]])
    for topic in lda.components_
]
topic_wise_wordcloud = [
    WordCloud(width=800, height=500, random_state=21, max_font_size=110, background_color="white").generate(all_words)
    for all_words in topic_wise_all_words
]

# Create a 4x2 subplot grid
fig, axes = plt.subplots(nrows=int(math.ceil(len(topic_wise_wordcloud)/2)), ncols=2, figsize=(15, 30))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Iterate over the axes and plot documents
for i, ax in enumerate(axes):
    if i < len(topic_wise_wordcloud):
        ax.imshow(topic_wise_wordcloud[i], interpolation="bilinear")
        ax.axis('off')
        ax.set_title(f'Topic {i+1}')
    else:
        ax.axis('off')  # Turn off the axis for any empty subplot

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()
