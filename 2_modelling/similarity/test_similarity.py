
import json
import pandas as pd

from similarity import Similarities

raw_articles = []
with open('../../3_production/output/20180523_articles.json', encoding='utf-8') as f:
	for line in f:
		raw_articles.append(json.loads(line))

df = pd.DataFrame(raw_articles)

print("\nTesting similarities with 20180523 articles ...")

sim = Similarities(df)
sim.build_similarity_matrix()
# sim.find_similar_articles()
sim.return_similar_articles(threshold=0.5, verbose=True)

import nltk
import string
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

token_dict = {}
stemmer = PorterStemmer()
translator = str.maketrans('', '', string.punctuation)

def stem_tokens(tokens, stemmer):
	stemmed = []
	for item in tokens:
		stemmed.append(stemmer.stem(item))
	return stemmed

def tokenize(text):
	tokens = nltk.word_tokenize(text)
	stems = stem_tokens(tokens, stemmer)
	return stems

with open('../../3_production/output/20180523_articles.json', encoding='utf-8') as f:
	for idx, line in enumerate(f):
		if idx < 3:
			article_obj = json.loads(line)
			text = article_obj['text']
			lowers = text.lower()
			no_punctuation = lowers.translate(translator)
			token_dict[idx] = no_punctuation

#this can take some time
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(token_dict.values())

# response = tfidf.transform(token_dict.values())
print(tfidf.vocabulary_)
print(tfs)