import gensim
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

raw_documents = pd.read_csv('/home/amoya/.kaggle/competitions/bcpnews/test.csv')['text'].values

test_article = raw_documents[0]


def train_similarties(raw_documents):
	"""
	Create a similarty matrix based on a collection of raw documents
	"""

	# # read articles
	# print('Reading data ...')
	# df_input = pd.read_csv('/home/amoya/.kaggle/competitions/bcpnews/test.csv')

	# # build documents
	# raw_documents = df_input.text.values
	print("Number of documents:",len(raw_documents))

	# We will now use NLTK to tokenize
	print('Word tokenization ...')
	gen_docs = [[w.lower() for w in word_tokenize(text)]
	            for text in raw_documents]

	# We will create a dictionary from a list of documents. A dictionary maps every word to a number.
	print("Creating dictionary ...")
	dictionary = gensim.corpora.Dictionary(gen_docs)
	print("Number of words in dictionary:",len(dictionary))
	for i in range(20):
	    print(i, dictionary[i])

	# Now we will create a corpus. A corpus is a list of bags of words.
	# A bag-of-words representation for a document just lists the number of times each word occurs in the document.
	print("Creating bag-of-words corpus ...")
	corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

	# Now we create a tf-idf model from the corpus. Note that num_nnz is the number of tokens.
	print("Building tf-idf model from corpus ...")
	tf_idf = gensim.models.TfidfModel(corpus)
	print(tf_idf)
	s = 0
	for i in corpus:
	    s += len(i)
	print(s)

	# Now we will create a similarity measure object in tf-idf space.
	print("Creating similarity measures and storing ...")
	sims = gensim.similarities.Similarity('./sims',
	                                      tf_idf[corpus],
	                                      num_features=len(dictionary))
	print(sims)
	print(type(sims))

	return sims, dictionary, tf_idf

def get_similarities(document, sims, dictionary, tf_idf):

	query_doc = [w.lower() for w in word_tokenize(document)]
	# print(query_doc)
	query_doc_bow = dictionary.doc2bow(query_doc)
	# print(query_doc_bow)
	query_doc_tf_idf = tf_idf[query_doc_bow]
	# print(query_doc_tf_idf)

	# We show an array of document similarities to query.
	document_sims = sims[query_doc_tf_idf]

	print(document_sims)

	return document_sims



sims, dictionary, tf_idf= train_similarties(raw_documents)

document_sims = get_similarities(test_article, sims, dictionary, tf_idf)