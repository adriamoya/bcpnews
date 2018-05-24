
import gensim
import pprint
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

pp = pprint.PrettyPrinter(indent=4)


class Similarities(object):

	def __init__(self, df_articles):
		print("\nInitializing similarities ...")
		print("-"*80)
		self.df_articles = df_articles
		self.articles = df_articles['text'].values
		print("Number of articles:", "{:,}".format(len(self.articles)))
		self.build_tfidf()


	def build_tfidf(self):
		"""
		Train tf-idf model.
		"""
		print("\nBuilding tf-idf model ...")
		print("-"*80)
		# We will now use NLTK to tokenize
		print('Word tokenization ...')
		gen_docs = [[w.lower() for w in word_tokenize(text)]
		            for text in self.articles]

		# We will create a dictionary from a list of documents. A dictionary maps every word to a number.
		print("Creating dictionary ...")
		self.dictionary = gensim.corpora.Dictionary(gen_docs)
		print("Number of words in dictionary:", "{:,}".format(len(self.dictionary)))

		# Now we will create a corpus. A corpus is a list of bags of words.
		# A bag-of-words representation for a document just lists the number of times each word occurs in the document.
		print("Creating bag-of-words corpus ...")
		corpus = [self.dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

		# Now we create a tf-idf model from the corpus. Note that num_nnz is the number of tokens.
		print("Building tf-idf model from corpus ...")
		self.tf_idf = gensim.models.TfidfModel(corpus)
		print(self.tf_idf)

		# Now we will create a similarity measure object in tf-idf space.
		print("Creating similarity measures and storing ...")
		self.sims = gensim.similarities.Similarity('./sims',
		                                      self.tf_idf[corpus],
		                                      num_features=len(self.dictionary))
		print(self.sims)
		print(type(self.sims))
		print("Done.")


	def get_similarities(self, article):
		"""
		Get similarities for a given article.
		"""
		query_doc = [w.lower() for w in word_tokenize(article)]
		# print(query_doc)
		query_doc_bow = self.dictionary.doc2bow(query_doc)
		# print(query_doc_bow)
		query_doc_tf_idf = self.tf_idf[query_doc_bow]
		# print(query_doc_tf_idf)

		# We show an array of document similarities to query.
		document_sims = self.sims[query_doc_tf_idf]

		return document_sims


	def build_similarity_matrix(self):
		"""
		Generates the similarty matrix by getting the individual similarities of each article.
		"""
		print("\nBuilding similarity matrix ...")
		print("-"*80)

		self.sims_matrix = []
		for article in self.articles:
			document_sims = self.get_similarities(article)
			self.sims_matrix.append(document_sims)

		print("Done.")


	def find_similar_articles(self, threshold=0.5, verbose=True):
		"""
		Looks for similar articles ...
		"""
		print("\nLooking for similar items ...")
		print("-"*80)
		similar_articles = []
		for idx_article, article in enumerate(self.sims_matrix):
			for idx_sim_article, similarity in enumerate(article):
				if similarity > threshold and idx_article != idx_sim_article:
					similar_articles.append([(idx_article, idx_sim_article), similarity])

		print("Similar articles:")
		pp.pprint(similar_articles)

		# Print results
		if verbose:
			print("\nChecking similar articles ...")
			for item in similar_articles:
				print("")
				print("(%s)\t" % item[0][0], self.df_articles.loc[item[0][0]].title + "\n" + self.df_articles.loc[item[0][0]].url)
				print("(%s)\t" % item[0][1], self.df_articles.loc[item[0][1]].title + "\n" + self.df_articles.loc[item[0][1]].url)

		# return similar_articles





# raw_articles = []
# with open('./3_production/output/20180523_articles.json', encoding='utf-8') as f:
# 	for line in f:
# 		raw_articles.append(json.loads(line))
#
# df_documents = pd.DataFrame(raw_articles)
#
# raw_documents = df_documents['text'].values
#
# sim = Similarities(df_documents)
# sim.build_similarity_matrix()
# sim.find_similar_articles()
