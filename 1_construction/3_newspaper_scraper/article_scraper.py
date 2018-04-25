
# ArticleScraper class built on top of Article from newspaper

import json
import datetime

from newspaper import Article


class ArticleScraper(Article):

	''' For a given article url, it downloads and parses some specific data and writes a JSON in the output_file '''

	def __init__(self, url, output_file):

		''' Initialize ArticleScraper '''

		self.output_file = output_file

		self.article_obj = {}
		self.article_obj['url'] = url

		if self.article_obj:
			self.article = Article(url)
			self.parse_article()


	def my_converter(self, o):

		''' Convert datetime to unicode (str) '''

		if isinstance(o, datetime.datetime):
			return o.__str__()


	def parse_article(self):

		''' Download, Parse and NLP a given article '''

		try:
			# download source code
			self.article.download()

			# parse code
			self.article.parse()

			# populate article obj with parsed data
			self.article_obj['title'] = self.article.title
			self.article_obj['authors'] = self.article.authors
			self.article_obj['publish_date'] = self.article.publish_date
			self.article_obj['text'] = self.article.text
			self.article_obj['top_image'] = self.article.top_image

			# article nlp
			self.article.nlp()

			# populate article obj with nlp data
			self.article_obj['summary'] = self.article.summary
			self.article_obj['keywords'] = self.article.keywords

			print(self.article_obj)

			return self.dump_article()
		
		except:
			pass


	def dump_article(self):

		''' Dump article to output JSON file '''

		line = json.dumps(dict(self.article_obj), default=self.my_converter, ensure_ascii=False) + "\n"

		self.output_file.write(line)

		return self.article_obj
