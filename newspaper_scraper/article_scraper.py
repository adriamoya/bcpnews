
# Imports

import os
import json
import codecs
import pprint
import datetime

from newspaper import Article

# Variables

pp = pprint.PrettyPrinter(indent=4)

input_file_path = '../expansion_hemeroteca/articles.json'
output_file 	= codecs.open('articles_standar.json', 'w', encoding='utf-8')


# Read data

data = []
with open(input_file_path) as input_file:

	for line in input_file:
		data.append(json.loads(line))

# Functions (perhaps create a class on top of Article class)

def parse_article(article_obj):

	''' Download, Parse and NLP a given article '''

	if article_obj:

		article = Article(article_obj['url'])

		# download source code
		article.download()

		# parse code
		article.parse()

		# populate article obj with parsed data
		article_obj['title'] = article.title
		article_obj['authors'] = article.authors
		article_obj['publish_date'] = article.publish_date
		article_obj['text'] = article.text
		article_obj['top_image'] = article.top_image

		# article nlp
		article.nlp()

		# populate article obj with nlp data
		article_obj['summary'] = article.summary
		article_obj['keywords'] = article.keywords

		return article_obj


def my_converter(o):

	''' Convert datetime to unicode (str) '''

	if isinstance(o, datetime.datetime):

		return o.__str__()


def dump_article(output_file, article_obj):

	''' Dump article to output JSON file '''

	line = json.dumps(dict(article_obj), default=my_converter, ensure_ascii=False) + "\n"

	output_file.write(line)

	return article_obj


# Execution

for item in data:

	article_obj = {}

	article_obj['url'] = item['url']

	article_obj = parse_article(article_obj)

	article_obj = dump_article(output_file, article_obj)

	pp.pprint(article_obj)


# Close output file

output_file.close()








# article = Article(article_obj['url'])

# class Article2(Article):

# 	def __int__(self, url):
		
# Article2(url)
