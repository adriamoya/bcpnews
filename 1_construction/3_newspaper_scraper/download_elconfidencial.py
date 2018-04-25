
import os
import json
import codecs

from article_scraper import ArticleScraper


# I/O files.

input_file_path = '../scrapy_projects/elconfidencial/urls_elconfidencial.json'
output_file 	= codecs.open('articles_elconfidencial.json', 'w', encoding='utf-8')


# Read data.

data = []
with open(input_file_path) as input_file:

	for line in input_file:
		data.append(json.loads(line))


# Download, parse and store articles.

for article in data:
	new_article = ArticleScraper(article['url'], output_file)


# Close output file.

output_file.close()