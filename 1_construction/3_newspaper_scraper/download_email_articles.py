
import os
import json
import codecs

from article_scraper import ArticleScraper


# I/O files.

input_file_path = '../mail_download/emails.json'
output_file 	= codecs.open('articles_email.json', 'w', encoding='utf-8')


# Read data.

data = []
with open(input_file_path) as input_file:

	for line in input_file:
		data.append(json.loads(line))


# Download, parse and store articles.

for email in data:
	try:
		if len(email['urls']) > 0:
			for url in email['urls']:
				new_article = ArticleScraper(url, output_file)
	except:
		pass

# Close output file.

output_file.close()