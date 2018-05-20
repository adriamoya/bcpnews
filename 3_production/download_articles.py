
import os
import json
import codecs
import logging

from utils.file_len import file_len
from utils.set_logger import set_logger
from utils.article_scraper import ArticleScraper

# read log file and get crawling datetime
log_file ="./run_spiders.log"

try:
    with open(log_file) as f:
        f = f.readlines()
    crawl_date = f[-1][-9:-1]
except:
    print("\nNo log file found...\n")
    exit()

# create Logger
logger = set_logger('download_articles')

# proceed ?
while True:
    print("\nCrawl date is %s" % crawl_date)
    start_crawling = input("Do you wish to proceed? [y/n]: ")
    if start_crawling.lower() in ["y", ""]:
        print("")
        break
    elif start_crawling.lower() == "n": # abort the process
        print("\nStopping process ...\n")
        exit()
    else:
        print("\nPlease type y or n.\n")


# output file
output_file = codecs.open('./output/%s_articles.json' % crawl_date, 'w', encoding='utf-8')

logger.info("")
logger.info("Initializing download ...")

def process_newspaper(newspp, output_file, logger):

	input_file_path = "./urls_%s.json" % newspp

	# Read data.

	data = []
	with open(input_file_path) as input_file:

		for line in input_file:
			data.append(json.loads(line))


	# Download, parse and store articles.

	logger.info('(%s) %s urls' % (str(len(data)), newspp))

	for article in data:
		new_article = ArticleScraper(article['url'], output_file, logger)


process_newspaper('expansion', output_file, logger)
process_newspaper('cincodias', output_file, logger)
process_newspaper('elconfidencial', output_file, logger)

logger.info("(%d) articles downloaded" % file_len('./output/%s_articles.json' % crawl_date))

# Close output file.

output_file.close()
