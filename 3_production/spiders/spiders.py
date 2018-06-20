
import os
import json
import codecs
import logging
import datetime

import scrapy
from scrapy.crawler import CrawlerProcess

# Import spiders
from spiders.expansion.expansion.spiders.expansion_spider import ExpansionSpider
from spiders.cincodias.cincodias.spiders.cincodias_spider import CincodiasSpider
from spiders.elconfidencial.elconfidencial.spiders.elconfidencial_spider import ElconfidencialSpider
from spiders.eleconomista.eleconomista.spiders.eleconomista_spider import EleconomistaSpider

from spiders.utils.article_scraper import ArticleScraper
from spiders.utils.set_logger import set_logger

# ------------------------------------ #
#  Run spiders                         #
# ------------------------------------ #

# Logger
logger = set_logger('main')

logger.info("")
logger.info("Initializing run spiders ...")

def crawl_newspapers(crawl_date):
    # Multiple scraping
    process = CrawlerProcess()

    process.crawl(ExpansionSpider, crawl_date=crawl_date)
    process.crawl(CincodiasSpider, crawl_date=crawl_date)
    process.crawl(ElconfidencialSpider, crawl_date=crawl_date)
    process.crawl(EleconomistaSpider, crawl_date=crawl_date)

    process.start() # the script will block here until all crawling jobs are finished


# ------------------------------------ #
#  Download articles                   #
# ------------------------------------ #

def process_newspaper(newspp, output_file, logger):

	dirname = os.path.dirname(os.path.dirname(__file__))
	input_file_path = "output/urls_%s.json" % newspp
	fullpath = os.path.join(dirname, input_file_path)
	print(fullpath)

	# Read data.
	data = []
	with open(fullpath) as input_file:

		for line in input_file:
			data.append(json.loads(line))

	# Download, parse and store articles.
	logger.info('(%s) %s urls' % (str(len(data)), newspp))

	for i, article in enumerate(data):
		new_article = ArticleScraper(article['url'], output_file, logger)

def process_all_newspapers(crawl_date):
    # Output file
    output_file = codecs.open('./output/%s_articles.json' % crawl_date.strftime("%Y%m%d"), 'w', encoding='utf-8')

    logger.info("")
    logger.info("Initializing download ...")

    process_newspaper('expansion', output_file, logger)
    process_newspaper('cincodias', output_file, logger)
    process_newspaper('elconfidencial', output_file, logger)
    process_newspaper('eleconomista', output_file, logger)

    # Close output file.
    output_file.close()
