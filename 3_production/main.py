
import os
import sys
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

from utils.aux_functions import pause_execution
from utils.check_output import check_output

# ------------------------------------ #
#  Run spiders                         #
# ------------------------------------ #

# Create output directory if not exists
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Logger
from spiders.utils.set_logger import set_logger
logger = set_logger('main')

logger.info("")
logger.info("Initializing run spiders ...")

# Input control: crawl date
if len(sys.argv) > 1: # if passed as an argument
    if len(sys.argv[1]) == 8: # valid YYYYMMDD date format
        crawl_date_input = sys.argv[1]
        crawl_date = datetime.datetime.strptime(crawl_date_input,"%Y%m%d")
        print("\nCrawling datetime is:", crawl_date.strptime(crawl_date_input,"%Y%m%d"), "\n")
    else:
        raise ValueError('The input date format expected is YYYYMMDD. Please try again.')

else: # if no argument specified
    crawl_date = datetime.datetime.today()
    print("\nCrawling datetime not specified. Crawling newspapers for today:", crawl_date, "\n")


# Proceed ?
pause_execution()

logger.info('crawl_date: %s' % crawl_date.strftime("%Y%m%d"))

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

pause_execution()

from spiders.utils.article_scraper import ArticleScraper

# Output file
output_file_name = './output/%s_articles.json' % crawl_date.strftime("%Y%m%d")
output_file = codecs.open(output_file_name, 'w', encoding='utf-8')

logger.info("")
logger.info("Initializing download ...")

def process_newspaper(newspp, output_file, logger):

	input_file_path = "./output/urls_%s.json" % newspp

	# Read data.
	data = []
	with open(input_file_path, encoding="utf8") as input_file:

		for line in input_file:
			data.append(json.loads(line))


	# Download, parse and store articles.
	logger.info('(%s) %s urls' % (str(len(data)), newspp))

	for article in data:
		new_article = ArticleScraper(article['url'], output_file, logger)


process_newspaper('expansion', output_file, logger)
process_newspaper('cincodias', output_file, logger)
process_newspaper('elconfidencial', output_file, logger)
process_newspaper('eleconomista', output_file, logger)

# Close output file.
output_file.close()

#check_output(output_file_name)
