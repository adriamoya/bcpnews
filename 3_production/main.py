
import os
import sys
import json
import codecs
import logging
import datetime

import scrapy
from scrapy.crawler import CrawlerProcess

# Import spiders
from expansion.expansion.spiders.expansion_spider import ExpansionSpider
from cincodias.cincodias.spiders.cincodias_spider import CincodiasSpider
from elconfidencial.elconfidencial.spiders.elconfidencial_spider import ElconfidencialSpider

# ------------------------------------ #
#  Run spiders                         #
# ------------------------------------ #

# Create output directory if not exists
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Logger
from utils.set_logger import set_logger
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
while True:
    start_crawling = input("Do you wish to proceed? [y/n]: ") # raw_input for python2
    if start_crawling.lower() in ["y", ""]:
        print("")
        break
    elif start_crawling.lower() == "n": # abort the process
        print("\nStopping process ...\n")
        exit()
    else:
        print("\nPlease type y or n.\n")

logger.info('crawl_date: %s' % crawl_date.strftime("%Y%m%d"))

# Multiple scraping

process = CrawlerProcess()

process.crawl(ExpansionSpider, crawl_date=crawl_date)
process.crawl(CincodiasSpider, crawl_date=crawl_date)
process.crawl(ElconfidencialSpider, crawl_date=crawl_date)

process.start() # the script will block here until all crawling jobs are finished


# Ouput check results
#
# def read_urls(news_paper):
#
#     input_file_path = './output/urls_%s.json' % news_paper.lower()
#
#     # read data
#     data = []
#     with open(input_file_path) as input_file:
#
#     	for line in input_file:
#     		data.append(json.loads(line))
#     print("(%d)" % len(data), "\t", news_paper)
#
# # print results
# print("\nProcess finished\n", "-"*80)
# read_urls("Expansion")
# read_urls("Cincodias")
# read_urls("ElConfidencial")
# print("")


# ------------------------------------ #
#  Download                            #
# ------------------------------------ #

from utils.article_scraper import ArticleScraper

# Output file
output_file = codecs.open('./output/%s_articles.json' % crawl_date.strftime("%Y%m%d"), 'w', encoding='utf-8')

logger.info("")
logger.info("Initializing download ...")

def process_newspaper(newspp, output_file, logger):

	input_file_path = "./output/urls_%s.json" % newspp

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

# Close output file.

output_file.close()
