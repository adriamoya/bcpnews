
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
from spiders.spiders import crawl_newspapers, process_all_newspapers

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

pause_execution()

logger.info('crawl_date: %s' % crawl_date.strftime("%Y%m%d"))

# Crawling newspapers.
print('\nCrawling the news...')
print('-'*80)
crawl_newspapers(crawl_date)
print('Done.')

# ------------------------------------ #
#  Download articles                   #
# ------------------------------------ #

pause_execution()

# Save all articles into a unique JSON (YYYYMMDD_articles.json`).
print('\nProcessing all the articles and saving them...')
print('-'*80)
process_all_newspapers(crawl_date)
print('Done.')
