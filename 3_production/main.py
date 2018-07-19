# -*- coding: utf-8 -*-

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

from utils.aux_functions import get_date_input, pause_execution
from utils.check_output import check_output

# Create output directory if not exists
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Logger
from spiders.utils.set_logger import set_logger
logger = set_logger('main')

logger.info("")
logger.info("Initializing run spiders ...")

# Request input date to crawl. Default is today().
crawl_date = get_date_input() #datetime.datetime.strptime('20180608',"%Y%m%d")

logger.info('crawl_date: %s' % crawl_date.strftime("%Y%m%d"))

# ------------------------------------ #
#  Run spiders                         #
# ------------------------------------ #

pause_execution()

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
