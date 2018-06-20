
import os
import sys
import json
import codecs
import logging
import datetime

# Import spiders
from spiders.spiders import crawl_newspapers, process_all_newspapers

# Import other functions
from spiders.utils.set_logger import set_logger
from utils.aux_functions import get_date_input, pause_execution

# ------------------------------------ #
#  Initiate process                    #
# ------------------------------------ #

# Set the logger file `main.log`.
logger = set_logger('main')

# Create output directory if not exists.
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Request input date to crawl. Default is today().
crawl_date = get_date_input() #datetime.datetime.strptime('20180608',"%Y%m%d")

# Running date.
date = crawl_date.strftime("%Y%m%d")

# Proceed ?
pause_execution()

# Log scrapping date.
logger.info('crawl_date: %s' % crawl_date.strftime("%Y%m%d"))

# ------------------------------------ #
#  Run spiders                         #
# ------------------------------------ #

# Crawling newspapers.
print('\nCrawling the news...')
print('-'*80)
crawl_newspapers(crawl_date)
print('Done.')

# ------------------------------------ #
#  Download articles                   #
# ------------------------------------ #

# Save all articles into a unique JSON (YYYYMMDD_articles.json`).
print('\nProcessing all the articles and saving them...')
print('-'*80)
process_all_newspapers(crawl_date)
print('Done.')
