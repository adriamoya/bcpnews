
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

# Logger
from utils.set_logger import set_logger
logger = set_logger('run_spiders')

# Input control: crawl date

# raw_input('Specify crawling date (YYYYMMDD): ') # raw_input() in python 2 / input() in python 3
# raise ValueError('The input date format expected is YYYYMMDD. Please try again.')

if len(sys.argv) > 1: # if passed as an argument
    if len(sys.argv[1]) == 8: # valid YYYYMMDD date format
        crawl_date_input = sys.argv[1]
        crawl_date = datetime.datetime.strptime(crawl_date_input,"%Y%m%d")
        print "\n", "Crawling datetime is:", crawl_date.strptime(crawl_date_input,"%Y%m%d"), "\n"
    else:
        raise ValueError('The input date format expected is YYYYMMDD. Please try again.')

else: # if no argument specified
    crawl_date = datetime.datetime.today()
    print "\n", "Crawling datetime not specified. Crawling newspapers for today:", crawl_date, "\n"


# proceed ?
while True:
    start_crawling = raw_input("Do you wish to proceed? [y/n]: ")
    if start_crawling.lower() in ["y", ""]:
        print ""
        break
    elif start_crawling.lower() == "n": # abort the process
        print "\nStopping process ...\n"
        exit()
    else:
        print "\nPlease type y or n.\n"

logger.info('crawl_date: %s' % crawl_date.strftime("%Y%m%d"))

# Multiple scraping

process = CrawlerProcess()

process.crawl(ExpansionSpider, crawl_date=crawl_date)
process.crawl(CincodiasSpider, crawl_date=crawl_date)
process.crawl(ElconfidencialSpider, crawl_date=crawl_date)

process.start() # the script will block here until all crawling jobs are finished


# Ouput check results

def read_urls(news_paper):

    input_file_path = './urls_%s.json' % news_paper.lower()

    # read data
    data = []
    with open(input_file_path) as input_file:

    	for line in input_file:
    		data.append(json.loads(line))
    print "(%d)" % len(data), "\t", news_paper

# print results
print "\n", "Process finished\n", "-"*80
read_urls("Expansion")
read_urls("Cincodias")
read_urls("ElConfidencial")
print ""
