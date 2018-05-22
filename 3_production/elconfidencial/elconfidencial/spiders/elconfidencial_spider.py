
import re
import os.path
import csv
import scrapy
import datetime

from scrapy.spiders import CrawlSpider, Rule
from scrapy.http.request import Request

from ..items import ElconfidencialItem


BASE_URL = 'https://www.elconfidencial.com/hemeroteca/'


class ElconfidencialSpider(scrapy.Spider):

	def __init__(self, crawl_date):
		"""
		Initialize Spider with crawling datetime (specified in `run_spider.py`)

		:param crawl_date: crawling datetime
		"""

		print("\nInitializing ElConfidencial spider ...\n", "-"*80)

		try:
			if isinstance(crawl_date, datetime.datetime): # check if argument is datetime.datetime
				self.crawl_date = crawl_date
				self.start_urls = [BASE_URL + self.crawl_date.strftime("%Y-%m-%d") + "/1"]
				print("\nCrawl date selected is:", self.crawl_date.strftime("%Y-%m-%d"), "\n")

		except TypeError:
			print("\nArgument type not valid.")
			pass

	name = 'elconfidencial'
	allowed_domains = ['www.elconfidencial.com']
	# start_urls = start_urls_list
	custom_settings = {
		'ITEM_PIPELINES': {
			'elconfidencial.elconfidencial.pipelines.ItemsPipeline': 400
		}
	}


	def parse(self, response):

		self.logger.info('Parsing general...')

		if response.status == 200:

			raw_articles = response.xpath("//article//a")

			# for item in raw_articles:
			for idx, item in enumerate(raw_articles):

				article = ElconfidencialItem()

				article = {
					"title": "",
					"url": ""
				}

				# title
				try:
					raw_title = item.xpath("./@title").extract()[0]
					if raw_title:
						article['title'] = raw_title
				except:
					raw_title = item.xpath("./text()").extract()[0]
					if raw_title:
						article['title'] = raw_title

				# url
				raw_url = item.xpath("./@href").extract()[0]
				if raw_url:

					# check if the url contains 'https:' or not
					if 'http' not in raw_url:
						article['url'] = 'https:' + raw_url
					else:
						article['url'] = raw_url

				yield article
