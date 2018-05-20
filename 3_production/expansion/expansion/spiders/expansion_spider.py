
import re
import os.path
import csv
import scrapy
import datetime

from scrapy.spiders import CrawlSpider, Rule
from scrapy.http.request import Request

from ..items import ExpansionItem


BASE_URL = 'http://www.expansion.com/hemeroteca/'


class ExpansionSpider(scrapy.Spider):

	def __init__(self, crawl_date):
		"""
		Initialize Spider with crawling datetime (specified in `run_spider.py`)

		:param crawl_date: crawling datetime
		"""

		print("\nInitializing Expansion spider ...\n", "-"*80)

		try:
			if isinstance(crawl_date, datetime.datetime): # check if argument is datetime.datetime
				self.crawl_date = crawl_date
				self.start_urls = [BASE_URL + self.crawl_date.strftime("%Y/%m/%d") + "/"]
				print("\nCrawl date selected is:", self.crawl_date.strftime("%Y-%m-%d"), "\n")

		except TypeError:
			print("\nArgument type not valid.")
			pass

	name = 'expansion'
	allowed_domains = ['www.expansion.com']
	# start_urls = start_urls_list
	custom_settings = {
		'ITEM_PIPELINES': {
			'expansion.expansion.pipelines.ItemsPipeline': 400
		}
	}


	def parse(self, response):

		self.logger.info('Parsing general...')

		if response.status == 200:

			raw_articles = response.xpath("//div[@id='destino']//article[@class='noticia']")

			# for item in raw_articles:
			for idx, item in enumerate(raw_articles):

				article = ExpansionItem()

				article = {
					"title": "",
					"url": ""
				}

				# title
				raw_title = item.xpath(".//h1/a/text()").extract()[0]
				if raw_title:
					# print idx, raw_title.extract()[0]
					article['title'] = raw_title

				# url
				raw_url = item.xpath(".//h1/a/@href").extract()[0]
				if raw_url:
					article['url'] = raw_url

				yield article
