
import re
import os.path
import csv
import scrapy
import datetime

from scrapy.spiders import CrawlSpider, Rule
from scrapy.http.request import Request

from ..items import CincodiasItem


BASE_URL = 'https://cincodias.elpais.com/tag/fecha/'


class CincodiasSpider(scrapy.Spider):

	def __init__(self, crawl_date):
		"""
		Initialize Spider with crawling datetime (specified in `run_spider.py`)

		:param crawl_date: crawling datetime
		"""

		print("\nInitializing Cincodias spider ...\n", "-"*80)

		try:
			if isinstance(crawl_date, datetime.datetime): # check if argument is datetime.datetime
				self.crawl_date = crawl_date

				start_urls_list = []
				for i in range(1,4):
					start_urls_list.append( BASE_URL + self.crawl_date.strftime("%Y%m%d") + "/" + str(i) )

				self.start_urls = start_urls_list
				print("\nCrawl date selected is:", self.crawl_date.strftime("%Y-%m-%d"), "\n")

		except TypeError:
			print("\nArgument type not valid.")
			pass

	name = 'cincodias'
	allowed_domains = ['cincodias.elpais.com']
	# start_urls = start_urls_list
	custom_settings = {
		'ITEM_PIPELINES': {
			'cincodias.cincodias.pipelines.ItemsPipeline': 400
		}
	}


	def parse(self, response):

		self.logger.info('Parsing general...')

		if response.status == 200:

			raw_urls = response.xpath('//article[@class="articulo"]//h2[@class="articulo-titulo"]//a')

			# for item in raw_articles:
			for idx, item in enumerate(raw_urls):

				article = CincodiasItem()

				article = {
					"title": "",
					"url": ""
				}

				# title
				raw_title = item.xpath("./text()").extract()[0]
				if raw_title:
					# print idx, raw_title.extract()[0]
					article['title'] = raw_title

				# url
				raw_url = item.xpath('./@href').extract()[0]
				if raw_url:
					article['url'] = "https://cincodias.elpais.com"+raw_url

				yield article
