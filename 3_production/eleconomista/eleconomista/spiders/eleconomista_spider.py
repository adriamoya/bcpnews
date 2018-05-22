
import re
import os.path
import csv
import scrapy
import datetime

from scrapy.spiders import CrawlSpider, Rule
from scrapy.http.request import Request

from ..items import EleconomistaItem


BASE_URL = 'https://www.eleconomista.es/'
section_list = ['mercados-cotizaciones', 'economia', 'empresas-finanzas', 'tecnologia']

class EleconomistaSpider(scrapy.Spider):

	def __init__(self, crawl_date):
		"""
		Initialize Spider with crawling datetime (specified in `run_spider.py`)

		:param crawl_date: crawling datetime
		"""

		print("\nInitializing Eleconomista spider ...\n", "-"*80)

		try:
			if isinstance(crawl_date, datetime.datetime): # check if argument is datetime.datetime
				self.crawl_date = crawl_date
				start_urls_list = []
				for section in section_list:
					start_urls_list.append(BASE_URL + section + "/")
				self.start_urls = start_urls_list
				print("\nCrawl date selected is:", self.crawl_date.strftime("%Y-%m-%d"), "\n")

		except TypeError:
			print("\nArgument type not valid.")
			pass

	name = 'eleconomista'
	allowed_domains = ['www.eleconomista.es']
	custom_settings = {
		'ITEM_PIPELINES': {
			'eleconomista.eleconomista.pipelines.ItemsPipeline': 400
		}
	}


	def parse(self, response):

		self.logger.info('Parsing general...')

		if response.status == 200:

			raw_articles = response.xpath("//div[contains(@class,'cols')]//h1[@itemprop='headline']/a")

			# for item in raw_articles:
			for idx, item in enumerate(raw_articles):

				article = EleconomistaItem()

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

					article['url'] = article['url'].replace("https", "http")

				yield article
