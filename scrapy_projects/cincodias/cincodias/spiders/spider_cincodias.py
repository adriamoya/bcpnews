
import re
import os.path
import csv
import scrapy
import datetime

from scrapy.conf import settings
from scrapy.spiders import CrawlSpider, Rule
from scrapy.http.request import Request

from cincodias.items import CincodiasItem



general_url = 'https://cincodias.elpais.com/tag/fecha/'

start_date = datetime.datetime.strptime("2017/04/10", "%Y/%m/%d")
end_date = datetime.datetime.strptime("2018/04/10", "%Y/%m/%d")
date_generated = [start_date + datetime.timedelta(days=x) for x in range(0, (end_date-start_date).days, 14)]

start_urls_list = []

for date in date_generated:  
	for i in range(1,4):
		start_urls_list.append( general_url+ date.strftime("%Y%m%d")+"/"+str(i))



class CincodiasSpider(scrapy.Spider):

	name = 'cincodias'
	allowed_domains = ['cincodias.elpais.com']
	start_urls = start_urls_list
	custom_settings = {
		'ITEM_PIPELINES': {
			'cincodias.pipelines.ItemsPipeline': 400
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