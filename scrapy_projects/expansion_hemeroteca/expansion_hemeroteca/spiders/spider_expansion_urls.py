
import re
import os.path
import csv
import scrapy
import datetime

from scrapy.conf import settings
from scrapy.spiders import CrawlSpider, Rule
from scrapy.http.request import Request

from expansion_hemeroteca.items import ExpansionHemerotecaItem


general_url = 'http://www.expansion.com/hemeroteca/'

start_urls_list = [general_url+'2018/04/06']



general_url = 'http://www.expansion.com/hemeroteca/'
start_date = datetime.datetime.strptime("2017/04/10", "%Y/%m/%d")
end_date = datetime.datetime.strptime("2018/04/10", "%Y/%m/%d")
date_generated = [start_date + datetime.timedelta(days=x) for x in range(0, (end_date-start_date).days, 7)]

start_urls_list = []

for date in date_generated:  
   start_urls_list.append( general_url+ date.strftime("%Y/%m/%d")+"/" )



class ExpansionUrls(scrapy.Spider):

	name = 'expansion_urls'
	allowed_domains = ['www.expansion.com']
	start_urls = start_urls_list
	custom_settings = {
		'ITEM_PIPELINES': {
			'expansion_hemeroteca.pipelines.ItemsPipeline': 400
		}
	}


	def parse(self, response):

		self.logger.info('Parsing general...')

		if response.status == 200:

			raw_articles = response.xpath("//div[@id='destino']//article[@class='noticia']")

			# for item in raw_articles:
			for idx, item in enumerate(raw_articles):

				article = ExpansionHemerotecaItem()

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