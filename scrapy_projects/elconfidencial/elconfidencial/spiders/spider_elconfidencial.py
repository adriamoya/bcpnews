
import re
import os.path
import csv
import scrapy
import datetime

from scrapy.conf import settings
from scrapy.spiders import CrawlSpider, Rule
from scrapy.http.request import Request

from elconfidencial.items import ElconfidencialItem


general_url = 'https://www.elconfidencial.com/hemeroteca/'

start_date = datetime.datetime.strptime("2017-04-10", "%Y-%m-%d")
end_date = datetime.datetime.strptime("2018-04-10", "%Y-%m-%d")
date_generated = [start_date + datetime.timedelta(days=x) for x in range(0, (end_date-start_date).days, 8)]

start_urls_list = []

for date in date_generated:
   start_urls_list.append( general_url+ date.strftime("%Y-%m-%d")+"/1/" )


class ElconfidencialUrls(scrapy.Spider):

	name = 'elconfidencial_urls'
	allowed_domains = ['www.elconfidencial.com']
	start_urls = start_urls_list
	custom_settings = {
		'ITEM_PIPELINES': {
			'elconfidencial.pipelines.ItemsPipeline': 400
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
