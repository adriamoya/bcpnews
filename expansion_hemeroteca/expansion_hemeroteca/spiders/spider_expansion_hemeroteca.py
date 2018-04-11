
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

class ExpansionHemerotecaSpider(scrapy.Spider):

	name = 'expansion_hemeroteca'
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
					"author": "",
					"url": "",
					"seccion": "",
					"summary": "",
					"raw_timestamp": "",
					"timestamp": "2018-04-06",
					"text": "",
					"kicker": ""
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

				# seccion
				try:
					raw_seccion = raw_url.split("/")[3]
					if raw_seccion:
						article['seccion'] = raw_seccion
				except:
					pass

				# summary
				try:
					raw_summary = item.xpath('.//div[@class="entradilla"]/p/text()').extract()[0]
					if raw_summary:
						article['summary'] = raw_summary
				except:
					pass


				request = scrapy.Request(
					raw_url,
					callback=self.parse_article,
					dont_filter=True)
				
				request.meta['item'] = article

				yield request


	def parse_article(self, response):

		article = response.meta['item']

		if response.status == 200:

			# timestamp
			try:
				raw_timestamp = response.xpath('//time/@datetime').extract()[0]
				if raw_timestamp:
					article['raw_timestamp'] = raw_timestamp
			except:
				pass

			# kicker
			try:
				raw_kicker = response.xpath("//h2[@class='kicker']//text()").extract()[0]
				if raw_kicker:
					article['kicker'] = raw_kicker
			except:
				pass

			# author
			try:
				raw_author = response.xpath("//li[@class='author-name']//text()").extract()[0]
				if raw_author:
					article['author'] = raw_author
			except:
				pass

			# text
			try:
				raw_summary = response.xpath('//p[@class="entradilla"]')
				raw_text = [raw_summary.xpath('.//text()').extract()[0].strip()]
				if raw_summary:

					for p in raw_summary.xpath('./following-sibling::p//text()').extract():
						raw_text.append(p.strip())

					article['text'] = ' '.join(raw_text)
			except:
				pass

			yield article





