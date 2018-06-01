
# ArticleScraper class built on top of Article from newspaper

import json
import logging
import datetime

from newspaper import Article


class ArticleScraper(Article):

	''' For a given article url, it downloads and parses some specific data and writes a JSON in the output_file '''

	def __init__(self, url, output_file, logger):

		''' Initialize ArticleScraper '''

		# logger
		self.logger = logger

		# other variables
		self.output_file = output_file

		self.article_obj = {}
		self.article_obj['url'] = url

		if self.article_obj:
			self.article = Article(url, language='es')
			self.parse_article()


	def parse(self):

		self.throw_if_not_downloaded_verbose()

		self.doc = self.config.get_parser().fromstring(self.html)
		self.clean_doc = copy.deepcopy(self.doc)

		if self.doc is None:
			# `parse` call failed, return nothing
			return

		# TODO: Fix this, sync in our fix_url() method
		parse_candidate = self.get_parse_candidate()
		self.link_hash = parse_candidate.link_hash  # MD5

		document_cleaner = DocumentCleaner(self.config)
		output_formatter = OutputFormatter(self.config)

		title = self.extractor.get_title(self.clean_doc)
		self.set_title(title)

		authors = self.extractor.get_authors(self.clean_doc)
		self.set_authors(authors)

		meta_lang = self.extractor.get_meta_lang(self.clean_doc)
		self.set_meta_language(meta_lang)

		if self.config.use_meta_language:
			self.extractor.update_language(self.meta_lang)
			output_formatter.update_language(self.meta_lang)

		meta_favicon = self.extractor.get_favicon(self.clean_doc)
		self.set_meta_favicon(meta_favicon)

		meta_description = \
			self.extractor.get_meta_description(self.clean_doc)
		self.set_meta_description(meta_description)

		canonical_link = self.extractor.get_canonical_link(
		self.url, self.clean_doc)
		self.set_canonical_link(canonical_link)

		tags = self.extractor.extract_tags(self.clean_doc)
		self.set_tags(tags)

		meta_keywords = self.extractor.get_meta_keywords(
			self.clean_doc)
		self.set_meta_keywords(meta_keywords)

		meta_data = self.extractor.get_meta_data(self.clean_doc)
		self.set_meta_data(meta_data)

		self.publish_date = self.extractor.get_publishing_date(
			self.url,
			self.clean_doc)

		# Before any computations on the body, clean DOM object
		self.doc = document_cleaner.clean(self.doc)

		self.top_node = self.extractor.calculate_best_node(self.doc)
		if self.top_node is not None:
			video_extractor = VideoExtractor(self.config, self.top_node)
			self.set_movies(video_extractor.get_videos())

			self.top_node = self.extractor.post_cleanup(self.top_node)
			self.clean_top_node = copy.deepcopy(self.top_node)

			text, article_html = output_formatter.get_formatted(
				self.top_node)
			self.set_article_html(article_html)
			self.set_text(text)

		self.fetch_images()

		self.is_parsed = True
		self.release_resources()


	def my_converter(self, o):

		''' Convert datetime to unicode (str) '''

		if isinstance(o, datetime.datetime):
			return o.__str__()


	def parse_article(self):

		''' Download, Parse and NLP a given article '''

		try:
			# download source code
			self.article.download()

			# parse code
			self.article.parse()

			# populate article obj with parsed data
			self.article_obj['title'] = self.article.title
			self.article_obj['authors'] = self.article.authors
			self.article_obj['publish_date'] = self.article.publish_date
			self.article_obj['text'] = self.article.text
			self.article_obj['top_image'] = self.article.top_image

			# article nlp
			self.article.nlp()

			# populate article obj with nlp data
			self.article_obj['summary'] = self.article.summary
			self.article_obj['keywords'] = self.article.keywords

			print(self.article_obj)

			return self.dump_article()

		except:
			self.logger.error("Article not donwloaded: %s" % self.article_obj['url'])
			pass

	def dump_article(self):

		''' Dump article to output JSON file '''

		line = json.dumps(dict(self.article_obj), default=self.my_converter, ensure_ascii=False) + "\n"

		self.output_file.write(line)

		return self.article_obj
