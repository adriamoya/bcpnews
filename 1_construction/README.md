# Directory structure

* __1_mail_download:__ download all newsletter emails, parse them and extract all article urls (target articles)
* __2_scrapy_projects:__ scrapy projects to download desired article urls from the past
* __3_newspaper_scraper:__ use `newspaper3k` to retrieve standardized info from article urls

# Requirements

* [newspaper3k](https://newspaper.readthedocs.io/en/latest/)
* [Scrapy](https://scrapy.org/)
