# Classifying news articles to build a tailored newsletter

Real case application for a financial consulting firm.

For over more than two years, an experienced employee from this firm collected the most relevant press articles from different newspapers each day. The goal was spreading a tailored, daily newsletter to all employees with interesting news about the financial and economic landscape (with both national and international scope). This process could take up to 2 hours every day and comprise task such as reading multiple newspaper front pages, keep track of cross-cutting topics from other sectors, structuring and ranking articles and manually writing and sending the newsletter.

![alt text](https://github.com/adriamoya/bcpnews/blob/master/Trump-fake.jpg)

### Project structure

```
.
├── 1_construction
│   ├── 1_mail_download
│   ├── 2_scrapy_projects
│   │   ├── cincodias
│   │   │   └── cincodias
│   │   │       └── spiders
│   │   ├── elconfidencial
│   │   │   └── elconfidencial
│   │   │       └── spiders
│   │   └── expansion_hemeroteca
│   │       └── expansion_hemeroteca
│   │           └── spiders
│   └── 3_newspaper_scraper
│       └── analyses
├── 2_modelling
│   ├── classification
│   └── similarity
└── 3_production
    ├── output
    ├── spiders
    │   ├── cincodias
    │   │   └── cincodias
    │   │       └── spiders
    │   ├── elconfidencial
    │   │   └── elconfidencial
    │   │       └── spiders
    │   ├── eleconomista
    │   │   └── eleconomista
    │   │       └── spiders
    │   ├── expansion
    │   │   └── expansion
    │   │       └── spiders
    │   └── utils
    └── utils
```
