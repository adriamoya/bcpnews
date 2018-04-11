
import newspaper
# import pymongo
# from pymongo import MongoClient
import datetime

# def send_to_mongo(client, news_collection):
#     db = client.database_names()
#     db = client.newspaper
#     newspaper_db = db.newspaper
#     new_entries = newspaper_db.insert_many(news_collection)
#     return new_entries

def news_collector(principal_url, memoize = True):

    article_run_collection = []
    # build the newspaper content (check all the urls in the web site)
    news_paper = newspaper.build(principal_url, memoize_articles = memoize, language='es')

    print('El periodico seleccionado es: '+news_paper.brand)
    print(news_paper.description)
    print('---------------------------------------------------------')
    print('ready for trying to include', news_paper.size(), 'articles')
    print('Categorias ---------------------------------------------------------')

    # categorias en el periodico
    for category in news_paper.category_urls():
        print(category)
    print('Feeds ---------------------------------------------------------')

    # feeds en el periodico
    for feed_url in news_paper.feed_urls():
        print(feed_url)
    print('Ariculos ---------------------------------------------------------')

    # articulos en el periodico
    for i, article in enumerate(news_paper.articles):
        print(str(i)+'. '+article.url+' ------')
        article.download()
        if article.is_valid_url():
            article.parse()
            print('Autores: ')
            print(article.authors)
            print('Fecha de publicacion: ')
            print(article.publish_date)
            print('Texto: ')
            print(article.text)
            print('Top image: ')
            print(article.top_image)
            print('Images: ')
            print(article.images)
            print('Movies: ')
            print(article.movies)
            article.nlp()
            print('Summary: ')
            print(article.summary)
            print('Keywords: ')
            print(article.keywords)
            # if len(article.text) > 0:
            #     #print(i)
            #     if [article.publish_date] != [None]:
            #          publication_date = datetime.datetime.isoformat(article.publish_date)
            #     else:
            #         publication_date = datetime.datetime.now().isoformat()
            #     article_entry = {
            #         'principal_url': principal_url,
            #         'article_url': article.url,
            #         'title': article.title,
            #         'publication_date': publication_date,
            #         'scrape_date':  datetime.datetime.now().isoformat(),
            #         'tags': list(article.tags),
            #         'keywords_npl': article.keywords,
            #         'keywords_md': article.meta_data['keywords'],
            #         'keywords_md_news': article.meta_data['news_keywords'],
            #         'html': article.html,
            #         'text': article.text,
            #         'summary': article.summary,
            #     }
            #     article_run_collection.append(article_entry)
    # send the new information to MongoDB database
    # new_entries = send_to_mongo(client, article_run_collection)
    # n_new_entries = len(article_run_collection)
    # print(n_new_entries,'article(s) was(were) included.')

# principal_url = 'http://www.cincodias.es/'
# principal_url = 'http://www.elpais.es/'
# principal_url = 'http://www.elconfidencial.es/'
# principal_url = 'http://www.eleconomista.es/'
principal_url = 'http://www.expansion.es/'
# principal_url = 'http://www.lavanguardia.es/'
news_collector(principal_url, memoize = False)