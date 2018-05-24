
import pandas as pd


df_train = pd.read_csv('1_construction/3_newspaper_scraper/analyses/train.csv', sep=',')

df_train.columns

df_train_clr = df_train[['id', 'keywords', 'summary', 'title', 'text', 'date', 'flag']]

df_train_clr.shape

df_train_clr.to_csv('1_construction/3_newspaper_scraper/analyses/cleaned_datasets/train.csv', sep=",", na_rep="", mode="w", index=False, encoding='utf-8')


df_test = pd.read_csv('1_construction/3_newspaper_scraper/analyses/test.csv', sep=',')

df_test_clr = df_test[['id', 'keywords', 'summary', 'title', 'text', 'date']]

df_test_clr.shape

df_test_clr.to_csv('1_construction/3_newspaper_scraper/analyses/cleaned_datasets/test.csv', sep=",", na_rep="", mode="w", index=False, encoding='utf-8')
