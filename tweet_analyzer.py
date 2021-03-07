import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import zipfile
import seaborn as sns
import re
from ds100_utils import fetch_and_cache

# Ensure that Pandas shows at least 280 characters in columns, so we can see full tweets
pd.set_option('max_colwidth', 280)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')

sns.set()
sns.set_context("talk")

# Download the dataset
data_url = 'http://www.ds100.org/fa19/assets/datasets/hw4-realdonaldtrump_tweets.json.zip'
file_name = 'hw4-realdonaldtrump_tweets.json.zip'

dest_path = fetch_and_cache(data_url=data_url, file=file_name)
print(f'Located at {dest_path}')

my_zip = zipfile.ZipFile(dest_path, 'r')
with my_zip.open('hw4-realdonaldtrump_tweets.json', 'r') as f:
    all_tweets = json.load(f)

trump = pd.DataFrame(all_tweets, columns=['id', 'created_at', 'source', 'text', 'full_text', 'retweet_count']).set_index('id').rename(columns={'created_at':'time'})
trump['time'] = pd.to_datetime(trump['time'])
trump['text'] = trump['text'].fillna('')
trump['full_text'] = trump['full_text'].fillna('')
trump['text'] = trump[['text', 'full_text']].sum(axis=1).values
trump = trump.drop(columns='full_text')

sent = pd.read_csv('vader_lexicon.txt', sep='\t', index_col=0, usecols=[0, 1], names=['word','polarity'])
sent.head()

trump['text'] = trump['text'].str.lower()
trump.head()

punct_re = r'[^\w\s]'
trump['no_punc'] = trump['text'].str.replace(punct_re, ' ', regex=True)

tidy_format = trump['no_punc'].str.split(expand=True).stack().reset_index(level=1).rename(columns={'level_1':'num', 0:'word'})

trump['polarity'] = tidy_format.reset_index().merge(sent, how='left', on='word').fillna(0).groupby('id').sum()['polarity']

print('Most negative tweets:')
for t in trump.sort_values('polarity').head()['text']:
    print('\n  ', t)
    
trump.sort_values('polarity').head()

print('Most positive tweets:')
for t in trump.sort_values('polarity', ascending=False).head()['text']:
    print('\n  ', t)

contains_fox = tidy_format[tidy_format['word'] == 'fox'].reset_index().groupby(['id', 'word']).count()
contains_nytimes = tidy_format[tidy_format['word'] == 'nytimes'].reset_index().groupby(['id', 'word']).count()
merged_fox = trump.merge(contains_fox, on='id')['polarity']
merged_nytimes = trump.merge(contains_nytimes, on='id')['polarity']
sns.distplot(merged_fox, kde=True, label='fox')
sns.distplot(merged_nytimes, kde=True, label='nytimes')
plt.legend()
plt.title('Distribution of Tweet Sentiments Containing fox and nytimes')
