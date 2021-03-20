import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import zipfile
import seaborn as sns

from ds100_utils import fetch_and_cache

# Ensure that Pandas shows at least 280 characters in columns, so we can see full tweets
pd.set_option('max_colwidth', 280)

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

# Create a dataframe to store Trump's tweets
trump = pd.DataFrame(all_tweets, columns=['id', 'created_at', 'source', 'text', 'full_text', 'retweet_count']).set_index('id').rename(columns={'created_at':'time'})
trump['time'] = pd.to_datetime(trump['time'])
trump['text'] = trump['text'].fillna('')
trump['full_text'] = trump['full_text'].fillna('')
trump['text'] = trump[['text', 'full_text']].sum(axis=1).values
trump = trump.drop(columns='full_text')

sentiment = pd.read_csv('vader_lexicon.txt', sep='\t', index_col=0, usecols=[0, 1], names=['word','polarity'])

# Transform text to lowercase
trump['text'] = trump['text'].str.lower()

punct_re = r'[^\w\s]'
trump['no_punc'] = trump['text'].str.replace(punct_re, ' ', regex=True)

# Clean and format data following the guidelines: https://cran.r-project.org/web/packages/tidyr/vignettes/tidy-data.html
tidy_format = trump['no_punc'].str.split(expand=True).stack().reset_index(level=1).rename(columns={'level_1':'num', 0:'word'})

# Determine the polarity of the tweets using VADER lexicon
trump['polarity'] = tidy_format.reset_index().merge(sentiment, how='left', on='word').fillna(0).groupby('id').sum()['polarity']

print('Most negative tweets:')
for t in trump.sort_values('polarity').head()['text']:
    print('\n  ', t)
    
trump.sort_values('polarity').head()

print('Most positive tweets:')
for t in trump.sort_values('polarity', ascending=False).head()['text']:
    print('\n  ', t)

# Find tweets that contain 'fox' and 'nytimes'
contains_fox = tidy_format[tidy_format['word'] == 'fox'].reset_index().groupby(['id', 'word']).count()
contains_nytimes = tidy_format[tidy_format['word'] == 'nytimes'].reset_index().groupby(['id', 'word']).count()
merged_fox = trump.merge(contains_fox, on='id')['polarity']
merged_nytimes = trump.merge(contains_nytimes, on='id')['polarity']

# Plot the distribution of tweet sentiments for 'fox' and 'nytimes'
sns.distplot(merged_fox, kde=True, label='fox')
sns.distplot(merged_nytimes, kde=True, label='nytimes')
plt.legend()
plt.title('Distribution of Tweet Sentiments Containing fox and nytimes')
plt.show()

# Find tweets that contain 'clinton' and 'obama'
contains_clinton = tidy_format[tidy_format['word'] == 'clinton'].reset_index().groupby(['id', 'word']).count()
contains_obama = tidy_format[tidy_format['word'] == 'obama'].reset_index().groupby(['id', 'word']).count()
merged_clinton = trump.merge(contains_clinton, on='id')['polarity']
merged_obama = trump.merge(contains_obama, on='id')['polarity']

# Plot the distrubtion of tweet sentiments for 'clinton' and 'obama'
sns.distplot(merged_clinton, kde=True, label='clinton')
sns.distplot(merged_obama, kde=True, label='obama')
plt.legend()
plt.title('Distribution of Tweet Sentiments Containing clinton and obama')
plt.show()