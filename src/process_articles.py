import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import numpy as np
import math
from tqdm import tqdm

from Article import Article
from Sample import Sample
from ArticlePreprocessor import ArticlePreprocessor

file = "../data/preprocessed/samples.csv"
df = pd.read_csv(file)
df = df.dropna(subset=['Article Location'])

# Replace headers
new_headers = dict()
for header in df.columns:
	new_headers[header] = header.replace("Answer.","")

df = df.rename(columns=new_headers)

# Create article objects from every article in BBC CNN and FOX
pre = ArticlePreprocessor()

possible_articles = {}

for key in pre.article_counts.keys():
	for article_num in tqdm(range(0, pre.article_counts[key]+1), ascii=True, desc=f"Loading {key} articles"):
		article_path = f"{pre.write_path}/{key}/{article_num}.txt"

		f = open(article_path)
		article_text = f.read()

		possible_articles[article_path] = Article(article_text)

# Process samples
samples = []
transform_samples = []
count = 0
for _, row in tqdm(df.iterrows(), ascii=True, desc=f"Loading samples"):
	article = possible_articles[row['Article Location']]
	header = article.headline
	text = article.text
	source = article.source
	label = row['bias-question']
	party = row['politics']

	sample = Sample(header, text, source, label, party)
	samples.append(sample)
	transform_samples.append([' '.join(text), label])

df = pd.DataFrame(transform_samples, columns=['text', 'labels'])
train=df.sample(frac=0.8,random_state=200)
test=df.drop(train.index)
#.reset_index(drop=True, inplace=True)
train.to_csv('../data/processed_transformer_train.csv', index=False)
test.to_csv('../data/processed_transformer_test.csv', index=False)


output_file = f"../data/processed_articles.p"
with open(output_file, 'wb') as f:
	pickle.dump(samples,f)