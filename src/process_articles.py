import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
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

output_file = f"../data/processed_articles.p"
with open(output_file, 'wb') as f:
	pickle.dump(samples,f)