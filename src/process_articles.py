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

#file = "../data/300-samples.csv"
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


# Process text
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
	count = count + 1
	print(count)

# for i, article_num in enumerate(df['articleNumber']):
# 	batch_num = df['batch'][i]
# 	if not np.isnan(batch_num):
# 		print("sure")
# 		print(f"\n\n[{i}/{len(df)}] -- Article {int(article_num)}")
			
# 		article_file = df["Article Location"][i]

# 		f = open(article_file)
# 		text = f.read()

# 		source = df['newsOutlet'][i]
# 		label = df['bias-question'][i]
# 		party = df['politics'][i]

# 		article = Article(text, source, article_num, label, party)
# 		articles.append(article)

# 		print(f"\n\n\nArticle from {source}:")
# 		print("Headline: ", article.headline)
# 		print(article.text)


output_file = f"../data/processed_articles.p"
with open(output_file, 'wb') as f:
	pickle.dump(samples,f)