import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from Article import Article

file = "../data/300-samples.csv"
df = pd.read_csv(file)

# Replace headers
new_headers = dict()
for header in df.columns:
	new_headers[header] = header.replace("Answer.","")

df = df.rename(columns=new_headers)

# Process text
articles = []
for i, article_num in enumerate(df['articleNumber']):

	print(f"\n\n[{i}/{len(df)}] -- Article {article_num}")
	
	article_file = f"../data/articles/{article_num}.txt"
	f = open(article_file)
	text = f.read()

	source = df['newsOutlet'][i]
	label = df['bias-question'][i]
	party = df['politics'][i]

	article = Article(text, source, article_num, label, party)
	articles.append(article)

	print(f"\n\n\nArticle from {source}:")
	print("Headline: ", article.headline)
	print(article.text)


output_file = f"../data/processed_articles.p"
with open(output_file, 'wb') as f:
	pickle.dump(articles,f)