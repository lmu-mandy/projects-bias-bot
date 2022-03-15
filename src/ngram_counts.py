import pandas as pd
import pickle
import numpy as np
from operator import itemgetter
from collections import Counter

from Article import Article

# FUNCTIONS

def merge_dictionaries(*dicts):
	output = Counter(dict())
	for d in dicts:
		output.update(Counter(d))
	return output

def get_unique_count(d):
	all_items = merge_dictionaries(d["Conservative"], d["Independent"], d["Liberal"], d["Other"])
	unique_count = len(all_items)
	return unique_count

articles_file = f"../data/processed_articles.p"
with open(articles_file, 'rb') as f:
	articles = pickle.load(f)

data_file = "../data/300-samples.csv"
df = pd.read_csv(data_file)

# Replace headers
new_headers = dict()
for header in df.columns:
	new_headers[header] = header.replace("Answer.","")

df = df.rename(columns=new_headers)

print(f"Headers: {df.columns}")

#Find n-grams that are most predictive of a "biased" label for Democrats/Republicans

unigram_counts = {"Conservative":dict(), "Independent":dict(), "Liberal":dict(), "Other":dict()}
bigram_counts = {"Conservative":dict(), "Independent":dict(), "Liberal":dict(), "Other":dict()}
trigram_counts = {"Conservative":dict(), "Independent":dict(), "Liberal":dict(), "Other":dict()}

for i, article in enumerate(articles):

	print(f"\n\n\nArticle: {article.number} from {article.source}")

	rows = df.loc[df['articleNumber'] == article.number]
	biased_labels = rows['bias-question'].to_numpy()
	politics = rows['politics'].to_numpy()

	for j, biased in enumerate(biased_labels):

		if biased == 'is-not-biased':
			bias_factor = -1
		else:
			bias_factor = 1

		political_identity = politics[j]

		for word_num, word in enumerate(article.text):

			# Add unigrams
			if word in unigram_counts[political_identity]:
				unigram_counts[political_identity][word] += 1 * bias_factor
			else:
				unigram_counts[political_identity][word] = 1

			# Add bigrams
			if word_num > 0:
				prev_word = article.text[word_num-1]
				bigram = f"{prev_word} {word}"
				if bigram in bigram_counts[political_identity]:
					bigram_counts[political_identity][bigram] += 1 * bias_factor
				else:
					bigram_counts[political_identity][bigram] = 1

			# Add bigrams
			if word_num > 0 and word_num < len(article.text)-1:
				prev_word = article.text[word_num-1]
				next_word = article.text[word_num+1]
				trigram =  f"{prev_word} {word} {next_word}"
				if trigram in trigram_counts[political_identity]:
					trigram_counts[political_identity][trigram] += 1 * bias_factor
				else:
					trigram_counts[political_identity][trigram] = 1

num_unigrams = get_unique_count(unigram_counts)
num_bigrams = get_unique_count(bigram_counts)
num_trigrams = get_unique_count(trigram_counts)

N = 10
print(f"\n\nMost frequent unigrams resulting in 'biased' score:\n")
for party in unigram_counts:

	if party in ["Independent","Other"]:
		continue

	print(f"\n\n\n==================================\nPolitical leaning: {party}\n==================================\n")
	top_results = dict(sorted(unigram_counts[party].items(), key = itemgetter(1), reverse = True)[:N])
  
	# printing result
	#print(f"Most frequent unigrams for respondents who identified as {party}:")
	for unigram in top_results:
		print(f"{unigram}: {round(100*top_results[unigram]/num_unigrams,4)}% (count = {top_results[unigram]})")


print(f"\n\nMost frequent bigrams resulting in 'biased' score:\n")
for party in bigram_counts:

	if party in ["Independent","Other"]:
		continue

	print(f"\n\n\n==================================\nPolitical leaning: {party}\n==================================\n")
	top_results = dict(sorted(bigram_counts[party].items(), key = itemgetter(1), reverse = True)[:N])
  
	# printing result
	#print(f"Most frequent bigrams for respondents who identified as {party}:\n")
	for bigram in top_results:
		print(f"{bigram}: {round(100*top_results[bigram]/num_bigrams,4)}% (count = {top_results[bigram]})")

print(f"\n\nMost frequent trigrams resulting in 'biased' score:\n")
for party in trigram_counts:

	if party in ["Independent","Other"]:
		continue

	print(f"\n\n\n==================================\nPolitical leaning: {party}\n==================================\n")
	top_results = dict(sorted(trigram_counts[party].items(), key = itemgetter(1), reverse = True)[:N])
  
	# printing result
	#print(f"Most frequent trigrams for respondents who identified as {party}:\n")
	for trigram in top_results:
		print(f"{trigram}: {round(100*top_results[trigram]/num_trigrams,4)}% (count = {top_results[trigram]})")




