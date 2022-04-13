import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from bs4 import BeautifulSoup

def remove_html_tags(text):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_urls(text):
    """Remove urls from a string"""
    return re.sub(r'http\S+', '', text)

def remove_punctuation(text):
	punc = '!()-[]{};:"\,<>./?@#$%^&*_~`'
	for char in text:
		if char in punc:
			text = text.replace(char,"")
	return text

def get_headline(source, text):

	if source == "FOX" or source == "CNN":
		exp = '<div><h1 class="(pg-)?headline">.*?>'
	else:
		exp = '<div><h1 class="ssrcss.*?" id="main-heading" tabindex="-1">.*?>'

	headline_exp = re.compile(exp)
	headline = remove_html_tags(headline_exp.match(text).group())
	text = re.sub(headline_exp, '', text)
	return headline, text

class Article:
	def __init__(self, text):
		self.text = text
		self.source = self.extract_source()

		self.headline, text = get_headline(self.source, self.text.lower())
		self.headline = self.process_text(self.headline)
		self.text = self.process_text(text.lower())

	def process_text(self,text):

		text = remove_html_tags(text)
		text = remove_urls(text)
		text = remove_punctuation(text)
		before_intro = True
		for char in text:
			if char == ">":
				before_intro = False 
				continue
			if not before_intro:
				text_without_intro += char

		text_tokens = word_tokenize(text)
		tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
		return tokens_without_sw

	def extract_source(self):
		article_soup = BeautifulSoup(self.text, features="html.parser")
		try:
			return article_soup.find("input", {"id": "newsOutlet"}).get('value')
		except Exception:
			print(f"Could not find outlet for article {self.location}")
			return ""
