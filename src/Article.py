import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

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

class Article:
	def __init__(self,text, source, article_num):
		self.text = self.process_text(text)
		self.source = source	
		self.number = article_num

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