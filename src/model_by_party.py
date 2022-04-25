import numpy as np
import pandas as pd
import pickle
import re
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from tqdm import tqdm
import gensim.downloader as api
from sklearn.metrics import classification_report

from Article import Article

class LSTM(nn.Module):
	def __init__(self, input_size, emb_dim, output_size, num_layers, embeds=None):
		super().__init__()
		self.emb = nn.Embedding(input_size, emb_dim)
		if embeds is not None:
			self.emb.weight = nn.Parameter(torch.Tensor(embeds))
		
		self.lstm = nn.LSTM(emb_dim, emb_dim, num_layers=num_layers, bidirectional=True)
		self.linear = nn.Linear(emb_dim*2, output_size)
		# self.linear = nn.Linear(emb_dim, output_size)
		
		
	def forward(self, input_seq):

		# print(f"input seq size: {input_seq.size()}")

		embeds = self.emb( input_seq )

		# print(f"embeds size: {embeds.size()}")		

		output_seq , (h_last, c_last) = self.lstm( embeds )

		h_direc_1 = h_last[4,:,:]
		h_direc_2 = h_last[5,:,:]
		h_direc_12 = torch.cat( (h_direc_1, h_direc_2), dim=1 )

		return self.linear(h_direc_12)

		# return self.linear(h_last)

def load_vocab(data):
	word_to_index = {"FOX":1,"CNN":2,"BBC":3}
	vocab = []
	count = 4
	for item in data:
		for word in item.headline + item.text:
			if word not in word_to_index:
				vocab.append(word)
				word_to_index[word] = count 
				count += 1
	return vocab, word_to_index

def split_data(data, word_to_index, party):
	processed_data = []
	for article in data:
		if article.party != party:
			continue
		datapoint = [word_to_index[word] for word in article.headline] + [word_to_index[word] for word in article.text]
		label = label_to_index[article.label]

		processed_data.append( (datapoint, label) )
	return processed_data[:math.floor(0.7*len(processed_data))], processed_data[math.floor(0.7*len(processed_data)):]


def process_batch(batch):
	x = torch.zeros((len(batch), max_len), dtype=torch.long)
	y = torch.zeros((len(batch)), dtype=torch.long)
	for idx, (text, label) in enumerate(batch):
		x[idx,:len(text)] = torch.Tensor(text)
		y[idx] = label
	return x.to(device), y.to(device)

def get_error(scores, labels):
    bs=scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches=indicator.sum()
    
    return 1-num_matches.float()/bs  

def evaluate(model, test_data):
	with torch.no_grad():
		model.eval()
		x_test, y_test = process_batch(test_data)

		x_test = x_test.view(-1, len(test_data))

		pred_y_test = model(x_test).view(-1,2)

		labels = y_test.tolist()
		predictions = [torch.argmax(pred).item() for pred in pred_y_test]

		print("Evaluation on test set:")
		print(classification_report(labels, predictions, target_names=["is-biased","is-not-biased"], zero_division=0))

device= torch.device("cpu")

# Run only for a single political group
# party = "Liberal"
party = "Conservative"

# Load data
print("Loading data...")
with open("../data/processed_articles.p", "rb") as f:
	data = pickle.load(f)

vocab, word_to_index = load_vocab(data)
label_to_index = {"is-biased":0, "is-not-biased":1}
max_len = max([len(article.headline + article.text) for article in data])

print("Creating train data set...")
train_data, test_data = split_data(data, word_to_index, party)

# Hyper parameters
input_size = len(word_to_index) 
output_size = 2 
num_layers = 3
batch_size = 16 
learning_rate = 0.0001
epochs = 5

#Load pre-trained word embeddings, if using them.
embeds = api.load('glove-twitter-25').vectors
emb_dim = embeds.shape[1]
# emb_dim = 200


# Build model
model = LSTM(input_size, emb_dim, output_size, num_layers, embeds).to(device)
criterion = nn.CrossEntropyLoss()

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=process_batch)

# Train loop
for epoch in range(epochs):

	print(f"\n\nEpoch {epoch}")
	evaluate(model, test_data)

	# learning_rate = learning_rate/2
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	model.train()

	running_error = 0
	count = 0

	for x,y in train_dataloader:

		if x.size()[0] != batch_size:
			continue 

		x = x.view(-1, batch_size)

		# print(f"x size: {x.size()}")
		# print(f"y size: {y.size()}")

		scores = model(x)
		scores = scores.view(-1,2)
		# print(f"scores size: {scores.size()}")

		loss = criterion(scores, y)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		error = get_error(scores, y)
		print(error)
		running_error += error.item()
		count += 1

	print("Error:", running_error/count)

# Evaluate
evaluate(model, test_data)























