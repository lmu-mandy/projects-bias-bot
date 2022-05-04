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
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer


print("Loading data...")
dataset = load_dataset('csv', data_files={'train': '../data/processed_transformer_train.csv', 'test': '../data/processed_transformer_test.csv'})

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
def tokenize_function(data):
  return tokenizer(data["text"], padding=True, truncation=True)

print("Tokenize data...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns('text')
tokenized_dataset = tokenized_dataset.class_encode_column("labels")
tokenized_dataset = tokenized_dataset.with_format('torch')
data_collator = DataCollatorWithPadding(tokenizer)

print("Train model...")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy='epoch'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()