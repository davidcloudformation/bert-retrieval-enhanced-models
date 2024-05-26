import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained('./results')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load data
queries = pd.read_csv('data/queries.csv')
documents = pd.read_csv('data/documents.csv')

# Scoring function
def score(query, document):
    inputs = tokenizer(query, document, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return scores

# Example scoring
query = "What is machine learning?"
document = "This is a sample document about machine learning."
print(score(query, document))