import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

# Load data
documents = pd.read_csv('data/documents.csv')
queries = pd.read_csv('data/queries.csv')

# Merge documents and queries for training
data = pd.merge(queries, documents, left_on='id', right_on='id', suffixes=('_query', '_doc'))

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['query'], examples['document'], padding='max_length', truncation=True)

tokenized_data = data.apply(tokenize_function, axis=1)

# Convert to torch dataset
class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(tokenized_data, data['id'], test_size=0.2)

train_encodings = tokenizer(train_texts['query'], train_texts['document'], truncation=True, padding=True)
val_encodings = tokenizer(val_texts['query'], val_texts['document'], truncation=True, padding=True)

train_dataset = RetrievalDataset(train_encodings, train_labels)
val_dataset = RetrievalDataset(val_encodings, val_labels)

# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Training
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()