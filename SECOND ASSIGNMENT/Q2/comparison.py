import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
import gensim.downloader as api
from keras.layers import TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, GlobalMaxPooling1D
import wget
import zipfile
import os

# Load dataset (IMDB Movie Review or custom dataset)
df = pd.read_csv('Train.csv')  # Ensure Train.csv is correctly formatted
df['label'] = df['label'].map({1: 'positive', 0: 'negative'})  # Correct mapping of labels
X = df['text']
y = df['label']

# Check for class imbalance
print(df['label'].value_counts())

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text vectorization for Word2Vec and GloVe
vectorizer = TextVectorization(max_tokens=5000)
vectorizer.adapt(X_train)

X_train_seq = vectorizer(X_train).numpy()
X_test_seq = vectorizer(X_test).numpy()

max_len = 100  # Set max sequence length
X_train_padded = pad_sequences(X_train_seq, maxlen=max_len)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len)

# ============================= Word2Vec Embeddings ==============================
# Load pre-trained Word2Vec model
w2v_model = api.load("word2vec-google-news-300")

# Create an embedding matrix for Word2Vec
vocab_size = len(vectorizer.get_vocabulary())
embedding_dim = 300
embedding_matrix_w2v = np.zeros((vocab_size, embedding_dim))

for i, word in enumerate(vectorizer.get_vocabulary()):
    try:
        embedding_vector = w2v_model[word]
        embedding_matrix_w2v[i] = embedding_vector
    except KeyError:
        pass  # Words not found in Word2Vec will be zeros.

# Build LSTM model for Word2Vec
model_w2v = Sequential()
model_w2v.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len, 
                        weights=[embedding_matrix_w2v], trainable=False))
model_w2v.add(LSTM(128, return_sequences=True))
model_w2v.add(GlobalMaxPooling1D())
model_w2v.add(Dense(10, activation='relu'))
model_w2v.add(Dense(1, activation='sigmoid'))

# Compile and train the Word2Vec model
model_w2v.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_w2v.fit(X_train_padded, y_train, epochs=10, batch_size=64, validation_data=(X_test_padded, y_test))

# Evaluate Word2Vec model
y_pred_w2v = (model_w2v.predict(X_test_padded) > 0.5).astype("int32")
accuracy_w2v = accuracy_score(y_test, y_pred_w2v)
precision_w2v = precision_score(y_test, y_pred_w2v, zero_division=1)
recall_w2v = recall_score(y_test, y_pred_w2v, zero_division=1)
f1_w2v = f1_score(y_test, y_pred_w2v, zero_division=1)

# ============================= GloVe Embeddings ==============================
# Download GloVe embeddings
glove_zip = 'glove.6B.zip'
if not os.path.exists(glove_zip):
    glove_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    wget.download(glove_url, glove_zip)

with zipfile.ZipFile(glove_zip, 'r') as zip_ref:
    zip_ref.extractall()

# Load GloVe embeddings
glove_embeddings = {}
with open('glove.6B.300d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        glove_embeddings[word] = vector

# Create embedding matrix for GloVe
embedding_matrix_glove = np.zeros((vocab_size, embedding_dim))
for i, word in enumerate(vectorizer.get_vocabulary()):
    if word in glove_embeddings:
        embedding_matrix_glove[i] = glove_embeddings[word]

# Build LSTM model for GloVe
model_glove = Sequential()
model_glove.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len, 
                          weights=[embedding_matrix_glove], trainable=False))
model_glove.add(LSTM(128, return_sequences=True))
model_glove.add(GlobalMaxPooling1D())
model_glove.add(Dense(10, activation='relu'))
model_glove.add(Dense(1, activation='sigmoid'))

# Compile and train the GloVe model
model_glove.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_glove.fit(X_train_padded, y_train, epochs=10, batch_size=64, validation_data=(X_test_padded, y_test))

# Evaluate GloVe model
y_pred_glove = (model_glove.predict(X_test_padded) > 0.5).astype("int32")
accuracy_glove = accuracy_score(y_test, y_pred_glove)
precision_glove = precision_score(y_test, y_pred_glove, zero_division=1)
recall_glove = recall_score(y_test, y_pred_glove, zero_division=1)
f1_glove = f1_score(y_test, y_pred_glove, zero_division=1)

# ============================= BERT Embeddings ==============================
# Load pre-trained BERT model
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Encode dataset for BERT
def encode_data(data, max_len):
    return tokenizer_bert(data.tolist(), padding=True, truncation=True, max_length=max_len, return_tensors='pt')

train_data = encode_data(X_train, max_len)
test_data = encode_data(X_test, max_len)

# Convert the labels to PyTorch tensors (no need for .values)
train_labels = torch.tensor(y_train)
test_labels = torch.tensor(y_test)


train_dataset = torch.utils.data.TensorDataset(train_data['input_ids'], train_data['attention_mask'], train_labels)
test_dataset = torch.utils.data.TensorDataset(test_data['input_ids'], test_data['attention_mask'], test_labels)

train_loader = DataLoader(train_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Fine-tune BERT
optimizer = torch.optim.Adam(model_bert.parameters(), lr=2e-5)

for epoch in range(5):  # Increased epochs for better fine-tuning
    model_bert.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model_bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader)}")

# Evaluate BERT model
model_bert.eval()
preds = []
for batch in test_loader:
    input_ids, attention_mask, labels = batch
    outputs = model_bert(input_ids=input_ids, attention_mask=attention_mask)
    preds.append(torch.argmax(outputs.logits, axis=1))

y_pred_bert = torch.cat(preds).cpu().numpy()
accuracy_bert = accuracy_score(y_test, y_pred_bert)
precision_bert = precision_score(y_test, y_pred_bert, zero_division=1)
recall_bert = recall_score(y_test, y_pred_bert, zero_division=1)
f1_bert = f1_score(y_test, y_pred_bert, zero_division=1)

# ============================= Results Comparison ==============================
print("Word2Vec:")
print(f"Accuracy: {accuracy_w2v}, Precision: {precision_w2v}, Recall: {recall_w2v}, F1-Score: {f1_w2v}")

print("GloVe:")
print(f"Accuracy: {accuracy_glove}, Precision: {precision_glove}, Recall: {recall_glove}, F1-Score: {f1_glove}")

print("BERT:")
print(f"Accuracy: {accuracy_bert}, Precision: {precision_bert}, Recall: {recall_bert}, F1-Score: {f1_bert}")
