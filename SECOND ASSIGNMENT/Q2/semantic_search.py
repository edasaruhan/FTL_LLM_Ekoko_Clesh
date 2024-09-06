import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import faiss
from flask import Flask, request

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example documents (can be replaced with any corpus)
documents = [
    "The climate is changing rapidly.",
    "Polar bears are affected by the melting ice.",
    "Solar energy is an alternative to fossil fuels.",
    "Global warming is causing sea levels to rise.",
    "Biodiversity is being lost due to deforestation."
]

# Function to generate BERT embeddings for text
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Return the mean of the last hidden state to get the sentence embedding
    return torch.mean(outputs.last_hidden_state, dim=1).squeeze()

# Create embeddings for all documents
embeddings = []
for doc in documents:
    embedding = get_bert_embedding(doc)
    embeddings.append(embedding)

# Convert embeddings to numpy array and store in FAISS index
embedding_matrix = np.array([emb.numpy() for emb in embeddings])
d = embedding_matrix.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(d)   # L2 distance index
index.add(embedding_matrix)    # Add the embeddings to the FAISS index

print(f"Added {index.ntotal} document embeddings to the FAISS index.")

# Function to search for similar documents using FAISS
def search(query, top_k=5):
    query_embedding = get_bert_embedding(query).unsqueeze(0).numpy()
    distances, indices = index.search(query_embedding, top_k)
    results = [documents[i] for i in indices[0]]
    return results

# Flask app for the search interface
app = Flask(__name__)

@app.route('/')
def home():
    return '''
        <h1>Semantic Search Engine</h1>
        <form action="/search" method="POST">
            <label for="query">Enter your search query:</label><br>
            <input type="text" id="query" name="query"><br><br>
            <input type="submit" value="Search">
        </form>
    '''

@app.route('/search', methods=['POST'])
def search_results():
    query = request.form['query']
    results = search(query)
    
    return f'''
        <h1>Search Results for: "{query}"</h1>
        <ul>
            {''.join([f"<li>{result}</li>" for result in results])}
        </ul>
        <a href="/">Go back</a>
    '''

if __name__ == '__main__':
    app.run(debug=True)
