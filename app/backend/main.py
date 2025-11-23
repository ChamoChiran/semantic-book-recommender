import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.embeddings.base import Embeddings
from langchain_text_splitters import CharacterTextSplitter

# -----------------------------
# Setup base directory
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# -----------------------------
# Load books dataset
# -----------------------------
books = pd.read_csv(os.path.join(DATA_DIR, "books_with_emotions.csv"))
books['large_thumbnail'] = books['thumbnail'] + '&file=w800'
books['large_thumbnail'] = np.where(
    books['large_thumbnail'].isna(),
    os.path.join(IMAGES_DIR, "cover-not-found.png"),
    books['large_thumbnail']
)

# -----------------------------
# Embedding model
# -----------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
device = "cpu"
model.to(device)
model.eval()

def create_embeddings(texts, batch_size=32):
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        with torch.no_grad():
            tokens = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            outputs = model(**tokens)
            token_embeddings = outputs.last_hidden_state
            attention_mask = tokens["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeds.extend(embeddings.cpu().tolist())
    return all_embeds

class SimpleEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return create_embeddings(texts, batch_size=32)
    def embed_query(self, text):
        return create_embeddings([text], batch_size=1)[0]

emb = SimpleEmbeddings()

# -----------------------------
# Chroma persistence
# -----------------------------
persist_dir = "./chroma_db_cpu"

if not os.path.exists(persist_dir):
    raw_documents = TextLoader(os.path.join(DATA_DIR, 'tagged_descriptions.txt'), encoding='utf-8').load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator="\n", keep_separator=False)
    documents = text_splitter.split_documents(raw_documents)
    print('Remaking the db...')
    db_books = Chroma.from_documents(documents, embedding=emb, persist_directory=persist_dir)
else:
    db_books = Chroma(embedding_function=emb, persist_directory=persist_dir)

# -----------------------------
# Recommendation logic
# -----------------------------
def retrieve_semantic_recommendations(query, category=None, tone=None, initial_top_k=10, final_top_k=16):
    recs = db_books.similarity_search(query, k=initial_top_k)
    book_list = [int(rec.page_content.strip('""').split()[0]) for rec in recs]
    books_recs = books[books['isbn13'].isin(book_list)].head(final_top_k)

    if category and category != 'All':
        books_recs = books_recs[books_recs['simple_categories'] == category][:final_top_k]

    if tone == 'Happy':
        books_recs.sort_values(by='joy', ascending=False, inplace=True)
    elif tone == 'Surprising':
        books_recs.sort_values(by='surprise', ascending=True, inplace=True)
    elif tone == 'Angry':
        books_recs.sort_values(by='anger', ascending=False, inplace=True)
    elif tone == 'Suspenseful':
        books_recs.sort_values(by='fear', ascending=False, inplace=True)
    elif tone == 'Sad':
        books_recs.sort_values(by='sadness', ascending=False, inplace=True)

    return books_recs

def recommend_books(query, category, tone):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        description = row['description']
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."
        authors_split = row['authors'].split(';')
        if len(authors_split) == 2:
            authors_str = f'{authors_split[0]} and {authors_split[1]}'
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row['authors']
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row['large_thumbnail'], caption))
    return results

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI()

# Allow frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class Query(BaseModel):
    query: str
    category: str
    tone: str

@app.post("/recommend")
def recommend(query: Query):
    return recommend_books(query.query, query.category, query.tone)

@app.get("/categories")
def get_categories():
    return ['All'] + sorted(books['simple_categories'].unique())

@app.get("/tones")
def get_tones():
    return ['All', 'Happy', 'Surprising', 'Angry', 'Suspenseful', 'Sad']