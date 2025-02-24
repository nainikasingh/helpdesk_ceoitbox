from fastapi import FastAPI, HTTPException
import gspread
import pandas as pd
import faiss
import spacy
import logging
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from collections import defaultdict

# Initialize FastAPI app
app = FastAPI()

# Load NLP models
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("multi-qa-mpnet-base-dot-v1")  # Transformer for embedding
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Google Sheets authentication
gc = gspread.service_account(filename='service_account.json')  # Update with correct credentials

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Load Google Sheet data
def load_sheets_data(sheet_url):
    sheet = gc.open_by_url(sheet_url).sheet1
    data = sheet.get_all_values()
    df = pd.DataFrame(data)
    
    # Set the correct header row
    df.columns = df.iloc[0]  # Use the first row as the header
    df = df[2:]  # Skip the rows above the actual data
    df = df.reset_index(drop=True)
    
    # Ensure column names are trimmed of extra spaces
    df.columns = df.columns.str.strip()
    
    # Print column names for debugging
    print("Columns in the dataframe:", df.columns.tolist())
    
    # Check available columns before dropping
    expected_columns = ["Sheet Name", "Issue Text", "Solution Text", "Solution Image"]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing expected columns: {missing_columns}")
    
    # Clean up and filter necessary columns
    df = df.dropna(subset=["Issue Text", "Solution Text"])
    df['processed_text'] = df['Issue Text'].apply(preprocess_text)
    return df

# Create FAISS index
def create_faiss_index(sentences):
    embeddings = np.array([embedder.encode(sent) for sent in sentences], dtype=np.float32)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# Search function combining FAISS and keyword matching
def search_query(query, df, index, embeddings):
    query_embedding = embedder.encode([query]).astype(np.float32)
    _, faiss_result = index.search(query_embedding, 3)
    keyword_results = df[df['processed_text'].str.contains(query, case=False, na=False)]['Solution Text']
    return df.iloc[faiss_result[0]]['Solution Text'].tolist() + keyword_results.tolist()

# Extract entities from queries
def extract_entities(query):
    doc = nlp(query)
    return [ent.text for ent in doc.ents]

@app.get("/query")
def get_solution(query: str, sheet_url: str):
    try:
        df = load_sheets_data(sheet_url)
        index, embeddings = create_faiss_index(df['processed_text'].tolist())
        extracted_entities = extract_entities(query)
        solutions = search_query(query, df, index, embeddings)
        if not solutions:
            logging.warning(f"Low confidence query: {query}")
            return {"message": "No direct solution found. Please refine your query."}
        return {"solutions": solutions, "entities": extracted_entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
