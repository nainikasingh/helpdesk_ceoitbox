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
import nltk
from fuzzywuzzy import process

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')

# Initialize FastAPI app
app = FastAPI()

# Load NLP models
nlp = spacy.load("en_core_web_sm")
sentence_transformers.SENTENCE_TRANSFORMERS_HOME = "./models"
embedder = SentenceTransformer("multi-qa-mpnet-base-dot-v1")  # Transformer for embedding
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Google Sheets authentication
gc = gspread.service_account(filename='service_account.json')  # Update with correct credentials

# Static Google Sheet URL
SHEET_URL = "https://docs.google.com/spreadsheets/d/1-5ao5vcVOSCyD91TUgPSPYAOOv5f_Cxt5zKDwM7mwg0/edit?usp=sharing"

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Load Google Sheet data
def load_sheets_data():
    sheet = gc.open_by_url(SHEET_URL).sheet1
    data = sheet.get_all_values()
    df = pd.DataFrame(data)
    
    # Set the correct header row
    df.columns = df.iloc[0]  # Use the first row as the header
    df = df[2:]  # Skip the rows above the actual data
    df = df.reset_index(drop=True)
    
    # Ensure column names are trimmed of extra spaces
    df.columns = df.columns.str.strip()
    
    # Print column names for debugging
    logging.info("Columns in the dataframe: %s", df.columns.tolist())
    
    # Check available columns before dropping
    expected_columns = ["Sheet Name", "Issue Text", "Solution Text", "Solution Image"]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing expected columns: {missing_columns}")
    
    # Clean up and filter necessary columns
    df = df.dropna(subset=["Issue Text", "Solution Text"])
    df['processed_text'] = df['Issue Text'].apply(preprocess_text)
    
    return df

# Create FAISS index (cached)
def create_faiss_index(sentences):
    embeddings = np.array([embedder.encode(sent) for sent in sentences], dtype=np.float32)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# Search function using FAISS (vectorized and optimized)
def search_query(query, df, index, embeddings):
    query_embedding = embedder.encode([query]).astype(np.float32)
    _, faiss_result = index.search(query_embedding, 3)
    keyword_results = df[df['processed_text'].str.contains(query, case=False, na=False)][['Solution Text', 'Sheet Name', 'Solution Image']]
    
    solutions = []
    for idx in faiss_result[0]:
        row = df.iloc[idx]
        solutions.append(f"{row['Solution Text']} {{in {row['Sheet Name']}}}")
        solutions.append(f"Solution Image: {row['Solution Image']}")
    
    for _, row in keyword_results.iterrows():
        solutions.append(f"{row['Solution Text']} {{in {row['Sheet Name']}}}")
        solutions.append(f"Solution Image: {row['Solution Image']}")
    
    return solutions

# Extract entities from queries (asynchronous execution)
def extract_entities(query):
    doc = nlp(query)
    return [ent.text for ent in doc.ents]

# Improved Sheet Name Matching
def match_sheet_name(query, available_sheets):
    best_match, score = process.extractOne(query, available_sheets)
    if score > 80:  # Adjust threshold as needed
        return best_match
    return None

@app.get("/query")
async def get_solution(query: str):
    try:
        df = load_sheets_data()
        available_sheets = [ws.title for ws in gc.open_by_url(SHEET_URL).worksheets()]
        
        extracted_entities = extract_entities(query)
        matched_sheet = match_sheet_name(query, available_sheets)
        
        if matched_sheet:
            df = df[df["Sheet Name"] == matched_sheet]
            solutions = search_query(query, df, *create_faiss_index(df['processed_text'].tolist()))
        else:
            solutions = search_query(query, df, *create_faiss_index(df['processed_text'].tolist()))
        
        if not solutions:
            logging.warning(f"Low confidence query: {query}")
            return {"message": "No direct solution found. Please refine your query."}
        
        return {"solutions": solutions, "entities": extracted_entities}
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))