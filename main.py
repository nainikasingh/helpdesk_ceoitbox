from fastapi import FastAPI, HTTPException
import gspread
import pandas as pd
import faiss
import spacy
import logging
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
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

# Load Fine-Tuned NLP Models
nlp = spacy.load("./spacy_model/model-best")  # Load fine-tuned spaCy NER model
embedder = SentenceTransformer("./fine_tuned_sentence_transformer")  # Load fine-tuned FAISS model

# Load fine-tuned T5 model
t5_tokenizer = T5Tokenizer.from_pretrained("./fine_tuned_t5")
t5_model = T5ForConditionalGeneration.from_pretrained("./fine_tuned_t5")

# Google Sheets authentication
gc = gspread.service_account(filename='service_account.json')  # Update with correct credentials

# Static Google Sheet URL
SHEET_URL = "https://docs.google.com/spreadsheets/d/1-5ao5vcVOSCyD91TUgPSPYAOOv5f_Cxt5zKDwM7mwg0/edit?usp=sharing"
USER_QUERY_SHEET_NAME = "UserQueries"

def get_user_query_sheet():
    try:
        sheet = gc.open_by_url(SHEET_URL)
        worksheet = sheet.worksheet(USER_QUERY_SHEET_NAME)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=USER_QUERY_SHEET_NAME, rows="1000", cols="2")
        worksheet.append_row(["Query", "Timestamp"])
    return worksheet

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
    df.columns = df.iloc[0]  # Use the first row as the header
    df = df[2:].reset_index(drop=True)  # Skip metadata rows
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["Issue Text", "Solution Text", "Solution Image", "Sheet Name"])
    df['processed_text'] = df['Issue Text'].apply(preprocess_text)
    return df

# Extract sheet names from queries
def extract_sheet_names(query, available_sheets):
    detected_sheets = []
    for sheet in available_sheets:
        if sheet.lower() in query.lower():
            detected_sheets.append(sheet)
    return detected_sheets

# Create FAISS index
def create_faiss_index(sentences):
    embeddings = np.array([embedder.encode(sent) for sent in sentences], dtype=np.float32)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# Search function using FAISS
def search_query(query, df, index, embeddings):
    query_embedding = embedder.encode([query]).astype(np.float32)
    _, faiss_result = index.search(query_embedding, 3)
    solutions = [(df.iloc[idx]['Solution Text'], df.iloc[idx]['Solution Image'], df.iloc[idx]['Sheet Name']) for idx in faiss_result[0] if idx < len(df)]
    return solutions

# Store user query for fine-tuning
def store_user_query(query):
    try:
        worksheet = get_user_query_sheet()
        worksheet.append_row([query, pd.Timestamp.now().isoformat()])
    except Exception as e:
        logging.error(f"Failed to store user query: {str(e)}")

@app.get("/query")
async def get_solution(query: str):
    try:
        store_user_query(query)  # Collect query for fine-tuning
        df = load_sheets_data()
        available_sheets = df['Sheet Name'].unique().tolist()
        detected_sheets = extract_sheet_names(query, available_sheets)
        index, embeddings = create_faiss_index(df['processed_text'].tolist())
        solutions = search_query(query, df, index, embeddings)
        
        if detected_sheets:
            exact_match = [sol for sol in solutions if sol[2] in detected_sheets]
            if exact_match:
                return {"message": "Here is your solution:", "solutions": [{"text": sol[0], "image": sol[1], "sheet": sol[2]} for sol in exact_match]}
            else:
                return {"message": "Sheet name is incorrect, but here are solutions to similar errors:", "solutions": [{"text": sol[0], "image": sol[1], "sheet": sol[2]} for sol in solutions]}
        
        if not solutions:
            solution_text = "No direct solution found. Please refine your query."
            return {"message": solution_text, "solutions": []}
        
        return {"message": "Sheet name not mentioned, here are solutions:", "solutions": [{"text": sol[0], "image": sol[1], "sheet": sol[2]} for sol in solutions]}
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
