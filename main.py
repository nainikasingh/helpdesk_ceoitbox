from fastapi import FastAPI, HTTPException, BackgroundTasks
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
import uvicorn
import os
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor
from cachetools import TTLCache
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import json
import base64

# Google Cloud Storage Bucket Name
BUCKET_NAME = "helpdesk-chatbot-main-bucket"
PROJECT_ID = "helpdesk-451910"

# Cache for Google Sheets Data
cache = TTLCache(maxsize=1, ttl=600)  # Cache for 10 minutes

# Function to download models from Google Cloud Storage
def download_model_from_gcs(bucket_name, source_folder, destination_folder):
    try:
        service_account_path = download_service_account()
        storage_client = storage.Client.from_service_account_json(service_account_path, project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=source_folder)

        os.makedirs(destination_folder, exist_ok=True)
        for blob in blobs:
            relative_path = blob.name[len(source_folder):].lstrip("/")
            dest_path = os.path.join(destination_folder, relative_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            blob.download_to_filename(dest_path)
            print(f"Downloaded {blob.name} to {dest_path}")
    except Exception as e:
        logging.error(f"Error accessing GCS: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to access Google Cloud Storage")

# Authenticate Google Cloud service account
def download_service_account():
    service_account_b64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64")
    service_account_path = "/tmp/service_account.json"
    if service_account_b64:
        service_account_json = base64.b64decode(service_account_b64).decode("utf-8")
        with open(service_account_path, "w") as f:
            f.write(service_account_json)
        return service_account_path
    
    raise FileNotFoundError("GOOGLE_SERVICE_ACCOUNT_JSON_B64 environment variable not found. Make sure it is set correctly.")

try:
    service_account_path = download_service_account()
    print(f"Service account file saved at: {service_account_path}")
    if not os.path.exists(service_account_path):
        raise FileNotFoundError("Service account file not found after writing to /tmp/")
    
    gc = gspread.service_account(filename=service_account_path)
except FileNotFoundError as e:
    logging.error(str(e))
    gc = None

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize FastAPI app
app = FastAPI()

# Parallel Model Download
def parallel_download():
    with ThreadPoolExecutor() as executor:
        executor.submit(download_model_from_gcs, BUCKET_NAME, "spacy_model/model-best", "/tmp/spacy_model")
        executor.submit(download_model_from_gcs, BUCKET_NAME, "fine_tuned_sentence_transformer", "/tmp/fine_tuned_sentence_transformer")
        executor.submit(download_model_from_gcs, BUCKET_NAME, "fine_tuned_t5", "/tmp/fine_tuned_t5")

parallel_download()

# Load Models
nlp = spacy.load("/tmp/spacy_model")
embedder = SentenceTransformer("/tmp/fine_tuned_sentence_transformer")
t5_tokenizer = T5Tokenizer.from_pretrained("/tmp/fine_tuned_t5")
t5_model = T5ForConditionalGeneration.from_pretrained("/tmp/fine_tuned_t5")

# Static Google Sheet URL
SHEET_URL = "https://docs.google.com/spreadsheets/d/1-5ao5vcVOSCyD91TUgPSPYAOOv5f_Cxt5zKDwM7mwg0/edit?usp=sharing"
USER_QUERY_SHEET_NAME = "UserQueries"

# Google Sheets Handling
def get_user_query_sheet():
    if not gc:
        raise HTTPException(status_code=500, detail="Google Sheets authentication failed.")
    
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

# Load Google Sheet data with caching
def load_sheets_data():
    if "sheet_data" in cache:
        return cache["sheet_data"]

    if not gc:
        raise HTTPException(status_code=500, detail="Google Sheets authentication failed.")
    
    sheet = gc.open_by_url(SHEET_URL).sheet1
    data = sheet.get_all_values()
    df = pd.DataFrame(data)
    df.columns = df.iloc[0]
    df = df[2:].reset_index(drop=True)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["Issue Text", "Solution Text", "Solution Image", "Sheet Name"])
    df['processed_text'] = df['Issue Text'].apply(preprocess_text)

    cache["sheet_data"] = df
    return df

# Load sheets data once and create FAISS index
df = load_sheets_data()
index, embeddings = create_faiss_index(df['processed_text'].tolist())

# Extract sheet names from queries
def extract_sheet_names(query, available_sheets):
    return [sheet for sheet in available_sheets if sheet.lower() in query.lower()]

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
    return [(df.iloc[idx]['Solution Text'], df.iloc[idx]['Solution Image'], df.iloc[idx]['Sheet Name']) for idx in faiss_result[0] if idx < len(df)]

# Store user query asynchronously
def store_query_in_background(query: str):
    try:
        worksheet = get_user_query_sheet()
        worksheet.append_row([query, pd.Timestamp.now().isoformat()])
    except Exception as e:
        logging.error(f"Failed to store user query: {str(e)}")

@app.get("/query")
async def get_solution(query: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(store_query_in_background, query)
    df = load_sheets_data()
    available_sheets = df['Sheet Name'].unique().tolist()
    detected_sheets = extract_sheet_names(query, available_sheets)
    solutions = search_query(query, df, index, embeddings)

    if detected_sheets:
        exact_match = [sol for sol in solutions if sol[2] in detected_sheets]
        if exact_match:
            return {"message": "Here is your solution:", "solutions": [{"text": sol[0], "image": sol[1], "sheet": sol[2]} for sol in exact_match]}
        else:
            return {"message": "Sheet name is incorrect, but here are solutions to similar errors:", "solutions": [{"text": sol[0], "image": sol[1], "sheet": sol[2]} for sol in solutions]}

    if not solutions:
        return {"message": "No direct solution found. Please refine your query.", "solutions": []}

    return {"message": "Sheet name not mentioned, here are solutions:", "solutions": [{"text": sol[0], "image": sol[1], "sheet": sol[2]} for sol in solutions]}


@app.get("/")
def read_root():
    return {"message": "Hello from Cloud Run!"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
