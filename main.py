from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import RedirectResponse, Response
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import base64
from utils.secrets import JWT_SECRET
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
from fuzzywuzzy import process, fuzz
import uvicorn
import os
import boto3
from concurrent.futures import ThreadPoolExecutor
from cachetools import TTLCache

# AWS S3 Bucket
S3_BUCKET_NAME = "helpdesk-ai-nainika-ap-south1"
S3_MODELS = {
    "sentence_transformer": "fine_tuned_sentence_transformer/",
    "t5": "fine_tuned_t5/",
    "spacy_best": "spacy_model/model-best/",
    "spacy_last": "spacy_model/model-last/"
}
S3_CSV_PATH = "Helpdesk_Issues_Database_Glossary.csv"

# Local Model Directory
MODEL_DIR = "/home/ubuntu/models"

# Cache for CSV Data
cache = TTLCache(maxsize=1, ttl=600)  # Cache for 10 minutes

# Initialize AWS S3 client
s3_client = boto3.client("s3")

# Function to download models from S3
def download_model_from_s3(model_key, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=model_key):
        if "Contents" in page:
            for obj in page["Contents"]:
                file_key = obj["Key"]
                relative_path = file_key[len(model_key):].lstrip("/")
                dest_path = os.path.join(destination_folder, relative_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                s3_client.download_file(S3_BUCKET_NAME, file_key, dest_path)
                print(f"Downloaded {file_key} to {dest_path}")


# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize FastAPI app
app = FastAPI()

# Define security scheme (Basic Auth)
security = HTTPBasic()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to enforce authentication for Swagger UI & OpenAPI
async def enforce_docs_auth(request: Request):
    if request.url.path in ["/docs", "/openapi.json"]:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Basic "):
            return Response(
                headers={"WWW-Authenticate": "Basic"},
                status_code=401,
                content="Unauthorized: Missing authentication"
            )

        # Decode the Base64-encoded credentials
        encoded_credentials = auth_header.split("Basic ")[1]
        decoded_credentials = base64.b64decode(encoded_credentials).decode("utf-8")
        username, password = decoded_credentials.split(":", 1)

        # Validate credentials
        correct_username = "Admin"
        correct_password = JWT_SECRET  # Ensure JWT_SECRET is set correctly

        if username != correct_username or password != correct_password:
            return Response(
                headers={"WWW-Authenticate": "Basic"},
                status_code=401,
                content="Unauthorized: Incorrect credentials"
            )

# Add authentication middleware for Swagger UI & OpenAPI
@app.middleware("http")
async def docs_auth_middleware(request: Request, call_next):
    response = await enforce_docs_auth(request)
    if response:
        return response  # Return 401 if unauthorized

    return await call_next(request)

# Secure the Swagger UI
@app.get("/docs", include_in_schema=False)
async def get_documentation():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="API Docs")

# Secure the OpenAPI schema (optional)
@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    return get_openapi(title="API", version="1.0.0", routes=app.routes)

# Load Models
nlp = spacy.load(os.path.join(MODEL_DIR, "spacy_best"))
embedder = SentenceTransformer(os.path.join(MODEL_DIR, "sentence_transformer"))
t5_tokenizer = T5Tokenizer.from_pretrained(os.path.join(MODEL_DIR, "t5"))
t5_model = T5ForConditionalGeneration.from_pretrained(os.path.join(MODEL_DIR, "t5"))

# Load CSV data from S3
def load_csv_data():
    if "csv_data" in cache:
        return cache["csv_data"]
    
    s3_object = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=S3_CSV_PATH)
    df = pd.read_csv(s3_object["Body"])
    df = df.iloc[1:].reset_index(drop=True)  # Skip only the first row if needed
    df.columns = df.columns.str.strip()  # Remove any unwanted spaces
    df = df.dropna(subset=["Issue Text", "Solution Text", "Solution Image", "Sheet Name"])
    df['processed_text'] = df['Issue Text'].apply(preprocess_text)
    
    cache["csv_data"] = df
    return df

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Define `create_faiss_index` BEFORE calling it
def create_faiss_index(sentences):
    embeddings = np.array([embedder.encode(sent) for sent in sentences], dtype=np.float32)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# Load dataset and create FAISS index after defining the function
df = load_csv_data()
if df is None:
    raise ValueError("Dataframe loading failed. Check CSV file or S3 access.")

print("CSV Data Loaded. Number of records:", len(df))

FAISS_INDEX_PATH = os.path.join(MODEL_DIR, "faiss_index.index")
EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "embeddings.npy")

if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(EMBEDDINGS_PATH):
    print("📦 Loading FAISS index from disk...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
else:
    print("⚙️ Rebuilding FAISS index...")
    index, embeddings = create_faiss_index(df["processed_text"].tolist())

print("FAISS Index Created. Number of vectors:", len(embeddings))

# Search function using FAISS
def search_query(query, df, index, embeddings, top_k=6):
    query_embedding = embedder.encode([query]).astype(np.float32)
    _, faiss_result = index.search(query_embedding, top_k)

    results = []
    for idx in faiss_result[0]:
        if idx < len(df):
            results.append((df.iloc[idx]['Solution Text'], df.iloc[idx]['Solution Image'], df.iloc[idx]['Sheet Name']))

    logging.info(f"🔎 Top {top_k} FAISS Matches: {results}")

    return results


# Store user query asynchronously
def store_query_in_background(query: str):
    try:
        logging.info(f"Storing user query: {query}")
    except Exception as e:
        logging.error(f"Failed to store user query: {str(e)}")

# Extract relevant sheet names from query
def extract_sheet_names(query, available_sheets):
    return [sheet for sheet in available_sheets if sheet.lower() in query.lower()]

# Function to check error message occurrence in sheets
def check_error_in_sheets(sheets_data):
    #error_sheets = []
    #for sheet_name, data in sheets_data.items():
     #   if "query" in data:
      #      error_sheets.append(sheet_name)
    
    #if len(error_sheets) > 3:
     #   logging.info(f"Message appears in multiple sheets: {error_sheets}")
      #  return {
       #     "message": f"The error appears in {len(error_sheets)} sheets. It might be a common issue.",
        #    "affected_sheets": error_sheets
        #}
    #return {"sheets": [], "message": "No widespread issue detected"}
    best_match, score = process.extractOne(query, available_sheets)
    return best_match if score > 80 else None

def clean_text(text):
    """Lowercase, remove special characters, and strip spaces"""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text).strip().lower()

def fuzzy_extract_sheet_names(query: str, available_sheets: list, threshold: int = 60):
    """Extract relevant sheet names from query using fuzzy matching after normalizing text."""
    
    # Clean query and sheet names
    query_clean = clean_text(query)
    sheets_clean = [clean_text(sheet) for sheet in available_sheets]

    matches = process.extract(query_clean, sheets_clean, scorer=fuzz.partial_ratio, limit=10)

    logging.info(f"✅ Raw Query: {query}")
    logging.info(f"✅ Cleaned Query: {query_clean}")
    logging.info(f"✅ Top 10 Matches: {matches}")

    # Fuzzy match
    matches = process.extract(query_clean, sheets_clean, scorer=fuzz.partial_ratio, limit=5)
    
    # Debugging: Print top matches
    logging.info(f"🟢 Top Matches: {matches}")

    # Extract sheet names where score is above the threshold
    detected_sheets = [available_sheets[i] for i, (match, score) in enumerate(matches) if score >= threshold]
    
    # Debugging: Print detected sheets
    logging.info(f"✅ Final Detected Sheets (Threshold {threshold}+): {detected_sheets}")
    
    return detected_sheets

def normalize_text_for_match(text: str) -> str:
    """Lowercase, remove special characters, deduplicate, sort tokens."""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower().strip()
    tokens = list(set(text.split()))  # Remove duplicates
    tokens.sort()
    return ' '.join(tokens)

def is_query_only_sheet_name(query: str, sheet_names: list, threshold: int = 90) -> bool:
    """
    Determines if the query is basically just a fuzzy variant of a sheet name.
    """
    query_norm = normalize_text_for_match(query)

    for sheet in sheet_names:
        sheet_norm = normalize_text_for_match(sheet)
        score = fuzz.token_set_ratio(query_norm, sheet_norm)  # Better for unordered text
        if score >= threshold:
            return True
    return False

@app.get("/query")
async def get_solution(query: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(store_query_in_background, query)
    df = load_csv_data()
    available_sheets = df['Sheet Name'].unique().tolist()

    # ✅ NEW: Check if query is just a sheet name
    if is_query_only_sheet_name(query, available_sheets):
        return {"message": "Please enter more details."}

    detected_sheets = fuzzy_extract_sheet_names(query, available_sheets)
    solutions = search_query(query, df, index, embeddings)
    matching_sheets = list(set(sol[2] for sol in solutions))
    
    print(f"Query: {query}")
    print(f"Available Sheets: {available_sheets}")
    print(f"Detected Sheets: {detected_sheets}")
    print(f"Solutions Found: {solutions}")
    
      # **1. Exact Match: If query has a sheet name and matches exactly**
    if detected_sheets:
        exact_match = [sol for sol in solutions if sol[2] in detected_sheets]
        if exact_match:
            return {
                "message": "Here is your solution:",
                "solutions": [{"text": sol[0], "image": sol[1], "sheet": sol[2]} for sol in exact_match]
            }
        else:
            # **Sheet detected but incorrect → Suggest top 3 solutions + matching sheet names**
            return {
                "message": "Sheet name seems incorrect, here are the top 3 suggestions:",
                "solutions": [{"text": sol[0], "image": sol[1], "sheet": sol[2]} for sol in solutions[:3]],
                "other matching_sheets": detected_sheets  # Suggest the closest matching sheets
            }

    # **2. No sheet name in query but matches multiple sheets**
    matching_sheets = list(set([sol[2] for sol in solutions]))  # Unique sheets

    if len(matching_sheets) > 3:
        return {
            "message": "Please mention sheet name with query",
            "sheets": matching_sheets
        }

    # **3. Query matches ≤ 3 sheets → Provide solutions**
    if 1 <= len(matching_sheets) <= 3:
        return {
            "message": "Sheet name not mentioned, here are solutions:",
            "solutions": [{"text": sol[0], "image": sol[1], "sheet": sol[2]} for sol in solutions if sol[2] in matching_sheets]
        }

    # **4. No match found**
    return {"message": "No direct solution found. Please refine your query."}

@app.get("/")
def read_root():
    return {"message": "Hello from AWS!"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting FastAPI on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
