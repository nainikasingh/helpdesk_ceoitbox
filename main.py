from fastapi import FastAPI, Query, Request, HTTPException, Depends
import pandas as pd
import gspread
from sentence_transformers import SentenceTransformer, InputExample, losses
import faiss
import numpy as np
from pydantic import BaseModel
from torch.utils.data import DataLoader
import torch
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import re
from fuzzywuzzy import process


# Load dataset from Google Sheets
def load_data():
    gc = gspread.service_account(filename=r"service_account.json")  # Add service account JSON
    sheet = gc.open_by_url(
        "https://docs.google.com/spreadsheets/d/1-5ao5vcVOSCyD91TUgPSPYAOOv5f_Cxt5zKDwM7mwg0/edit?usp=sharing").sheet1  # Replace with your sheet URL

    # Get all values and convert to DataFrame
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
    return df


# Load model and create embeddings
def create_faiss_index(df):
    # Load pre-trained Sentence-Transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["Issue Text"].tolist(), convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return model, index


# List of known sheet names
KNOWN_SHEET_NAMES = {
    "Auto Invoice combined", "Delegation old", "Central Report Scheduler", "Combined Greeting Sheet",
    "Newsletter Sheet", "Cyclic Sheet", "CRM old", "Advanced Message Scheduler", "Cashflow Management Sheet",
    "Google Drive Dasboard Management", "Attendence Sheet", "IMS", "Delegation Advanced", "FMS",
    "Payment Reminder Multi Temp", "Interactive List", "Auto Document Create & Send", "Interview Feedback",
    "Mobile Workflow Automation", "Lead Nurturing Combined", "Offer letter", "Google Contact Management",
    "CRM Advanced", "Recurring Reminder Sheet", "Intro Sender", "AI Email Parser", "Bulk Invoice Generator",
    "Email Response Tracker", "Payment reminder sheet", "SMS", "Auto Importer", "Recruitment Tracker",
    "Instant Response Links", "HR Automator", "Indiamart Enquiries", "Recruitment Gathering(HRMS)",
    "Recruitment Requisition (Responses)", "Central MIS Scoring Dashboard", "ChatGPT Data Analysis",
    "Drive Image Gallery", "Sheet to Ecommerce", "Web whatsapp Scheduling Sheet", "Repeat Info Capture",
    "Web Whatsapp Scheduling Sheet", "Central Archival System", "Flash sharing document sharing",
    "Sheet to HTML", "Gmail to Whatsapp", "Automated Dispatch", "Custom Catalogue Create Slides Images",
    "Appointment Letter Generator", "WA Group Management Sheet", "Sheet to App", "PMS",
    "Virtual Assistant Sheet", "Whatsapp From Any Sheet", "Interactive Photo Gallery", "PEOPLE DIRECTORY",
    "Drive Defender Sheet", "Auto Importer 2.0", "Advance Flash Sharing", "Auto Ownership change of Google Drive Files",
    "Google Storage Manager", "Advanced Email Newsletter", "Advanced Project Management System",
    "Call Recording Automation", "Online Visiting Card", "Drive Image Converter", "Central Form Data updater",
    "AI Content Generator", "Meeting Scheduler", "Calendar Task Management", "Auto Form Filler",
    "Salary slip sheet", "Lead Nurturing Business API", "Advanced Message Scheduler Business API"
}

df = load_data()
model, index = create_faiss_index(df)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


def extract_sheet_name(query):
    # Extract sheet name from query using fuzzy matching
    potential_sheets = [name.lower() for name in KNOWN_SHEET_NAMES]
    best_match, score = process.extractOne(query.lower(), potential_sheets)
    if score > 80:
        return best_match
    return None


@app.post("/query")
def search_solution(request: QueryRequest):
    query = request.query
    # Try to extract sheet name from query
    sheet_name = extract_sheet_name(query)

    if sheet_name:
        # If sheet name is detected, filter dataframe to that sheet
        filtered_df = df[df["Sheet Name"].str.lower() == sheet_name]
        if filtered_df.empty:
            return {"error": f"No issues found for the sheet: {sheet_name}"}
    else:
        # If no sheet name is detected, use the full dataframe but request clarification
        filtered_df = df

    query_embedding = model.encode([query], convert_to_numpy=True)
    _, idx = index.search(query_embedding, 1)

    sheet_name = df.iloc[idx[0][0]]['Sheet Name']
    issue_text = df.iloc[idx[0][0]]['Issue Text']

    if sheet_name.lower() != extract_sheet_name(query):
        return {
            "error": f"Your query mentions '{extract_sheet_name(query)}' but the closest issue found is in '{sheet_name}'. Please clarify the sheet name."}

    result_text = df.iloc[idx[0][0]]["Solution Text"]
    result_image = df.iloc[idx[0][0]]["Solution Image"]
    return {"solution": result_text, "solution_image": result_image, "sheet_name": sheet_name}

