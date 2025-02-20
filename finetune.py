from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import torch

# Load dataset from Google Sheets
def load_data():
    gc = gspread.service_account(filename="service_account.json")  # Add your service account JSON
    sheet = gc.open_by_url("https://docs.google.com/spreadsheets/d/1-5ao5vcVOSCyD91TUgPSPYAOOv5f_Cxt5zKDwM7mwg0/edit?usp=sharing").sheet1  # Replace with your sheet URL
    
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

# Convert data to Sentence-Transformers format
train_examples = [InputExample(texts=[row["Issue Text"], row["Solution Text"]]) for _, row in df.iterrows()]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

# Load pre-trained Sentence-Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define loss function
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100)

# Save the fine-tuned model
model.save("fine_tuned_chatbot_model")

# Load fine-tuned model
model = SentenceTransformer("fine_tuned_chatbot_model")
