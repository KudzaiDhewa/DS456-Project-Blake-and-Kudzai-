import os
import json
import glob
import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage
from prompt_config import SYSTEM_INSTRUCTION_TEXT

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
PROJECT_ID = os.getenv("PROJECT_ID")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME") 

# Destination Paths in Bucket
BLOB_TRAINING_DESTINATION = "training_data/final_mixed_model.jsonl"
BLOB_BATCH_DESTINATION_PREFIX = "batch_inputs/"

# Path to your labeled data (merged gold standard from Phase 01)
GOLD_MERGED_PATH = "gold_standard_sample_labeled/labeled_gold_standard_merged.csv"
JSONL_DIR_BASE = "full_commit_jsonl"

# Robust Mapping for Labels
CATEGORY_MAP = {
    "Memory": "Memory Safety & Robustness",
    "Memory Safety": "Memory Safety & Robustness",
    "Concurrency": "Concurrency & Thread Safety",
    "Thread Safety": "Concurrency & Thread Safety",
    "Logic": "Logic & Correctness",
    "Logic Error": "Logic & Correctness",
    "Refactor": "Build, Refactor & Internal",
    "Build": "Build, Refactor & Internal",
    "Maintenance": "Build, Refactor & Internal",
    "Feature": "Feature & Value Add",
    "Feat": "Feature & Value Add"
}

def clean_text(text):
    """Handles Excel encoding artifacts and line endings."""
    if not isinstance(text, str): return ""
    return text.replace('\r\n', '\n').strip()

def clean_reasoning(text):
    """Truncates reasoning to max 15 words."""
    text = clean_text(str(text))
    text = text.replace("\n", " ")
    words = text.split()
    if len(words) > 15:
        return " ".join(words[:15]) + "..."
    return text

def parse_bool(value):
    """Handles Excel's TRUE/False/1/0 variations."""
    s = str(value).upper().strip()
    return s in ['TRUE', '1', 'T', 'YES', 'Y']

def normalize_category(raw_cat):
    """Ensures category matches the valid list exactly."""
    raw_cat = str(raw_cat).strip()
    for key, val in CATEGORY_MAP.items():
        if raw_cat.lower() == key.lower():
            return val
    if raw_cat in CATEGORY_MAP.values():
        return raw_cat
    print(f"[WARNING] Unknown category '{raw_cat}'. Defaulting to 'Build, Refactor & Internal'")
    return "Build, Refactor & Internal"

def format_training_prompt(language, commit_message, cleaned_diff):
    return f"Lang: {language} Msg: {commit_message}\nDiff:\n{cleaned_diff}"

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"✅ Uploaded: gs://{bucket_name}/{destination_blob_name}")
    except Exception as e:
        print(f"❌ Error uploading {destination_blob_name}: {e}")

def create_fine_tuning_jsonl(merged_df, output_filename):
    print(f"Generating {output_filename}...")
    with open(output_filename, 'w', encoding='utf-8') as f:
        for index, row in merged_df.iterrows():
            
            # 1. Skip rows without a category label
            if pd.isna(row['label_category']) or str(row['label_category']).strip() == "":
                continue

            # 2. Prepare User Input
            lang = row['language_inferred']
            diff_text = clean_text(row['diff_preview'])
            user_text = format_training_prompt(
                lang,
                clean_text(row['message']), 
                diff_text
            )

            # 3. Prepare Model Output
            try:
                comp_score = int(float(row['label_complexity']))
            except(ValueError, TypeError):
                comp_score = 1 

            model_response = {
                "cat": normalize_category(row['label_category']),
                "feat": parse_bool(row['label_is_feature']),
                "sec": parse_bool(row['label_is_security']),
                "comp": comp_score,
                "reas": clean_reasoning(row['label_reasoning'])
            }

            # 4. Write Entry (Correct Vertex AI Gemini Format)
            entry = {
                "systemInstruction": {
                    "role": "system", 
                    "parts": [{"text": SYSTEM_INSTRUCTION_TEXT}]
                },
                "contents": [
                    {
                        "role": "user", 
                        "parts": [{"text": user_text}]
                    },
                    {
                        "role": "model", 
                        "parts": [{"text": json.dumps(model_response)}]
                    }
                ]
            }
            f.write(json.dumps(entry) + "\n")
            
    print(f"Refined training data saved locally to: {output_filename}")

if __name__ == "__main__":
    if not PROJECT_ID or not GCS_BUCKET_NAME:
        print("❌ Error: Missing .env variables (PROJECT_ID or GCS_BUCKET_NAME)")
        exit(1)

    # --- 1. LOAD MERGED LABELED DATA ---
    if not os.path.exists(GOLD_MERGED_PATH):
        print(f"❌ ERROR: Merged labeled CSV not found at {GOLD_MERGED_PATH}")
        print("Run 09_merge_labeled_gold_standard.py first.")
        exit(1)

    print(f"Loading merged labeled data from {GOLD_MERGED_PATH}...")
    full_training_df = pd.read_csv(GOLD_MERGED_PATH, dtype=str).fillna("")
    print(f"Total labeled rows found: {len(full_training_df)}")

    TRAINING_FILE_LOCAL = "final_mixed_training_data.jsonl"
    create_fine_tuning_jsonl(full_training_df, TRAINING_FILE_LOCAL)

    # --- 3. UPLOAD TRAINING DATA ---
    #upload_to_gcs(GCS_BUCKET_NAME, TRAINING_FILE_LOCAL, BLOB_TRAINING_DESTINATION)

    # --- 4. UPLOAD BATCH INPUTS ---
    '''
    print("\nUploading Inference Batches...")
    found_batches = False
    for root, dirs, files in os.walk(JSONL_DIR_BASE):
        for file in files:
            if file.startswith("BATCH_input") and file.endswith(".jsonl"):
                found_batches = True
                local_path = os.path.join(root, file)
                subdir = os.path.basename(root)
                if subdir not in ['c', 'rust']:
                    blob_path = f"{BLOB_BATCH_DESTINATION_PREFIX}{file}"
                else:
                    blob_path = f"{BLOB_BATCH_DESTINATION_PREFIX}{subdir}/{file}"
                upload_to_gcs(GCS_BUCKET_NAME, local_path, blob_path)
    '''
    if not found_batches:
        print("⚠️ WARNING: No BATCH_input files found.")
    
    print("\n--- PHASE 02 COMPLETE ---")
    print(f"Training Data: gs://{GCS_BUCKET_NAME}/{BLOB_TRAINING_DESTINATION}")
