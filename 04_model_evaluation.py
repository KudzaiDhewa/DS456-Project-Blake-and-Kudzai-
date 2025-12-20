import os

from google import genai
from google.genai import types
from google.cloud import storage
from dotenv import load_dotenv


load_dotenv()

# --- Configuration ---
PROJECT_ID = os.getenv("PROJECT_ID")  # Make SURE this is the right one!
REGION = os.getenv("REGION")
CURRENT_LANGUAGE_REPO = os.getenv("CURRENT_LANGUAGE_REPO")

# File where 03_model_training.py stored the tuned model resource name
TUNED_MODEL_ID = f"FINETUNED_RESOURCENAME/{CURRENT_LANGUAGE_REPO}"

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# Input JSONL in GCS for batch classification (env var)
BLOB_BATCHING_TO_CLASSIFY_DESTINATION = "batch_to_classify"
FULL_COMMIT_DATA_RETRIEVAL_URI = f"gs://{GCS_BUCKET_NAME}/{BLOB_BATCHING_TO_CLASSIFY_DESTINATION}"

# Output prefix in GCS for batch results (env var)
FULL_COMMIT_DATA_RESULTS = os.getenv("BLOB_BATCHING_RESULTS")
GCS_OUTPUT_URI_PREFIX = f"gs://{GCS_BUCKET_NAME}/{FULL_COMMIT_DATA_RESULTS}"

# Local folder with per-repo JSONLs from phase 01
LOCAL_JSONL_ROOT = "full_commit_jsonl"

# Local merged JSONL that will be uploaded for batch prediction
LOCAL_MERGED_JSONL = os.path.join(LOCAL_JSONL_ROOT, "full_commit_to_classify_merged.jsonl")


def merge_local_jsonls() -> str:
    """
    Merge all BATCH_input_*.jsonl files under full_commit_jsonl into a single JSONL.
    Returns the path to the merged local file.
    """
    jsonl_paths = []
    for root, _, files in os.walk(LOCAL_JSONL_ROOT):
        for fname in files:
            if fname.startswith("BATCH_input") and fname.endswith(".jsonl"):
                jsonl_paths.append(os.path.join(root, fname))

    if not jsonl_paths:
        raise RuntimeError(f"No BATCH_input*.jsonl files found under {LOCAL_JSONL_ROOT}")

    jsonl_paths.sort()

    print(f"Merging {len(jsonl_paths)} JSONL files into {LOCAL_MERGED_JSONL}")
    os.makedirs(os.path.dirname(LOCAL_MERGED_JSONL), exist_ok=True)

    with open(LOCAL_MERGED_JSONL, "w", encoding="utf-8") as out_f:
        for path in jsonl_paths:
            print(f"  - Adding {path}")
            with open(path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if line.strip():
                        if line.endswith("\n"):
                            out_f.write(line)
                        else:
                            out_f.write(line + "\n")

    return LOCAL_MERGED_JSONL


def upload_merged_jsonl_to_gcs(local_path: str) -> None:
    """
    Upload the merged JSONL to the GCS location used by the batch job.
    """
    if not BLOB_BATCHING_TO_CLASSIFY_DESTINATION:
        raise RuntimeError("BLOB_BATCHING_TO_CLASSIFY_DESTINATION is not set in .env")

    print(f"Uploading {local_path} to {FULL_COMMIT_DATA_RETRIEVAL_URI}")
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(BLOB_BATCHING_TO_CLASSIFY_DESTINATION)
    blob.upload_from_filename(local_path)
    print(f"✅ Uploaded merged JSONL to {FULL_COMMIT_DATA_RETRIEVAL_URI}")


def launch_batch_prediction_job():
    """
    Merge local JSONLs, upload the merged file to GCS, and launch
    an asynchronous batch prediction job using the fine-tuned Gemini model.
    """
    # 1. Merge local JSONL files
    #merged_path = merge_local_jsonls()

    # 2. Upload merged JSONL to GCS
    #upload_merged_jsonl_to_gcs(merged_path)

    # 3. Initialize Vertex AI GenAI client
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=REGION,  # IMPORTANT: SFT Batch MUST be regional. Global often fails.
    )

    # 4. Read the model resource name from the file saved by 03_model_training.py
    try:
        with open(TUNED_MODEL_ID, "r") as f:
            model_resource_name = f.read().strip()
            print(f"Read model resource name from file: {model_resource_name}")
    except FileNotFoundError:
        print(f"Error: Model resource name file not found at {TUNED_MODEL_ID}")
        print("Make sure you've run 03_model_training.py first to create the model.")
        return

    # 5. Launch batch job via GenAI batches API
    try:
        batch = client.batches.create(
            model=model_resource_name,
            src=FULL_COMMIT_DATA_RETRIEVAL_URI,
            config=types.CreateBatchJobConfig(dest=GCS_OUTPUT_URI_PREFIX),
        )

        print("\n✅ Batch prediction job launched successfully!")
        print(f"Job name: {batch.name}")
        print(f"Job state: {batch.state}")
        print("\nMonitor progress in the Vertex AI console:")
        print(
            f"https://console.cloud.google.com/vertex-ai/predictions/batch-prediction-jobs?project={PROJECT_ID}"
        )
    except Exception as e:
        print(f"\n❌ Error launching batch prediction job: {e}")
        print("\nTroubleshooting:")
        print(f"1. Verify the model resource name is correct: {model_resource_name}")
        print(f"2. Check that the input file exists: {FULL_COMMIT_DATA_RETRIEVAL_URI}")
        print("3. Ensure your service account has necessary permissions")
        print("4. Verify the model supports batch predictions")
        raise


if __name__ == "__main__":
    launch_batch_prediction_job()

    #merged_path = merge_local_jsonls()
    
    # 2. Upload merged JSONL to GCS
    #upload_merged_jsonl_to_gcs(merged_path)

