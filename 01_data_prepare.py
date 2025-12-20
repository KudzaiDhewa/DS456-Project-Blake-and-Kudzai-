import os
import json

import pandas as pd
from tqdm import tqdm
from git import Repo
from unidiff import PatchSet
from collections import Counter

from dotenv import load_dotenv
from prompt_config import SYSTEM_INSTRUCTION_TEXT


# Load environment variables (for GCS/Vertex API keys)
load_dotenv()

# --- CONFIGURATION (UPDATE .env with your IDs) ---
PROJECT_ID = os.getenv("PROJECT_ID")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME") 
BASE_DIR = "./repos"
MAX_GOLD_SAMPLES_PER_REPO = 100  # Cap on manual labels per repo

REPOSITORIES = {
    "c": {
        "libxml2": "https://gitlab.gnome.org/GNOME/libxml2.git",
        "coreutils": "https://git.savannah.gnu.org/git/coreutils.git",
        "sqlite": "https://github.com/sqlite/sqlite.git",
        "openssl": "https://github.com/openssl/openssl.git",
        "libcurl": "https://github.com/curl/curl.git",
    },
    "rust": {
        "quick-xml": "https://github.com/tafia/quick-xml.git",
        "coreutils": "https://github.com/uutils/coreutils.git",
        "limbo": "https://github.com/tursodatabase/limbo.git",
        "rustls": "https://github.com/rustls/rustls.git",
        "hyper": "https://github.com/hyperium/hyper.git", 
    }
}

# Source/Dest Paths
JSONL_DIR_BASE = "full_commit_jsonl"
GOLD_DIR_BASE = "gold_standard_sample"
BLOB_BATCH_DESTINATION_PREFIX = "batch_inputs/"

# --- SANITIZATION & LOW-LEVEL UTILS ---

def sanitize_text(text):
    """
    Fixes 'surrogates not allowed' errors and ensures clean string conversion.
    """
    if isinstance(text, bytes):
        return text.decode('utf-8', errors='replace')
    if text is None:
        return ""
    # Encode/decode to replace invalid bytes with '�'
    return text.encode('utf-8', 'surrogateescape').decode('utf-8', 'replace')

def is_hunk_repetitive(lines):
    """
    Returns True if the added lines in a hunk are structurally repetitive.
    (e.g., long lists of imports, XML entities, data arrays)
    """
    added_lines = [line.value.strip() for line in lines if line.is_added]
    if len(added_lines) < 20:
        return False

    first_tokens = [l.split()[0] for l in added_lines if l]
    if not first_tokens:
        return False
        
    count = Counter(first_tokens)
    # Check if >80% of lines start with the same token
    most_common_token, frequency = count.most_common(1)[0]
    return (frequency / len(added_lines)) > 0.8


# --- SMART DIFF CLEANER (The Core Logic) ---

# Global sets for case-insensitive checking
NOISE_EXTENSIONS = {
    '.lock', '.svg', '.png', '.jpg', '.min.js', '.css', '.map', 
    '.pdf', '.o', '.a', '.so', '.exe', '.dll', '.po', '.pot',
    '.ico', '.bmp', '.gif', '.ttf', '.woff', '.woff2', '.html',
    '.htm', '.xml', '.xsa', '.sgml', '.man', '.json', '.yaml', '.yml'
}
BUILD_FILES_LOWER = {
    'makefile', 'cmakelists.txt', 'configure.in', 'cargo.toml', 
    'news', 'changelog', 'readme', 'authors', 'license', 'copying'
}


def clean_and_format_diff(raw_diff_text):
    try:
        patch = PatchSet(raw_diff_text)
    except Exception:
        return raw_diff_text[:1000] + "\n[...PARSE ERROR - TRUNCATED]"

    processed_lines = []
    
    for f in patch:
        file_path_lower = f.path.lower()
        
        # --- HEADER GENERATION ---
        status_tag = ""
        if f.is_removed_file: status_tag = " [DELETED]"
        elif f.is_added_file: status_tag = " [NEW]"
        elif f.is_rename:     status_tag = " [RENAMED]"
        
        file_header = f"File: {f.path}{status_tag}"

        # Check A: Binary/Asset Noise
        if any(file_path_lower.endswith(ext) for ext in NOISE_EXTENSIONS):
            processed_lines.append(f"{file_header} (Binary/Asset - Content Skipped)")
            continue

        # Check B: Build/Config/Docs (Case-insensitive & Doc-folder check)
        is_doc_folder = file_path_lower.startswith('doc/') or file_path_lower.startswith('docs/') or file_path_lower.startswith('man/')
        is_primary_build_file = any(name in file_path_lower for name in BUILD_FILES_LOWER)
        
        if is_primary_build_file or is_doc_folder:
            added = f.added
            removed = f.removed
            
            # SPECIAL CASE: ChangeLog & NEWS (Show content with limited accordion)
            is_log_file = ('changelog' in file_path_lower or 'news' in file_path_lower)
            
            if is_log_file:
                processed_lines.append(file_header)
                # Apply accordion to prevent 10k line change logs
                for hunk in f:
                    lines = list(hunk)
                    if len(lines) > 50:
                        for line in lines[:10]:
                            processed_lines.append(str(line).rstrip())
                        processed_lines.append(f"   ... [SNIPPED {len(lines)-20} LINES OF LOG] ...")
                        for line in lines[-10:]:
                            processed_lines.append(str(line).rstrip())
                    else:
                        for line in lines: processed_lines.append(str(line).rstrip())
            
            # STANDARD ARTIFACTS (Just skip body)
            else:
                 processed_lines.append(f"{file_header} (Build/Doc Artifact - {added} lines added, {removed} lines removed)")
                 if len(f) > 0:
                     for line in list(f[0])[:3]:
                         processed_lines.append(f"   {str(line).rstrip()}")
                     processed_lines.append("   ... [REMAINING ARTIFACT SKIPPED] ...")
            continue

        # --- CONTENT PROCESSING (Code Files: .c, .h, .rs) ---
        processed_lines.append(file_header)
        
        for hunk in f:
            lines = list(hunk)
            
            # 1. REPETITION CHECK
            if is_hunk_repetitive(lines):
                # Truncate repetitive blocks
                for line in lines[:5]: processed_lines.append(str(line).rstrip())
                processed_lines.append(f"   ... [SNIPPED {len(lines)-10} REPETITIVE LINES] ...")
                for line in lines[-5:]: processed_lines.append(str(line).rstrip())
                continue

            # 2. STANDARD ACCORDION (Cut if logic is sparse or block is huge)
            MAX_HUNK_SIZE = 50
            if len(lines) > MAX_HUNK_SIZE:
                middle_slice = lines[10:-10]
                changes_in_middle = sum(1 for line in middle_slice if not line.is_context)
                
                # Cut if logic is sparse OR block is huge (>200)
                if changes_in_middle < 5 or len(lines) > 200:
                    for line in lines[:10]: processed_lines.append(str(line).rstrip())
                    processed_lines.append(f"   ... [SNIPPED {len(lines)-20} LINES OF CONTEXT/BODY] ...")
                    for line in lines[-10:]: processed_lines.append(str(line).rstrip())
                else:
                    for line in lines: processed_lines.append(str(line).rstrip())
            else:
                for line in lines: processed_lines.append(str(line).rstrip())

    # Final Safety Truncation 
    full_text = "\n".join(processed_lines)
    if len(full_text) > 40000: 
        full_text = full_text[:40000] + "\n\n[...TRUNCATED DUE TO EXCESSIVE LENGTH...]"
        
    return full_text

def get_commit_diff_raw(commit):
    """Fetches raw diff from Git and passes to cleaner."""
    if not commit.parents:
        parent_hash = "4b825dc642cb6eb9a060e54bf8d69288fbee4904" # Empty tree
    else:
        parent_hash = commit.parents[0].hexsha

    try:
        # Get raw diff with 5 lines of context
        diff_text = commit.repo.git.diff(
            parent_hash, commit.hexsha, '-U5', '--full-index', '--text',
        )
    except Exception as e:
        return f"[Error fetching diff: {str(e)}]"

    if not diff_text.strip():
        return "No content changes (Metadata or binary only)."

    return clean_and_format_diff(diff_text)


def format_model_input(language, commit_message, diff_text):
    # Match the user prompt format used for training examples
    return f"Lang: {language} Msg: {commit_message}\nDiff:\n{diff_text}"

def create_stratified_sample(df, sample_frac=0.05, max_items=100):
    # ... (stratified sampling logic remains the same) ...
    random_sample = df.sample(frac=sample_frac, random_state=42)
    keywords = ["fix", "leak", "segfault", "feature", "refactor", "unsafe", "atomic", "race", "overflow", "panic", "unwrap"]
    pattern = '|'.join(keywords)
    
    interesting_rows = df[
        df['message'].str.contains(pattern, case=False, na=False) | 
        df['diff'].str.contains("unsafe", case=False, na=False)
    ]
    
    combined = pd.concat([random_sample, interesting_rows])
    combined = combined.drop_duplicates(subset=['commit_id'])
    
    if len(combined) > max_items:
        combined = combined.head(max_items)
        
    return combined

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    for language, repos in REPOSITORIES.items():
        lang_path = os.path.join(BASE_DIR, language)
        
        for name, url in repos.items():
            print(f"\n--- Processing {language.upper()}: {name} ---")
            
            local_path = os.path.join(lang_path, name)
            
            # Directory setup
            csv_dir = f"full_commit_with_author_data/{language}"
            jsonl_dir = f"full_commit_jsonl/{language}"
            gold_dir = f"gold_standard_sample/{language}"

            os.makedirs(csv_dir, exist_ok=True)
            os.makedirs(jsonl_dir, exist_ok=True)
            os.makedirs(gold_dir, exist_ok=True)

            full_commit_set = f"{csv_dir}/full_commit_{name}.csv"
            BATCH_INPUT_JSONL = f"{jsonl_dir}/BATCH_input_{name}.jsonl"
            GOLD_STANDARD_CSV = f"{gold_dir}/commits_to_label_{name}.csv"

            # 1. CLONE
            if not os.path.exists(local_path):
                print(f"Cloning {url}...")
                Repo.clone_from(url, local_path)

            # 2. EXTRACT (or Load Cache)
            if os.path.exists(full_commit_set):
                print(f"Loading cached CSV: {full_commit_set}")
                commits_df = pd.read_csv(full_commit_set)
                commits_df['diff'] = commits_df['diff'].fillna("")
                commits_df['message'] = commits_df['message'].fillna("")
            else:
                print("Extracting commits & stats (this might take a while)...")
                try:
                    repo = Repo(local_path)
                    diff_data = []
                    
                    for commit in tqdm(list(repo.iter_commits())):
                        stats = commit.stats.total
                        entropy = stats.get('files', 0)
                        churn = stats.get('insertions', 0) + stats.get('deletions', 0)

                        safe_message = sanitize_text(commit.message.strip())
                        raw_diff = get_commit_diff_raw(commit) # Calls the new cleaner
                        safe_diff = sanitize_text(raw_diff)
                        safe_author = sanitize_text(commit.author.name)

                        diff_data.append({
                            "commit_id": commit.hexsha,
                            "message": safe_message,
                            "diff": safe_diff, 
                            "author": safe_author,
                            "date": commit.authored_datetime,
                            "entropy": entropy,
                            "churn": churn
                        })
                    
                    commits_df = pd.DataFrame(diff_data)
                    commits_df.to_csv(full_commit_set, index=False)
                except Exception as e:
                    print(f"[ERROR] Failed extracting {name}: {e}")
                    continue

            # 3. CREATE BATCH INPUT
            print(f"Creating Batch Input: {BATCH_INPUT_JSONL}")
            with open(BATCH_INPUT_JSONL, 'w', encoding='utf-8') as f:
                for _, row in commits_df.iterrows():
                    
                    unique_key = f"{language}_{name}_{row['commit_id']}"
                    
                    # Diff is already cleaned in Step 2, so we use row['diff']
                    user_content = format_model_input(language, row['message'], row['diff'])
                    
                    # Vertex AI Batch Payload
                    req = {
                        "request": {
                            "systemInstruction": {
                                "role": "system",
                                "parts": [{"text": SYSTEM_INSTRUCTION_TEXT}],
                            },
                            "contents": [
                                {
                                    "role": "user",
                                    "parts": [{"text": user_content}],
                                }
                            ],
                        },
                        "key": unique_key  # Pass-through identifier
                    }
                    f.write(json.dumps(req) + "\n")

            # 4. CREATE GOLD SAMPLE
            '''
            print(f"Creating sample for labeling: {GOLD_STANDARD_CSV}")
            gold_sample = create_stratified_sample(commits_df, max_items=MAX_GOLD_SAMPLES_PER_REPO)
            
            # Setup columns for manual entry
            cols_to_add = ['label_category', 'label_is_security', 'label_is_feature', 'label_complexity', 'label_reasoning']
            for col in cols_to_add:
                gold_sample[col] = ""
            
            gold_sample['diff_preview'] = gold_sample['diff']
            
            final_cols = ['commit_id', 'message', 'entropy', 'churn'] + cols_to_add + ['diff_preview']
            
            gold_sample[final_cols].to_csv(GOLD_STANDARD_CSV, index=False)
            '''
            
            print(f"Done with {name}.")
