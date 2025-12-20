import hashlib
from pathlib import Path

import pandas as pd


# How many final rows in each CSV should get synthetic IDs
NUM_FINAL_ROWS = 20

# CSVs to modify
TARGET_FILES = [
    Path("gold_standard_sample_labeled/c/labeled_gold_standard_libxml2.csv"),
    Path("gold_standard_sample_labeled/rust/labeled_gold_standard_quick-xml.csv"),
]


def make_synthetic_id(commit_id: str, salt: str) -> str:
    """
    Deterministically generate a synthetic 40-char hex ID from a real commit_id.
    This keeps the ID git-like but guarantees it's synthetic and reproducible.
    """
    raw = f"synth:{salt}:{commit_id}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()


def process_file(path: Path) -> None:
    print(f"\nProcessing {path}")
    df = pd.read_csv(path)

    if df.empty:
        print("  - File is empty, skipping.")
        return

    # Ensure helper flag column exists
    if "is_synthetic_commit_id" not in df.columns:
        df["is_synthetic_commit_id"] = False

    tail = df.tail(NUM_FINAL_ROWS)
    indices = tail.index

    salt = path.stem  # e.g., labeled_gold_standard_libxml2

    print(f"  - Ensuring synthetic IDs on last {len(indices)} rows")

    for idx in indices:
        real_id = str(df.at[idx, "commit_id"])
        synth_id = make_synthetic_id(real_id, salt)

        # Overwrite commit_id with the synthetic value so downstream
        # pipelines only see the synthetic ID.
        df.at[idx, "commit_id"] = synth_id
        df.at[idx, "is_synthetic_commit_id"] = True

    # If an old synthetic_commit_id column exists from earlier runs, drop it
    if "synthetic_commit_id" in df.columns:
        df = df.drop(columns=["synthetic_commit_id"])

    # Write back, keeping CSV shape consistent
    backup_path = path.with_suffix(path.suffix + ".bak")
    df.to_csv(backup_path, index=False)
    df.to_csv(path, index=False)

    print(f"  - Wrote updated CSV and backup at {backup_path}")


def main() -> None:
    for csv_path in TARGET_FILES:
        if not csv_path.exists():
            print(f"Skipping missing file: {csv_path}")
            continue
        process_file(csv_path)


if __name__ == "__main__":
    main()
