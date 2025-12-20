import glob
import json
import os
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math


# --- CONFIGURATION ---
INPUT_DIR = "CLASSIFED_FULL_JSONL" 
OUTPUT_FILE = "FINAL_PREDICTION_DATASET_clean.csv"
CSS_PREDICTION = "FINAL_CSS_WITH_PREDICTION.csv"

META_DATA = "full_commit_with_author_data/"
# The Canonical 5
VALID_CATEGORIES = [
    "Memory Safety & Robustness",
    "Concurrency & Thread Safety",
    "Logic & Correctness",
    "Build, Refactor & Internal",
    "Feature & Value Add",
]



# --- CONFIGURATION FOR CCS ---
CSV_SOURCE_DIR = "full_commit_with_author_data"
W_COG = 1.0   # Weight for Cognitive Load (LLM)
W_ENT = 0.5   # Weight for Entropy (Files)
W_CHURN = 0.5 # Weight for Churn (Lines)


DOMAIN_MAP = {
    "libxml2": "XML Parsing",
    "quick-xml": "XML Parsing",
    "libcurl": "HTTP/Net",
    "hyper": "HTTP/Net",
    "openssl": "Crypto/TLS",
    "rustls": "Crypto/TLS",
    "sqlite": "Database",
    "limbo": "Database",
    "coreutils": "Utils",  # Both C and Rust repos are named 'coreutils'
}


def plot_diverging_delta(data: pd.DataFrame) -> None:
    """
    Diverging bar plot of C vs Rust focus by domain and category,
    faceted into Feature vs Maintenance (like an R facet).
    """
    df = data.copy()

    # Map domains
    df["domain"] = df["repo"].map(DOMAIN_MAP)

    # Group Feature vs Maintenance
    df["work_type"] = df["category"].apply(
        lambda c: "Feature" if c == "Feature & Value Add" else "Maintenance"
    )

    # Normalized share per repo / language / category
    counts = (
        df.groupby(["language", "repo", "domain", "category", "work_type"])
        .size()
        .reset_index(name="count")
    )
    totals = (
        df.groupby(["language", "repo"])
        .size()
        .reset_index(name="total")
    )

    merged = pd.merge(counts, totals, on=["language", "repo"])
    merged["share"] = merged["count"] / merged["total"]

    # Pivot to get C and Rust side by side
    pivot_df = (
        merged.pivot_table(
            index=["domain", "category", "work_type"],
            columns="language",
            values="share",
            fill_value=0,
        )
        .reset_index()
    )

    # Delta: C share minus Rust share
    if "c" not in pivot_df.columns or "rust" not in pivot_df.columns:
        print("Expected 'c' and 'rust' language columns not found; skipping plot.")
        return
    pivot_df["delta"] = pivot_df["c"] - pivot_df["rust"]

    # Palette consistent with other plots
    palette = {
        "Memory Safety & Robustness": "#8870ad",  # Purple
        "Concurrency & Thread Safety": "#cc8e60",  # Orange
        "Logic & Correctness": "#c44e52",  # Red
        "Build, Refactor & Internal": "#4c72b0",  # Blue
        "Feature & Value Add": "#55a868",  # Green
    }

    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=pivot_df,
        x="domain",
        y="delta",
        hue="category",
        col="work_type",
        kind="bar",
        palette=palette,
        edgecolor="black",
        linewidth=0.5,
        height=6,
        aspect=1.3,
    )

    # Formatting
    for ax in g.axes.flatten():
        ax.axhline(0, color="black", linewidth=1)
        vals = ax.get_yticks()
        ax.set_yticklabels([f"{x:,.0%}" for x in vals])

    g.set_axis_labels(
        "Functional Domain",
        "← Higher in Rust      (Share Difference)      Higher in C →",
    )
    g.set_titles(col_template="{col_name}")
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle(
        'The "Language Tax": Difference in Commit Focus (C vs Rust)\n'
        "Feature vs Maintenance"
    )

    g.fig.savefig("diverging_distribution_delta.png", dpi=300)
    print("✅ Saved diverging_distribution_delta.png")

def load_source_metadata(base_dir: str) -> pd.DataFrame:
    """Loads entropy and churn from the original source CSVs."""
    all_rows = []
    # Walk through c/ and rust/ folders
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".csv"):
                path = os.path.join(root, file)
                try:
                    # Only read what we need
                    df = pd.read_csv(path, usecols=['commit_id', 'entropy', 'churn'])
                    # Add repo name from filename for safer merging
                    df['repo'] = file.replace('full_commit_', '').replace('.csv', '')
                    all_rows.append(df)
                except Exception as e:
                    print(f"Skipping {file}: {e}")
                    
    if not all_rows: return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)

def calculate_full_ccs(predictions_df: pd.DataFrame) -> pd.DataFrame:
    print("Loading source metadata for CCS calculation...")
    source_df = load_source_metadata(CSV_SOURCE_DIR)
    
    if source_df.empty:
        print("❌ Could not load source CSVs. Skipping CCS.")
        return predictions_df

    # Merge predictions with source data
    # We use both commit_id and repo to ensure unique matches
    merged_df = pd.merge(predictions_df, source_df, on=['commit_id', 'repo'], how='inner')
    
    print(f"Merged {len(merged_df)} records for CCS calculation.")

    # Apply the Formula
    # CCS = (W_COG * comp) + (W_ENT * log10(files)) + (W_CHURN * log10(lines))
    def get_score(row):
        try:
            comp = float(row['complexity'])
            # Log scale for git stats to dampen outliers
            ent = math.log10(max(1, float(row['entropy']))) 
            churn = math.log10(max(1, float(row['churn'])))
            
            return (W_COG * comp) + (W_ENT * ent) + (W_CHURN * churn)
        except:
            return None

    merged_df['ccs_score'] = merged_df.apply(get_score, axis=1)
    return merged_df

def plot_ccs_violin(df: pd.DataFrame):
    """Visualizes the Complexity Gap."""
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, hue='language', y='ccs_score', inner="quartile")
    plt.title('Distribution of Cognitive Complexity Score (CCS) by Language')
    plt.ylabel('CCS (Log-Scaled Metric)')
    plt.xlabel('Language')
    plt.tight_layout()
    plt.savefig("visualizations/css_by_language.png")
    plt.show()

def normalize_category(raw_cat: Any) -> str:
    """
    Maps any noisy category string to one of the Canonical 5.
    """
    cat = str(raw_cat).lower().strip()

    # 1. Exact match
    for valid in VALID_CATEGORIES:
        if cat == valid.lower():
            return valid

    # 2. Heuristic mapping (order matters)
    if any(x in cat for x in ["memory", "segfault", "overflow", "safety", "security", "vuln", "crash"]):
        return "Memory Safety & Robustness"

    if any(x in cat for x in ["concurren", "thread", "atomic", "race", "lock"]):
        return "Concurrency & Thread Safety"

    if any(x in cat for x in ["feature", "value", "perf", "optimiz", "platform", "porting"]):
        return "Feature & Value Add"

    if any(x in cat for x in ["build", "refactor", "doc", "test", "tool", "ci", "style", "lint"]):
        return "Build, Refactor & Internal"

    # Fallback
    return "Logic & Correctness"

def parse_key(key: str) -> Dict[str, str]:
    """
    Parse 'key' like 'c_libcurl_<commit>' -> language, repo, commit_id.
    """
    parts = key.split("_", 2)
    if len(parts) < 3:
        return {"language": "unknown", "repo": "unknown", "commit_id": key}
    language, repo, commit_id = parts
    return {"language": language, "repo": repo, "commit_id": commit_id}

def process_files() -> pd.DataFrame:
    all_data: List[Dict[str, Any]] = []

    files = glob.glob(os.path.join(INPUT_DIR, "*.jsonl"))
    print(f"Found {len(files)} prediction files to process...")

    for filepath in files:
        print(f"Processing: {os.path.basename(filepath)}")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Extract metadata from key
                meta = parse_key(data.get("key", ""))

                # Extract model JSON from response.candidates[0].content.parts[0].text
                text = (
                    data.get("response", {})
                    .get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )

                # Strip optional markdown fences if present
                if isinstance(text, str):
                    text = text.replace("```json", "").replace("```", "").strip()

                try:
                    pred_json = json.loads(text)
                except Exception:
                    continue

                clean_cat = normalize_category(pred_json.get("cat", "Logic & Correctness"))

                is_feat = str(pred_json.get("feat", False)).lower() in ["true", "1"]
                is_sec = str(pred_json.get("sec", False)).lower() in ["true", "1"]

                try:
                    comp = int(pred_json.get("comp", 1))
                    comp = max(1, min(5, comp))
                except Exception:
                    comp = 1

                reasoning = pred_json.get("reas", "")

                all_data.append(
                    {
                        "language": meta["language"],
                        "repo": meta["repo"],
                        "commit_id": meta["commit_id"],
                        "category": clean_cat,
                        "is_feature": is_feat,
                        "is_security": is_sec,
                        "complexity": comp,
                        "reasoning": reasoning,
                    }
                )

    df = pd.DataFrame(all_data)
    return df

def plot_category_distribution_by_language(df: pd.DataFrame) -> None:
    if df.empty:
        print("No data for plotting by language.")
        return

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="category", hue="language")
    plt.title("Commit Category Distribution by Language")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def plot_category_distribution_by_repo(df: pd.DataFrame) -> None:
    if df.empty:
        print("No data for plotting by repo.")
        return

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="repo", hue="category")
    plt.title("Commit Category Distribution by Library/Repo")
    plt.xlabel("Library / Repo")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def plot_normalized_by_language(df: pd.DataFrame) -> None:
    """
    Normalize category distribution by commit count per language
    (each row is one commit).
    """
    if df.empty:
        print("No data for normalized language plot.")
        return

    counts = (
        df.groupby(["language", "category"])
        .size()
        .reset_index(name="count")
    )
    counts["share"] = counts["count"] / counts.groupby("language")["count"].transform("sum")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=counts,
        x="language",
        y="share",
        hue="category",
    )
    
    plt.title("Normalized Category Distribution by Language\n(share of commits per language)")
    plt.ylabel("Share of Commits")
    plt.xlabel("Language")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

def plot_normalized_by_repo(df: pd.DataFrame) -> None:
    """
    Normalize category distribution by commit count per (language, repo),
    so C coreutils and Rust coreutils stay distinct.
    """
    if df.empty:
        print("No data for normalized repo plot.")
        return

    # Treat language+repo as a distinct library label
    df = df.copy()
    df["lib"] = df["language"] + " / " + df["repo"]

    counts = df.groupby(["lib", "category"]).size().reset_index(name="count")
    counts["share"] = counts["count"] / counts.groupby("lib")["count"].transform("sum")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=counts,
        x="lib",
        y="share",
        hue="category",
    )
    plt.title("Normalized Category Distribution by Library/Repo\n(share of commits per language-repo)")
    plt.ylabel("Share of Commits")
    plt.xlabel("Language / Library")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig("visualizations/normalized_category_distribution_by_lib.png")
    plt.show()

def plot_CCS_for_cat_by_language(df: pd.DataFrame) -> None:
    """
    Average CCS by category and language, split into two facets:
    one for Feature work and one for Maintenance categories.
    """
    if df.empty:
        print("No data for CCS plot.")
        return

    tmp = df.copy()
    tmp["work_type"] = tmp["category"].apply(
        lambda c: "Feature" if c == "Feature & Value Add" else "Maintenance"
    )

    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=tmp,
        x="category",
        y="ccs_score",
        hue="language",
        col="work_type",
        kind="bar",
        estimator="mean",
        errorbar=None,
        palette={"c": "#4C72B0", "rust": "#DD8452"},
        height=6,
        aspect=1.1,
    )

    g.set_xticklabels(rotation=45, ha="right")
    g.set_axis_labels("Category", "Avg Composite Complexity Score (0-100)")
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Average CCS by Category and Language\nFeature vs Maintenance")

    g.fig.savefig("visualizations/avg_ccs_by_category_faceted.png")
    plt.show()
    

if __name__ == "__main__":
    

    df = process_files()
    plot_css = calculate_full_ccs(df)
    print(f"\nLoaded {len(df)} cleaned prediction rows.")

    print("\n--- Final Cleaned Distribution (language, category) ---")
    if not df.empty:
        print(
            df.groupby(["language", "category"])
            .size()
            .reset_index(name="count")
            .to_string(index=False)
        )
    else:
        print("No rows.")

    # Save cleaned dataset 
    #plot_css.to_csv(CSS_PREDICTION,index=False)
    #df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n(CSV saving disabled) Would save to {OUTPUT_FILE}")

    # Plots (not saved, just shown)
    #plot_ccs_violin(plot_css)
    #plot_CCS_for_cat_by_language(plot_css)
    #plot_diverging_delta(plot_css)

    #plot_normalized_by_language(df)
    plot_normalized_by_repo(df)
