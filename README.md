### Open-Source Maintenance Analysis: C vs Rust
## Overview

This repository contains the full data pipeline, modeling workflow, and analysis used to study maintenance burden in open-source libraries, with a focus on comparing C and Rust projects.

The project uses Git commit data, commit diffs, and LLM-based classification to distinguish between maintenance and feature work, assign maintenance categories, and estimate maintenance burden using a complexity score.

The final outputs include cleaned datasets, model predictions, comparisons to a human-labeled gold standard, and publication-ready visualizations.

.
├── FPmilestones/              # Project milestones and checkpoints
├── KD/                        # commit analysis by Kudzai 
├── final-narrative/           # Final written narrative (Quarto / report files)
├── visualizations/            # Figures and plots used in the final report
│
├── 00_clone_repo.py           # Clone and initialize target GitHub repositories
├── 01_data_prepare.py         # Clean and preprocess raw commit data
├── 02_jsonl_uploader.py       # Convert data to JSONL format for LLM inference
├── 03_model_training.py       # Train / configure classification model
├── 04_model_evaluation.py     # Evaluate model performance
├── 07_libxml2_label_summary.py# Label summary for libxml2 case study
├── 08_add_synthetic_commit_ids.py # Add synthetic IDs for tracking commits
├── 09_merge_labeled_gold_standard.py # Merge LLM labels with gold standard
├── 10_inference_visuals.py    # Generate inference and time-series visualizations
│
├── FINAL_CSS_WITH_PREDICTION_TIMESERIES.csv
├── GOLD_VS_MODEL_COMPARISON.csv
│
├── prompt_config.py           # Prompt templates and configuration for LLM
├── requirements.txt           # Python dependencies
├── superset.txt               # Superset of labels / categories
├── README.md                  # Project documentation
└── .gitignore


### Data Description

Each observation corresponds to a Git commit, with key variables including:

| Variable         | Description                                              | Type        |
| ---------------- | -------------------------------------------------------- | ----------- |
| `commit_diff`    | Summary of code changes (lines added, removed, modified) | Text        |
| `commit_message` | Commit message written by the contributor                | Text        |
| `timestamp`      | Date and time the commit was authored                    | Datetime    |
| `author`         | Commit author identifier                                 | Categorical |
| `library`        | Open-source library name                                 | Categorical |
| `language`       | Programming language (C or Rust)                         | Categorical |

### Key Outputs

Time-series analysis of maintenance vs feature work

Category-level breakdowns of maintenance effort

Gold standard vs model comparison

Visual evidence supporting differences in maintenance burden between C and Rust

Key result files:

FINAL_CSS_WITH_PREDICTION_TIMESERIES.csv - this data set includes all the LLM classified commits. This is the main dataset we used for our analysis

GOLD_VS_MODEL_COMPARISON.csv

### Requirements

Install dependencies with:
pip install -r requirements.txt


### Running the Pipeline

Recommended execution order:
python 00_clone_repo.py
python 01_data_prepare.py
python 02_jsonl_uploader.py
python 03_model_training.py
python 04_model_evaluation.py
python 08_add_synthetic_commit_ids.py
python 09_merge_labeled_gold_standard.py
python 10_inference_visuals.py


### Authors 
Blake Kell
Kudzai Dhewa

### License
This project is for academic and research purposes.
