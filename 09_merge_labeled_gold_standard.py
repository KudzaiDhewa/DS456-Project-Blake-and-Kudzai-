from pathlib import Path

import pandas as pd


SRC_LIBXML2 = Path("gold_standard_sample_labeled/c/labeled_gold_standard_libxml2.csv")
SRC_QUICKXML = Path("gold_standard_sample_labeled/rust/labeled_gold_standard_quick-xml.csv")
OUT_MERGED = Path("gold_standard_sample_labeled/labeled_gold_standard_merged.csv")


def main() -> None:
    parts = []

    if SRC_LIBXML2.exists():
        df_c = pd.read_csv(SRC_LIBXML2)
        df_c["language_inferred"] = "C"
        parts.append(df_c)
    else:
        print(f"Missing source: {SRC_LIBXML2}")

    if SRC_QUICKXML.exists():
        df_r = pd.read_csv(SRC_QUICKXML)
        df_r["language_inferred"] = "Rust"
        parts.append(df_r)
    else:
        print(f"Missing source: {SRC_QUICKXML}")

    if not parts:
        print("No input CSVs found, nothing to merge.")
        return

    merged = pd.concat(parts, ignore_index=True)
    merged.to_csv(OUT_MERGED, index=False)
    print(f"Merged {len(merged)} rows into {OUT_MERGED}")


if __name__ == "__main__":
    main()

