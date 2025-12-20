import pandas as pd


CSV_PATH = "gold_standard_sample_labeled/rust/labeled_gold_standard_quick-xml.csv"


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    
    # 1) Count of label_category
    print("=== label_category counts ===")
    print(df["label_category"].value_counts(dropna=False))

    # 2) For each label_category, list label_reasoning values
    print("\n=== label_reasoning by label_category ===")
    grouped = df.groupby("label_category")["label_reasoning"].apply(list)

    for category, reasons in grouped.items():
        total = len(reasons)
        missing = sum(
            pd.isna(r) or (isinstance(r, str) and r.strip() == "")
            for r in reasons
        )

        print(f"\n--- {category} --- (total: {total}, NaN/missing: {missing})")
        for reason in reasons:
            if pd.isna(reason) or (isinstance(reason, str) and reason.strip() == ""):
                print("- [NaN / missing reasoning]")
            else:
                print(f"- {reason}")


if __name__ == "__main__":
    main()
