import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_CSV = "data/MMRDR_UWF_preprocessed_rgb/UWF.csv"
OUTPUT_DIR = "data/MMRDR_UWF_preprocessed_rgb"
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    df = pd.read_csv(INPUT_CSV)

    required_cols = ["image", "grade"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {INPUT_CSV}")

    df = df[["image", "grade"]].copy()

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["grade"],
        random_state=SEED
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(2 / 3),
        stratify=temp_df["grade"],
        random_state=SEED
    )

    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    val_path = os.path.join(OUTPUT_DIR, "val.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("===== SPLITS CREATED =====")
    print(f"Train: {len(train_df)} -> {train_path}")
    print(f"Val: {len(val_df)} -> {val_path}")
    print(f"Test: {len(test_df)} -> {test_path}")
    print(f"Total: {len(train_df) + len(val_df) + len(test_df)}")

    print("\nTrain distribution:")
    print(train_df["grade"].value_counts().sort_index())

    print("\nVal distribution:")
    print(val_df["grade"].value_counts().sort_index())

    print("\nTest distribution:")
    print(test_df["grade"].value_counts().sort_index())


if __name__ == "__main__":
    main()