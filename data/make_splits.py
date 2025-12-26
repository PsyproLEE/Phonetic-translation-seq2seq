import pandas as pd
import os
from sklearn.model_selection import train_test_split


def make_splits(
    input_csv,
    train_path,
    val_path,
    val_ratio=0.1,
    seed=42
):
    df = pd.read_csv(input_csv)

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed
    )

    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print("✅ Train / Val split completed")
    print(f"   Train samples : {len(train_df)}")
    print(f"   Val samples   : {len(val_df)}")


if __name__ == "__main__":
    make_splits(
        input_csv="../data/processed/augmented_dataset.csv",
        train_path="../data/splits/train.csv",
        val_path="../data/splits/val.csv"
    )
