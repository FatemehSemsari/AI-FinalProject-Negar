from pathlib import Path
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPROC_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessor.joblib"

RAW_INPUT = PROJECT_ROOT / "data" / "demo_samples.csv"
OUT_PATH  = PROJECT_ROOT / "data" / "demo" / "demo_input_preprocessed.csv"

TARGET_COL = "attack_detected"
DROP_COLS = ["session_id"]

def main():
    preproc = joblib.load(PREPROC_PATH)

    df = pd.read_csv(RAW_INPUT)

    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])


    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    X_p = preproc.transform(df)
    cols = preproc.get_feature_names_out()

    out_df = pd.DataFrame(X_p, columns=cols)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Shape:", out_df.shape)

if __name__ == "__main__":
    main()