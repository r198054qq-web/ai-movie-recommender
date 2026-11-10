import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "movies_cleaned.csv"

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0

def load_movies():
    df = pd.read_csv(DATA).fillna({"features": ""})
    df["feature_set"] = df["features"].apply(lambda s: set(str(s).split("|")) if s else set())
    return df

def recommend(title, top_k=5):
    df = load_movies()
    matches = df[df["title"].str.casefold() == title.casefold()]
    if matches.empty:
        matches = df[df["title"].str.contains(title, case=False, regex=False, na=False)]
    if matches.empty:
        raise ValueError(f"Movie not found: {title}")
    target = matches.iloc[0]
    scores = [(jaccard(target["feature_set"], row["feature_set"]), row["title"])
              for _, row in df.iterrows() if row["id"] != target["id"]]
    return sorted(scores, key=lambda x: (-x[0], x[1]))[:top_k]

if __name__ == "__main__":
    for score, title in recommend("Interstellar"):
        print(f"{title}: {score:.3f}")
