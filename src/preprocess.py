import ast
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def names(value, limit=None):
    try:
        items = ast.literal_eval(value) if isinstance(value, str) else []
        result = [str(x["name"]).strip().lower().replace(" ", "_") for x in items if isinstance(x, dict) and x.get("name")]
        return result[:limit] if limit else result
    except (ValueError, SyntaxError, TypeError):
        return []

def director(value):
    try:
        return [str(x["name"]).strip().lower().replace(" ", "_") for x in ast.literal_eval(value) if x.get("job") == "Director"][:1]
    except (ValueError, SyntaxError, TypeError):
        return []

def prepare():
    movies = pd.read_csv(DATA / "tmdb_5000_movies.csv")
    credits = pd.read_csv(DATA / "tmdb_5000_credits.csv")
    merged = movies.merge(credits[["movie_id", "cast", "crew"]], left_on="id", right_on="movie_id", how="inner", validate="one_to_one")
    merged = merged.drop_duplicates(subset="id").copy()
    merged["genres_list"] = merged["genres"].apply(names)
    merged["keywords_list"] = merged["keywords"].apply(names)
    merged["cast_list"] = merged["cast"].apply(lambda x: names(x, 3))
    merged["director_list"] = merged["crew"].apply(director)
    # Prefix features to distinguish the same word used in different fields.
    merged["features"] = merged.apply(lambda r: sorted(set(
        ["genre:"+x for x in r["genres_list"]] +
        ["keyword:"+x for x in r["keywords_list"]] +
        ["actor:"+x for x in r["cast_list"]] +
        ["director:"+x for x in r["director_list"]])), axis=1)
    output = merged[["id", "title", "overview", "vote_average", "vote_count", "features"]].copy()
    output["overview"] = output["overview"].fillna("")
    output["features"] = output["features"].apply(lambda x: "|".join(x))
    output.to_csv(DATA / "movies_cleaned.csv", index=False)
    print(f"Prepared {len(output)} movies; output: {DATA / 'movies_cleaned.csv'}")
    return output

if __name__ == "__main__":
    prepare()
