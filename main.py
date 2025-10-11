# main.py
# libraries import for argument parsing, data handling, numerical operations, and text similarity analysis
import argparse
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# path to the product dataset csv file
DATA_PATH = "products.csv"

# load and preprocess the product dataset
def load_df(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path) # reading the csv into dataframe
    df["tags"] = df.get("tags", "").fillna("").astype(str)
    df["tags_set"] = df["tags"].apply(lambda s: set(t.strip().lower() for t in s.split(";") if t.strip()))
    df["description"] = df.get("description", "").fillna("").astype(str).str.lower()
    return df

# compute jaccard similarity between two sets
def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b) # size of intersection
    union = len(a | b) # size of unnion
    return float(inter) / union if union else 0.0

# building TF - IDF matrix from product descriptions
def build_tfidf(descriptions, max_features: int = 1000):
    vect = TfidfVectorizer(stop_words="english", max_features=max_features)
    tfidf = vect.fit_transform(descriptions)
    return vect, tfidf

# generate top - N product recommendation based on tags and description similarity
def recommend(df: pd.DataFrame, tfidf, product_id: int, top_n: int = 5,
              w_tag: float = 0.6, w_desc: float = 0.4) -> List[Dict]:
    if product_id not in df["product_id"].values:
        raise ValueError(f"product_id {product_id} not found") # validate product ID
    idx = int(df.index[df["product_id"] == product_id][0]) # get the index of target product
    desc_sims = linear_kernel(tfidf[idx:idx+1], tfidf).flatten() # compute cosine similarity
    target_tags = df.at[idx, "tags_set"] # get tag set of target product
    tag_sims = df["tags_set"].apply(lambda s: jaccard(target_tags, s)).values # compute jaccard similarity for tags
    combined = w_tag * tag_sims + w_desc * desc_sims # weighted combination of similarities
    combined[idx] = -1.0 # exclude the target product itself
    top_idx = np.argsort(combined)[::-1][:top_n] # get indices of top - N similar products
    results = []
    for k in top_idx:
        results.append({
            "product_id": int(df.iloc[k]["product_id"]),
            "name": df.iloc[k]["name"],
            "category": df.iloc[k].get("category", ""),
            "score": float(combined[k]),
            "jaccard": float(tag_sims[k]),
            "desc_sim": float(desc_sims[k]),
        })
    return results

# command line interface for running the recommender
def cli_main():
    parser = argparse.ArgumentParser(description="Smart_Cart minimal recommender")
    parser.add_argument("--product", type=int, required=True, help="seed product_id")
    parser.add_argument("--top", type=int, default=5, help="top N recommendations")
    parser.add_argument("--w_tag", type=float, default=0.6, help="tag weight")
    parser.add_argument("--w_desc", type=float, default=0.4, help="description weight")
    args = parser.parse_args()

    df = load_df() # load and preprocess dataset
    vect, tfidf = build_tfidf(df["description"].tolist()) # building IT - IDF matrix
    recs = recommend(df, tfidf, args.product, top_n=args.top, w_tag=args.w_tag, w_desc=args.w_desc) # gets recommendations

    # Display recommendations
    print(f"Recommendations for product {args.product}:")
    for r in recs:
        print(f"- {r['product_id']}: {r['name']} ({r['category']})  score={r['score']:.4f} jaccard={r['jaccard']:.3f} desc={r['desc_sim']:.3f}")

# entry point for cli execution
if __name__ == "__main__":
    cli_main()
    