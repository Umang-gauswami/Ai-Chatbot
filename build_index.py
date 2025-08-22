import os
import faiss
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer

KB_CSV = os.path.join("kb", "faq.csv")
DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH = os.path.join(DATA_DIR, "meta.pkl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Read CSV and normalize column names
    df = pd.read_csv(KB_CSV)
    df.columns = df.columns.str.lower()  # make headers lowercase

    # Check required columns
    if not {"question", "answer"}.issubset(df.columns):
        raise ValueError("CSV must have 'question' and 'answer' columns.")

    # Load model
    model = SentenceTransformer(MODEL_NAME)

    # Convert questions into embeddings
    embeddings = model.encode(df["question"].tolist(), normalize_embeddings=True)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save metadata
    with open(META_PATH, "wb") as f:
        pickle.dump({
            "model_name": MODEL_NAME,
            "rows": df.to_dict(orient="records")
        }, f)

    # Save FAISS index
    faiss.write_index(index, INDEX_PATH)

    print(f"✅ Built index with {index.ntotal} entries.")
    print(f"📂 Saved index -> {INDEX_PATH}")
    print(f"📂 Saved metadata -> {META_PATH}")


if __name__ == "__main__":
    build()
