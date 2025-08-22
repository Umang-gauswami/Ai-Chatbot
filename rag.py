import os, pickle, faiss
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH  = os.path.join(DATA_DIR, "meta.pkl")

class RAGSearcher:
    def __init__(self):
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        self.rows: List[Dict] = meta["rows"]
        self.model = SentenceTransformer(meta["model_name"])
        self.index = faiss.read_index(INDEX_PATH)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[float, Dict]]:
        q = self.model.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(q, top_k)
        results = []
        for score, i in zip(scores[0], idxs[0]):
            if i == -1: 
                continue
            results.append((float(score), self.rows[int(i)]))
        return results