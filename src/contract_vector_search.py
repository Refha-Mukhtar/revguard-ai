"""
RevGuard AI — ChromaDB Semantic Contract Search
Provides embedding-based vector search across enterprise Master Service Agreements (MSAs).
100% offline-compatible using dense vectorized hashing embeddings.
"""

import os
import glob
from pypdf import PdfReader
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sklearn.feature_extraction.text import HashingVectorizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")
CONTRACTS_DIR = os.path.join(DATA_DIR, "contracts")

# 128-dimensional dense hashing vectorizer for zero-latency local embeddings
_vectorizer = HashingVectorizer(n_features=128, alternate_sign=False)

class LocalDenseEmbedding(EmbeddingFunction):
    """Zero-dependency, offline embedding function for local contract search."""
    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        return _vectorizer.transform(input).toarray().tolist()

_embedding_fn = LocalDenseEmbedding()

def get_chroma_collection():
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name="contract_clauses",
        embedding_function=_embedding_fn
    )

def extract_clauses_from_pdf(filepath: str) -> list[dict]:
    """Extract individual semantic clause paragraphs from a contract PDF."""
    reader = PdfReader(filepath)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    filename = os.path.basename(filepath)
    customer_id = filename.replace("_contract.pdf", "")

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    clauses = []
    
    # Bundle contract specifics into clean semantic chunks
    chunk_buffer = []
    for line in lines:
        if line.startswith("---") or line.startswith("MASTER"):
            continue
        chunk_buffer.append(line)
        if len(chunk_buffer) >= 2 or any(k in line for k in ["Agreed", "Rate", "SLA", "Tier"]):
            clause_text = " | ".join(chunk_buffer)
            clauses.append({
                "text": clause_text,
                "customer_id": customer_id,
                "source_file": filename
            })
            chunk_buffer = []

    if chunk_buffer:
        clauses.append({
            "text": " | ".join(chunk_buffer),
            "customer_id": customer_id,
            "source_file": filename
        })
    return clauses

def index_contracts(force_reindex: bool = False) -> int:
    """Index all contract PDFs into ChromaDB."""
    col = get_chroma_collection()
    if not force_reindex and col.count() > 0:
        return col.count()

    # Clear previous if force_reindex
    if force_reindex and col.count() > 0:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        client.delete_collection("contract_clauses")
        col = client.get_or_create_collection("contract_clauses", embedding_function=_embedding_fn)

    pdf_files = glob.glob(os.path.join(CONTRACTS_DIR, "*.pdf"))
    if not pdf_files:
        return 0

    all_docs = []
    all_metas = []
    all_ids = []

    for filepath in pdf_files:
        clauses = extract_clauses_from_pdf(filepath)
        for idx, clause in enumerate(clauses):
            clause_id = f"{clause['customer_id']}_c{idx}"
            all_ids.append(clause_id)
            all_docs.append(clause["text"])
            all_metas.append({
                "customer_id": clause["customer_id"],
                "source_file": clause["source_file"]
            })

    if all_docs:
        col.add(ids=all_ids, documents=all_docs, metadatas=all_metas)

    return col.count()

def search_contract_clauses(query: str, n_results: int = 4) -> list[dict]:
    """Semantic vector query across all indexed contract clauses."""
    col = get_chroma_collection()
    if col.count() == 0:
        index_contracts()

    if col.count() == 0:
        return []

    actual_n = min(n_results, col.count())
    results = col.query(query_texts=[query], n_results=actual_n)

    formatted = []
    if results and results.get("documents") and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

        for doc, meta, dist in zip(docs, metas, distances):
            # Convert L2 distance into intuitive 0-100% similarity score
            similarity_pct = max(0.0, round((1.0 - (dist / 2.0)) * 100, 1))
            formatted.append({
                "clause": doc,
                "customer_id": meta.get("customer_id", "UNKNOWN"),
                "source_file": meta.get("source_file", ""),
                "similarity_score": similarity_pct,
                "distance": round(dist, 3)
            })
    return formatted

def get_indexed_count() -> int:
    try:
        col = get_chroma_collection()
        return col.count()
    except Exception:
        return 0

if __name__ == "__main__":
    count = index_contracts(force_reindex=True)
    print(f"Indexed {count} contract clauses into ChromaDB.")
    test_res = search_contract_clauses("Cloud Compute unit rates and agreed discounts")
    print("\nSample Search Results:")
    for r in test_res:
        print(f"[{r['customer_id']}] (Sim: {r['similarity_score']}%) -> {r['clause']}")
