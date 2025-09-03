import argparse
import pickle
import numpy as np
import faiss
from typing import Any, Dict, Iterable, List, Tuple, Set

from datasets import load_dataset


# ----------------------------
# IO + normalization helpers
# ----------------------------

def load_pickle_as_tupledict(path: str) -> Dict[Tuple[Any, str], Iterable[float]]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Pickle at {path} must be a dict mapping (id, text) -> embedding.")
    sample_key = next(iter(obj.keys()))
    if not (isinstance(sample_key, tuple) and len(sample_key) == 2):
        raise ValueError(
            "Dict keys must be (id, text) tuples, e.g., (123, 'abstract text'). "
            f"Got key type: {type(sample_key)} / value: {sample_key}"
        )
    return obj


def to_unit_float32_matrix(vectors: List[Iterable[float]]) -> np.ndarray:
    X = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms


def unpack_docs(doc_dict: Dict[Tuple[Any, str], Iterable[float]]) -> Tuple[np.ndarray, List[Any], List[str]]:
    doc_items = list(doc_dict.items())
    doc_ids    = [k[0] for k, _ in doc_items]
    abstracts  = [k[1] for k, _ in doc_items]
    X_docs     = to_unit_float32_matrix([v for _, v in doc_items])
    return X_docs, doc_ids, abstracts


def unpack_claims(claim_dict: Dict[Tuple[Any, str], Iterable[float]]) -> Tuple[np.ndarray, List[Any], List[str]]:
    cl_items     = list(claim_dict.items())
    claim_ids    = [k[0] for k, _ in cl_items]
    claim_texts  = [k[1] for k, _ in cl_items]
    X_claims     = to_unit_float32_matrix([v for _, v in cl_items])
    return X_claims, claim_ids, claim_texts


# ----------------------------
# Gold labels from SciFact
# ----------------------------

def load_scifact_gold_claim2docs(split: str = "train") -> Dict[Any, Set[Any]]:
    """
    Build claim_id -> set(doc_id) from HuggingFace 'allenai/scifact' (claims).
    Primary source: 'cited_doc_ids' (sequence of int32).
    Fallback: 'evidence_doc_id' (string; often '' when absent).
    """
    ds = load_dataset("allenai/scifact", "claims")[split]
    claim2docs: Dict[Any, Set[Any]] = {}
    for row in ds:
        cid = row["id"]
        gold: Set[Any] = set()

        cited = row.get("cited_doc_ids", None)
        if cited:
            for d in cited:
                try:
                    gold.add(int(d))
                except Exception:
                    gold.add(d)

        ev_doc = row.get("evidence_doc_id", "")
        if isinstance(ev_doc, str) and ev_doc.strip():
            try:
                gold.add(int(ev_doc))
            except Exception:
                gold.add(ev_doc)

        claim2docs[cid] = gold
    return claim2docs


# ----------------------------
# FAISS (cosine via inner product)
# ----------------------------

def build_faiss_ip_index(X: np.ndarray) -> faiss.Index:
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(X)
    index.add(X)
    return index


# ----------------------------
# Ranking metrics
# ----------------------------

def rr_at_k(ranks: List[int], k: int) -> float:
    vals = [1.0 / (r + 1) if (r >= 0 and r < k) else 0.0 for r in ranks]
    return float(np.mean(vals)) if vals else 0.0


def ap_at_k(relevant_ranks: List[List[int]], k: int) -> float:
    ap_vals = []
    for ranks in relevant_ranks:
        ranks = sorted([r for r in ranks if r < k])
        if not ranks:
            ap_vals.append(0.0)
            continue
        hits = 0
        precs = []
        for r in ranks:
            hits += 1
            precs.append(hits / (r + 1))
        ap_vals.append(float(np.mean(precs)))
    return float(np.mean(ap_vals)) if ap_vals else 0.0


# ----------------------------
# Evaluation
# ----------------------------

def evaluate(
    index: faiss.Index,
    X_claims: np.ndarray,
    claim_ids: List[Any],
    gold_map: Dict[Any, Set[Any]],
    doc_ids: List[Any],
    ks=(1, 10, 50),
    batch: int = 256
) -> Dict[str, float]:
    id2pos = {d: i for i, d in enumerate(doc_ids)}
    max_k = max(ks)
    rr_first_hit: List[int] = []
    all_rel_ranks: List[List[int]] = []

    for i in range(0, X_claims.shape[0], batch):
        xb = X_claims[i:i+batch].copy()
        faiss.normalize_L2(xb)
        _, I = index.search(xb, max_k)
        for row, cid in zip(I, claim_ids[i:i+batch]):
            rel_docs = gold_map.get(cid, set())
            ranks = []
            first = -1
            if rel_docs:
                rel_positions = {id2pos[d] for d in rel_docs if d in id2pos}
                for j, idx in enumerate(row):
                    if int(idx) in rel_positions:
                        ranks.append(j)
                        if first == -1:
                            first = j
            rr_first_hit.append(first)
            all_rel_ranks.append(ranks)

    metrics = {}
    for k in ks:
        metrics[f"MRR@{k}"] = rr_at_k(rr_first_hit, k)
        metrics[f"MAP@{k}"] = ap_at_k(all_rel_ranks, k)
    return metrics


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_pkl", type=str, default="./data/documents.pkl")
    parser.add_argument("--claim_pkl", type=str, default="./data/claims.pkl")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--k", type=int, nargs="+", default=[1, 10, 50])
    args = parser.parse_args()

    doc_dict   = load_pickle_as_tupledict(args.doc_pkl)
    claim_dict = load_pickle_as_tupledict(args.claim_pkl)

    X_docs, doc_ids, _          = unpack_docs(doc_dict)
    X_claims, claim_ids, _texts = unpack_claims(claim_dict)

    if X_docs.shape[1] != X_claims.shape[1]:
        raise ValueError(f"Dim mismatch: docs d={X_docs.shape[1]} vs claims d={X_claims.shape[1]}")

    claim2docs = load_scifact_gold_claim2docs(split=args.split)

    index = build_faiss_ip_index(X_docs)
    metrics = evaluate(index, X_claims, claim_ids, claim2docs, doc_ids, ks=tuple(args.k))

    print("\n=== FAISS (OpenAI embeddings) on SciFact ===")
    for k in args.k:
        print(f"MRR@{k}: {metrics[f'MRR@{k}']:.4f}   MAP@{k}: {metrics[f'MAP@{k}']:.4f}")

    row = "OpenAI Embeddings & " + " & ".join(
        f"{metrics[f'MRR@{k}']:.4f} & {metrics[f'MAP@{k}']:.4f}" for k in [1, 10, 50]
    ) + " \\\\"
    print("\nLaTeX row:")
    print(row)


if __name__ == "__main__":
    main()