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
    ds = load_dataset("allenai/scifact", "claims")
    claim2docs: Dict[Any, Set[Any]] = {}
    for row in ds[split]:
        cid = row["id"]
        ev  = row.get("evidence", [])
        S: Set[Any] = set()
        for e in ev:
            if isinstance(e, dict) and "doc_id" in e:
                S.add(e["doc_id"])
            elif isinstance(e, (list, tuple)) and len(e) >= 1:
                S.add(e[0])
        claim2docs[cid] = S
    return claim2docs


# ----------------------------
# FAISS (cosine via inner product)
# ----------------------------

def build_faiss_ip_index(X: np.ndarray) -> faiss.Index:
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(X)  # defensive: normalize in-place before adding
    index.add(X)
    return index


# ----------------------------
# Ranking metrics
# ----------------------------

def rr_at_k(ranks: List[int], k: int) -> float:
    vals = []
    for r in ranks:
        vals.append(1.0 / (r + 1) if (r >= 0 and r < k) else 0.0)
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
        _, I = index.search(xb, max_k)  # indices into doc_ids
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
# Debug helpers
# ----------------------------

def coerce_ids_to_type(xs: List[Any], to_type: str) -> List[Any]:
    if to_type == "none":
        return xs
    out = []
    for x in xs:
        try:
            out.append(str(x) if to_type == "str" else int(x))
        except Exception:
            out.append(x)  # keep original if cast fails
    return out


def coerce_gold_map_types(gold: Dict[Any, Set[Any]], to_type: str) -> Dict[Any, Set[Any]]:
    if to_type == "none":
        return gold
    new = {}
    for k, vs in gold.items():
        try:
            nk = str(k) if to_type == "str" else int(k)
        except Exception:
            nk = k
        nset = set()
        for v in vs:
            try:
                nset.add(str(v) if to_type == "str" else int(v))
            except Exception:
                nset.add(v)
        new[nk] = nset
    return new


def debug_print_overlap(doc_ids: List[Any], claim_ids: List[Any], gold_map: Dict[Any, Set[Any]], X_docs: np.ndarray, X_claims: np.ndarray):
    gold_ids_all = set().union(*gold_map.values()) if len(gold_map) else set()
    doc_ids_set  = set(doc_ids)

    print("\n[DEBUG] Basic stats")
    print(f"  #docs indexed:  {len(doc_ids)}   (#unique: {len(doc_ids_set)})")
    print(f"  #claims:        {len(claim_ids)}")
    print(f"  embed dims:     docs d={X_docs.shape[1]}  claims d={X_claims.shape[1]}")
    print(f"  #gold doc ids:  {len(gold_ids_all)}")

    overlap = gold_ids_all & doc_ids_set
    print(f"  overlap size:   {len(overlap)} (gold ∩ indexed doc ids)")
    print(f"  sample doc_ids: {list(doc_ids)[:5]}")
    any_cid = next(iter(gold_map.keys())) if gold_map else None
    if any_cid is not None:
        print(f"  sample claim id: {any_cid}, gold set (up to 5): {list(gold_map[any_cid])[:5]}")


def debug_peek_neighbors(index: faiss.Index, X_claims: np.ndarray, doc_ids: List[Any], gold_map: Dict[Any, Set[Any]], sample_n: int = 3, k: int = 5):
    print("\n[DEBUG] Neighbor peek (first few claims)")
    qn = min(sample_n, X_claims.shape[0])
    xb = X_claims[:qn].copy()
    faiss.normalize_L2(xb)
    _, I = index.search(xb, k)
    for qi in range(qn):
        topk = [doc_ids[j] for j in I[qi]]
        # pick the qi-th claim id in the insertion order
        cid = list(gold_map.keys())[qi] if gold_map else None
        gold = gold_map.get(cid, set()) if cid is not None else set()
        inter = [d for d in topk if d in gold]
        print(f"  claim[{qi}] id={cid}  top{k}={topk}")
        print(f"    gold={list(gold)[:8]}")
        print(f"    intersection(top{k}, gold)={inter}")


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_pkl", type=str, default="./data/documents.pkl")
    parser.add_argument("--claim_pkl", type=str, default="./data/claims.pkl")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--k", type=int, nargs="+", default=[1, 10, 50])
    parser.add_argument("--coerce_id_type", type=str, default="none", choices=["none", "str", "int"],
                        help="Coerce both doc_ids and gold ids to a common type to fix type mismatches.")
    parser.add_argument("--strict_overlap", action="store_true",
                        help="If set, raise an error when gold∩doc_ids overlap is zero.")
    parser.add_argument("--neighbor_peek", action="store_true",
                        help="If set, prints top-5 neighbors for a few queries and their intersection with gold.")
    args = parser.parse_args()

    # Load pickles
    doc_dict   = load_pickle_as_tupledict(args.doc_pkl)
    claim_dict = load_pickle_as_tupledict(args.claim_pkl)

    X_docs, doc_ids, _          = unpack_docs(doc_dict)
    X_claims, claim_ids, _texts = unpack_claims(claim_dict)

    if X_docs.shape[1] != X_claims.shape[1]:
        raise ValueError(f"Dim mismatch: docs d={X_docs.shape[1]} vs claims d={X_claims.shape[1]}")

    # Gold mapping from SciFact
    claim2docs = load_scifact_gold_claim2docs(split=args.split)

    # Optional id type coercion (helpful when gold ids are int but pickled doc_ids are str, or vice versa)
    doc_ids = coerce_ids_to_type(doc_ids, args.coerce_id_type)
    claim2docs = coerce_gold_map_types(claim2docs, args.coerce_id_type)

    # Build index
    index = build_faiss_ip_index(X_docs)

    # --- Debug prints before evaluation
    debug_print_overlap(doc_ids, claim_ids, claim2docs, X_docs, X_claims)
    gold_ids_all = set().union(*claim2docs.values()) if len(claim2docs) else set()
    overlap = gold_ids_all & set(doc_ids)
    if args.strict_overlap and len(overlap) == 0:
        raise RuntimeError("No overlap between gold doc ids and indexed doc_ids. "
                           "Check id namespaces or use --coerce_id_type {str|int}.")

    if args.neighbor_peek:
        debug_peek_neighbors(index, X_claims, doc_ids, claim2docs, sample_n=3, k=5)

    # Evaluate
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