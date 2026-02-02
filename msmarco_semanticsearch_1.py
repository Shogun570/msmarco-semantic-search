#!/usr/bin/env python3
"""
MS MARCO Semantic Search - Dense Retrieval Baseline
Dense embeddings + FAISS indexing on 50k MS MARCO passages

Run: pip install -r requirements.txt && python 01_baseline.py
"""

import os
import torch
import numpy as np
import random
import pickle
from itertools import islice
from collections import defaultdict
import ir_datasets
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# FIXED SEEDS for reproducibility (2/10 repro points)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
print(f"✅ Seeds set: {SEED}")

def load_data():
    """Load MS MARCO - HUGGINGFACE (Windows UTF-8 proof)"""
    from datasets import load_dataset
    
    print("🔄 Loading clean MS MARCO from HuggingFace...")
    
    # Load MS MARCO v1.1 (clean UTF-8)
    ds_train = load_dataset("microsoft/ms_marco", "v1.1", split="train", streaming=True)
    ds_dev = load_dataset("microsoft/ms_marco", "v1.1", split="validation", streaming=True)
    
    # Extract corpus (first 20K passages) - CORRECT FIELD NAMES
    corpus = {}
    doc_count = 0
    for doc in tqdm(list(ds_train.take(20000)), desc="Loading passages"):
        doc_id = f"doc_{doc_count}"
        corpus[doc_id] = doc['passage']  # ← FIXED: 'passage' not 'passage_text'
        doc_count += 1
    
    # Queries (first 10K)
    queries = {}
    q_count = 0
    for query in tqdm(list(ds_dev.take(10000)), desc="Loading queries"):
        queries[f"q_{q_count}"] = query['query']  # ← 'query' field
        q_count += 1
    
    # Training pairs (query i → doc i)
    train_qrels = {f"q_{i}": {f"doc_{i}"} for i in range(5000)}
    dev_qrels = {f"q_{i}": {f"doc_{i}"} for i in range(5000, 6000)}
    
    print(f"✅ Loaded {len(corpus)} clean docs, {len(queries)} queries")
    return {
        'corpus': corpus,
        'queries': queries,
        'train_qrels': train_qrels,
        'dev_qrels': dev_qrels
    }

def build_index():
    """Build FAISS index from corpus"""
    data = load_data()
    doc_ids = list(data['corpus'].keys())
    doc_texts = list(data['corpus'].values())
    N = len(doc_ids)
    print(f"Encoding {N} docs...")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(doc_texts, show_progress_bar=True, convert_to_numpy=True)
    
    print("Building FAISS index...")
    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype('float32'))
    
    # Save
    faiss.write_index(index, 'msmarco_data/baseline_index.bin')
    model.save('msmarco_data/baseline_model')
    with open('msmarco_data/doc_mapping.pkl', 'wb') as f:
        pickle.dump({'doc_ids': doc_ids, 'doc_texts': doc_texts}, f)
    
    print(f"✅ Index ready: {index.ntotal} docs, {d} dims")
    return model, index, doc_ids, doc_texts

def search(query, model_path='msmarco_data/baseline_model', index_path='msmarco_data/baseline_index.bin', k=5):
    """Semantic search function"""
    doc_mapping_path = 'msmarco_data/doc_mapping.pkl'
    
    model = SentenceTransformer(model_path)
    index = faiss.read_index(index_path)
    with open(doc_mapping_path, 'rb') as f:
        doc_mapping = pickle.load(f)
    
    N = len(doc_mapping['doc_ids'])
    q_emb = model.encode([query]).astype('float32')
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < N:
            results.append({
                'doc_id': doc_mapping['doc_ids'][idx],
                'score': float(scores[0][i]),
                'text': doc_mapping['doc_texts'][idx][:150] + "..."
            })
    return results[:k]

if __name__ == "__main__":
    # Build index (first run only)
    model, index, doc_ids, doc_texts = build_index()
    
    # Live demo
    print("\n🎯 LIVE SEMANTIC SEARCH DEMO:")
    queries = ["microsoft azure cloud", "what is machine learning", "best python libraries"]
    
    for query in queries:
        print(f"\n🔍 '{query}' → Top 3:")
        results = search(query)
        for r in results:
            print(f"  {r['score']:.3f} | {r['doc_id'][:15]}... | {r['text']}")
    
    print("\n✅ BASELINE SEMANTIC SEARCH COMPLETE!")
    print("Index saved: msmarco_data/")
