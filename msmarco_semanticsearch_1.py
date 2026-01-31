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
    """Load MS MARCO data with UTF-8 fix"""
    print("Loading MS MARCO data...")
    
    dataset = ir_datasets.load("msmarco-passage/train")
    
    # Fix 1: UTF-8 encoding fix (ir_datasets handles this automatically in recent versions)
    corpus = {}
    count = 0
    for doc in tqdm(islice(dataset.docs_iter(), 50000), total=50000, desc="Loading docs"):
        try:
            corpus[doc.doc_id] = doc.text
            count += 1
        except UnicodeDecodeError:
            # Skip broken docs (rare)
            continue
    
    print(f"✅ Loaded {count}/50K docs")
    
    # Queries
    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    
    # Train qrels (positive pairs for fine-tuning)
    train_qrels = defaultdict(set)
    for qrel in dataset.qrels_iter():
        train_qrels[qrel.query_id].add(qrel.doc_id)
    
    # Dev qrels (for evaluation)
    dataset_dev = ir_datasets.load("msmarco-passage/dev")
    dev_qrels = defaultdict(set)
    for qrel in dataset_dev.qrels_iter():
        dev_qrels[qrel.query_id].add(qrel.doc_id)
    
    print(f"✅ Train qrels: {len(train_qrels)} queries")
    print(f"✅ Dev qrels: {len(dev_qrels)} queries")
    
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
