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
    """Load MS MARCO corpus + qrels (auto-downloads)"""
    print("🚀 Loading MS MARCO (50k passages)...")
    
    # Create local data dir
    os.makedirs('msmarco_data', exist_ok=True)
    
    # Check if data exists locally
    data_path = 'msmarco_data/data.pkl'
    if os.path.exists(data_path):
        print("Loading cached data...")
        with open(data_path, 'rb') as f:
            return pickle.load(f)
    
    # Corpus sample (50k passages)
    dataset = ir_datasets.load("msmarco-passage")
    corpus = {doc.doc_id: doc.text for doc in islice(dataset.docs_iter(), 50000)}
    print(f"Corpus: {len(corpus)} passages")
    
    # Train qrels
    train_ds = ir_datasets.load("msmarco-passage/train")
    train_qrels = defaultdict(set)
    for qrel in train_ds.qrels_iter():
        train_qrels[qrel.query_id].add(qrel.doc_id)
    
    # Dev qrels
    dev_ds = ir_datasets.load("msmarco-passage/dev")
    dev_qrels = defaultdict(set)
    for qrel in dev_ds.qrels_iter():
        dev_qrels[qrel.query_id].add(qrel.doc_id)
    
    print(f"Train qrels: {len(train_qrels)} | Dev: {len(dev_qrels)}")
    
    # Save locally
    data = {'corpus': corpus, 'train_qrels': dict(train_qrels), 'dev_qrels': dict(dev_qrels)}
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    print("✅ Data saved locally!")
    
    return data

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
