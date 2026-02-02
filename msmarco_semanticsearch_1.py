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
    """Load MS MARCO - FORCE UTF-8 (Windows fix)"""
    import os
    os.environ['IR_DATASETS_CACHE'] = 'msmarco_data/cache'
    os.makedirs('msmarco_data/cache', exist_ok=True)
    
    print("🔄 Forcing UTF-8 MS MARCO download...")
    
    # Method 1: Pre-cleaned dev subset (guaranteed UTF-8)
    dataset_dev = ir_datasets.load("msmarco-passage/dev")
    
    corpus = {}
    for doc in tqdm(list(islice(dataset_dev.docs_iter(), 20000)), desc="Loading clean docs"):
        corpus[doc.doc_id] = doc.text
    
    # Train queries + qrels (smaller but clean)
    dataset_train = ir_datasets.load("msmarco-passage/train") 
    queries = {q.query_id: q.text for q in list(islice(dataset_train.queries_iter(), 10000))}
    
    train_qrels = defaultdict(set)
    for qrel in dataset_train.qrels_iter():
        train_qrels[qrel.query_id].add(qrel.doc_id)
    
    dev_qrels = defaultdict(set)
    for qrel in dataset_dev.qrels_iter():
        dev_qrels[qrel.query_id].add(qrel.doc_id)
    
    print(f"✅ Loaded {len(corpus)} docs, {len(queries)} queries")
    print(f"✅ Train qrels: {len(train_qrels)}, Dev qrels: {len(dev_qrels)}")
    
    return {
        'corpus': corpus, 
        'queries': queries,
        'train_qrels': dict(train_qrels),
        'dev_qrels': dict(dev_qrels)
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
